"""Stress-test clutter rejection using auxiliary range and Doppler projections."""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from radar_benchmark import (
    BACKGROUND_TRAIN, PFA, Case, Detectors, background, clutter_kernels,
    target_trials, wilson, write_csv,
)
from radar_demo import PRI, SPEED_OF_LIGHT, plt
from radar_scene import CARRIER, CHANNELS, DWELL_PULSES, RangeVelocityCell, View


DEFAULT_SEED = 20261123
BASELINE_COUNT = len(CHANNELS)
BASELINES = ("fixed_long_ppc", "fixed_short", "whitened", "late_adaptive")
AUXILIARY_OFFSETS = (-22.5, -15.0, -7.5, 7.5, 15.0, 22.5)
DIRECT = "range_doppler_amf"


@dataclass(frozen=True)
class Stress:
    name: str
    case: Case
    pulses: int = DWELL_PULSES
    # Range and velocity of the strong scatterer relative to the target
    # range, used only by the scene generator.
    background_shift: tuple[float, float] = (19.0, 1.15)
    off_grid: bool = False
    refresh_background: bool = False


def stress_cases():
    for velocity in (0, 1.95, 3.9, 7.8):
        yield Stress("edge", Case("edge", 100, velocity, 1, 0.38, edge=True))
    yield Stress("nominal", Case("low", 100, 7.8, 1, 0.38))
    for distance in (110, 130, 142, 170, 185, 205, 225, 250):
        yield Stress("transition", Case("transition", distance, 3.9, 1, 0.38))
    yield Stress("edge_off_grid", Case("edge", 100, 0, 1, 0.38, edge=True), off_grid=True)
    yield Stress("shifted_clutter", Case("edge", 100, 0, 1, 0.38, edge=True),
                 background_shift=(14.0, 0.75), off_grid=True)
    yield Stress("coincident_clutter", Case("edge", 100, 0, 1, 0.38, edge=True),
                 background_shift=(0.0, 0.0))
    for pulses in (128, 256):
        yield Stress("longer_dwell", Case("transition", 250, 3.9, 1, 0.38), pulses=pulses)
    yield Stress("far", Case("far", 4000, 7.8, 1, 0.38, far_budget=True))
    yield Stress("refreshed_clutter", Case("edge", 100, 0, 1, 0.38, edge=True),
                 background_shift=(14.0, 0.75), off_grid=True, refresh_background=True)


def auxiliary_views(distance, velocity, pulse_count, channels=CHANNELS[-2:]):
    velocity_step = SPEED_OF_LIGHT / (2 * CARRIER * PRI * pulse_count)
    coordinates = (
        (distance, velocity),
        *((distance + offset, velocity) for offset in AUXILIARY_OFFSETS),
        (distance, velocity - velocity_step),
        (distance, velocity + velocity_step),
    )
    return tuple(
        View(channel, range_m, speed, "rectangular")
        for range_m, speed in coordinates
        for channel in channels
    )


def noise_whitening(noise_covariance):
    """Remove duplicated projection directions and scale the rest to unit noise."""
    values, vectors = np.linalg.eigh(noise_covariance)
    keep = values > max(values[-1] * 1e-7, 1e-12)
    return vectors[:, keep].conj() / np.sqrt(values[keep])


class AdaptiveProjection:
    """Target-preserving adaptive filter in the reduced observation space."""

    def __init__(self, training, noise_covariance, signal_model):
        thermal = noise_whitening(noise_covariance)
        whitened_training = training @ thermal
        clutter_covariance = (
            whitened_training.T @ whitened_training.conj() / len(training)
        )
        values, vectors = np.linalg.eigh(clutter_covariance)
        # The calibrated thermal floor limits inversion in directions with
        # little estimated clutter. Small load damps uncertain clutter modes.
        values = np.maximum(0.97 * values + 0.03, 1.0)
        self.whitening = thermal @ (vectors.conj() / np.sqrt(values))
        whitened_signal = self.whitening.T @ signal_model
        basis, strength, _ = np.linalg.svd(whitened_signal, full_matrices=False)
        retained = strength > strength[0] * 1e-5
        self.projector = self.whitening @ basis[:, retained].conj()
        # Weak-signal score for a zero-mean signal with unknown subband phase.
        self.weights = (strength[retained] / strength[0]) ** 2

    def scores(self, samples):
        power = abs(samples @ self.projector) ** 2
        return power.sum(axis=1), power @ self.weights


def target_ensemble(rng, cell, stress, count):
    if not stress.off_grid:
        return target_trials(rng, cell, stress.case, count)
    offsets = ((-2.5, -0.45), (-2.5, 0.45), (0, 0), (2.5, -0.45), (2.5, 0.45))
    group = rng.integers(len(offsets), size=count)
    targets = np.empty((count, len(cell.channels)), dtype=complex)
    for index, (range_offset, velocity_offset) in enumerate(offsets):
        selected = group == index
        shifted = replace(stress.case, distance=stress.case.distance + range_offset,
                          velocity=stress.case.velocity + velocity_offset)
        targets[selected] = target_trials(rng, cell, shifted, np.count_nonzero(selected))
    return targets


def target_models(cell, case):
    direct = cell.path_response(case.distance, case.velocity)[BASELINE_COUNT:]
    # The wider model tolerates range-grid and velocity errors and displaced
    # multipath. These are fixed hypotheses, independent of the target draw.
    neighbors = [
        0.30 * cell.path_response(case.distance + distance, case.velocity + velocity)[BASELINE_COUNT:]
        for distance, velocity in ((-2.5, 0), (2.5, 0), (0, -0.45), (0, 0.45))
    ]
    return direct, np.column_stack((direct, *neighbors))


class RefinedDetectors:
    def __init__(self, cell, case, training, noise_factor):
        noise = noise_factor @ noise_factor.conj().T
        self.reference = Detectors(training[:, :BASELINE_COUNT],
                                   cell.nominal_dual_gain(case.distance, case.velocity),
                                   case.distance, noise.diagonal().real[:BASELINE_COUNT])
        self.distance = case.distance
        if case.distance >= 1200:
            return
        direct, widened = target_models(cell, case)
        extra_training = training[:, BASELINE_COUNT:]
        extra_noise = noise[BASELINE_COUNT:, BASELINE_COUNT:]
        self.rectangular = AdaptiveProjection(extra_training[:, :2], extra_noise[:2, :2],
                                              direct[:2])
        self.direct = AdaptiveProjection(extra_training, extra_noise, direct)
        self.widened = AdaptiveProjection(extra_training, extra_noise, widened)

    def scores(self, samples):
        scores = self.reference.scores(samples[:, :BASELINE_COUNT], self.distance)
        result = {name: scores[name] for name in BASELINES}
        if self.distance >= 1200:
            for name in ("rectangular_cut", "range_doppler_glrt", DIRECT, "tolerant_amf"):
                result[name] = result["late_adaptive"]
            return result
        auxiliary = samples[:, BASELINE_COUNT:]
        _, rectangular = self.rectangular.scores(auxiliary[:, :2])
        glrt, direct = self.direct.scores(auxiliary)
        _, tolerant = self.widened.scores(auxiliary)
        return {**result, "rectangular_cut": rectangular, "range_doppler_glrt": glrt,
                DIRECT: direct, "tolerant_amf": tolerant}


def evaluate(stress, seed, null_count, target_count):
    rng = np.random.default_rng(seed)
    case = stress.case
    views = auxiliary_views(case.distance, case.velocity, stress.pulses) if case.distance < 1200 else ()
    cell = RangeVelocityCell(case.distance, case.velocity, extra_views=views,
                             pulse_count=stress.pulses)
    noise = cell.noise_factor()
    test_kernels = clutter_kernels(cell, case, *stress.background_shift)
    development_kernels = test_kernels if stress.refresh_background else clutter_kernels(cell, case)
    training = background(rng, BACKGROUND_TRAIN, noise, development_kernels, case.clutter)
    detector = RefinedDetectors(cell, case, training, noise)
    calibrated = detector.scores(
        background(rng, null_count, noise, development_kernels, case.clutter),
    )
    # The refresh scenario has independent target-free references from the
    # new environment. In the stale-map scenario the change only affects test.
    test_background = background(rng, null_count, noise, test_kernels, case.clutter)
    test_null = detector.scores(test_background)
    test_target = detector.scores(
        test_background[:target_count] + target_ensemble(rng, cell, stress, target_count),
    )
    del test_background
    rows = []
    thresholds = {
        method: np.quantile(score, 1 - PFA, method="higher")
        for method, score in calibrated.items()
    }
    reference_hits = test_target["late_adaptive"] > thresholds["late_adaptive"]
    for method, threshold in thresholds.items():
        count = np.count_nonzero(test_null[method] > threshold)
        hits = test_target[method] > threshold
        detections = np.count_nonzero(hits)
        paired_difference = hits.astype(int) - reference_hits
        delta = np.mean(paired_difference)
        half_width = 1.96 * np.std(paired_difference, ddof=1) / np.sqrt(target_count)
        rank = null_count - int(np.floor(null_count * PFA)) - 1
        roc_threshold = np.partition(test_null[method], rank)[rank]
        pd_low, pd_high = wilson(detections, target_count)
        pfa_low, pfa_high = wilson(count, null_count)
        rows.append({
            "scenario": stress.name, "range_m": case.distance, "velocity_mps": case.velocity,
            "pulses": stress.pulses, "method": method, "pd": detections / target_count,
            "pd_low": pd_low, "pd_high": pd_high, "pfa": count / null_count,
            "pfa_low": pfa_low, "pfa_high": pfa_high,
            "roc_pd": np.mean(test_target[method] > roc_threshold),
            "delta_pd_vs_late": delta, "delta_pd_half_width": half_width,
            "detections": detections, "false_alarms": count, "target_trials": target_count,
            "null_trials": null_count, "threshold": threshold,
        })
    return rows


def save_stress_plot(output, rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    panels = (
        ("edge", "100 m, strong ground scatterer", "velocity_mps", "Radial velocity (m/s)"),
        ("transition", "Near-to-far transition", "range_m", "Range (m)"),
        ("longer_dwell", "250 m, dwell sensitivity", "pulses", "Pulses in dwell"),
    )
    methods = {
        "fixed_long_ppc": ("Fixed long, partial match", "#496580", "-"),
        "whitened": ("Adjacent, two-channel whitening", "#d27736", "--"),
        "late_adaptive": ("Late, previous adaptive fusion", "#934d83", "--"),
        "rectangular_cut": ("Late, rectangular two-channel", "#71777c", ":"),
        DIRECT: ("Late, range–Doppler AMF", "#087b6c", "-"),
    }
    for axis, (scenario, title, key, label) in zip(axes, panels):
        for method, (caption, color, style) in methods.items():
            selected = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            if scenario == "longer_dwell":
                selected[:0] = [r for r in rows if r["scenario"] == "transition"
                                and r["range_m"] == 250 and r["method"] == method]
            axis.plot([r[key] for r in selected], [r["pd"] for r in selected], marker="o",
                      markersize=3, label=caption, color=color, linestyle=style)
        axis.set(title=title, xlabel=label, ylim=(-0.02, 1.03))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Detection probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
    fig.savefig(output / "bottleneck_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "benchmark_output" / "refinement")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--null-trials", type=int, default=200_000)
    parser.add_argument("--target-trials", type=int, default=6000)
    args = parser.parse_args()
    if args.null_trials < 20_000 or not 2 <= args.target_trials <= args.null_trials:
        parser.error("Require at least 20000 null trials and 2 to null-trials target trials.")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(stress_cases())
    seeds = np.random.SeedSequence(args.seed).spawn(len(cases))
    rows = []
    for stress, seed in zip(cases, seeds):
        rows.extend(evaluate(stress, seed, args.null_trials, args.target_trials))
        print(f"{stress.name:>20}  {stress.case.distance:>4g} m  "
              f"{stress.case.velocity:>4g} m/s  {stress.pulses} pulses", flush=True)
    write_csv(args.output / "bottleneck_operating_points.csv", rows)
    save_stress_plot(args.output, rows)


if __name__ == "__main__":
    main()
