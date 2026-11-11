"""Reproducible low-altitude radar comparison at a common calibrated Pfa."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radar_demo import plt
from radar_scene import DUAL, DUAL_SLICE, LATE_SLICE, LONG, RangeVelocityCell, reflected_paths


PFA = 1e-4
NEAR_RANGES = (100, 110, 120, 130, 142, 155, 170, 185, 205, 225, 250)
FAR_RANGES = (1200, 1420, 1582, 1700, 1900, 2000, 4000, 6000, 8000)
BACKGROUND_TRAIN = 4096
DEFAULT_SEED = 20261111
COMPARE_WITH = ("fixed_long_ppc", "fixed_short", "alternate", "hard_switch", "equal")
FUSION_METHODS = ("hard_switch", "equal", "energy", "background", "joint", "whitened", "maximum")


@dataclass(frozen=True)
class Case:
    name: str
    distance: float
    velocity: float
    clutter: float
    reflection: float
    edge: bool = False
    far_budget: bool = False
    amplitude_scale: float = 1.0
    equal_average_power: bool = False


def experiment_cases():
    for name, clutter, reflection in (
        ("high", 0.0, 0.0), ("clutter", 1.0, 0.0), ("low", 1.0, 0.38),
    ):
        yield Case(name, 100, 7.8, clutter, reflection)
    for distance in NEAR_RANGES:
        yield Case("transition", distance, 3.9, 1.0, 0.38)
    for velocity in (0.0, 1.95, 3.9, 7.8):
        yield Case("edge", 100, velocity, 1.0, 0.38, edge=True)
    for distance in FAR_RANGES:
        yield Case("far", distance, 7.8, 1.0, 0.38, far_budget=True)
    for relative_level in (0.5, 0.8, 1.25, 1.6):
        yield Case(f"far_amplitude_x{relative_level:g}", 4000, 7.8, 1.0, 0.38,
                   far_budget=True, amplitude_scale=relative_level)
    for reflection in (0.17, 0.65):
        yield Case(f"reflection_{reflection:g}", 100, 7.8, 1.0, reflection)
    for distance in (2000, 4000, 6000, 8000):
        yield Case("equal_average", distance, 7.8, 1.0, 0.38, far_budget=True,
                   equal_average_power=True)


def circular_normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def clutter_kernels(cell, case, edge_offset=19, edge_velocity=1.15):
    """A shared ground patch with near-zero Doppler and an optional strong edge."""
    offsets = (-105, -65, -32, -15, -8, 5, 13, 27, 48, 75, 115)
    velocities = (-0.7, 0.0, 0.5, 0.9, -0.3, 0.2, -0.6, 1.0, -1.0, 0.3, 0.0)
    kernels = []
    for offset, velocity in zip(offsets, velocities):
        distance = max(30, case.distance + offset)
        weight = (100 / distance) ** 2 * np.exp(-abs(offset) / 115)
        response = cell.path_response(distance, velocity)
        # One shared scatterer plus a frequency-dependent residual preserves
        # correlation between the channels without requiring equal speckle.
        kernels.append(weight * response.sum(axis=1))
        kernels.extend(0.28 * weight * response.T)
    if case.edge:
        distance = case.distance + edge_offset
        kernels.append(2.6 * cell.path_response(distance, edge_velocity).sum(axis=1))
    return np.array(kernels)


def background(rng, count, noise_factor, kernels, clutter):
    noise = circular_normal(rng, (count, noise_factor.shape[1])) @ noise_factor.T
    if clutter == 0:
        return noise
    # One lognormal power texture per independent dwell. The noise floor
    # remains constant as the intensity of the ground returns fluctuates.
    texture = np.exp(rng.normal(-0.5 * 0.55**2, 0.55, size=(count, 1)))
    ground = circular_normal(rng, (count, len(kernels))) @ kernels
    return noise + 0.065 * np.sqrt(clutter * texture) * ground


def target_amplitude(case):
    if case.far_budget:
        return case.amplitude_scale * 0.017 * (4000 / case.distance) ** 2
    return case.amplitude_scale * 0.105 * (100 / case.distance) ** 2


def target_trials(rng, cell, case, count):
    # Stratify the reflection response by target height across the independent
    # trials, with independent speckle amplitude and phase in each subband.
    heights = np.array([5, 7.5, 10, 15, 20, 30, 45])
    responses = []
    reflection = case.reflection * np.exp(-2.2j)
    for height in heights:
        response = sum(
            coefficient * cell.path_response(distance, case.velocity)
            for distance, coefficient in reflected_paths(case.distance, height, reflection)
            if coefficient != 0
        )
        responses.append(response)
    choice = rng.integers(len(heights), size=count)
    amplitude = np.exp(rng.normal(-0.23**2, 0.23, size=(count, 2)))
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(count, 2)))
    return target_amplitude(case) * np.einsum(
        "tcb,tb->tc", np.asarray(responses)[choice], amplitude * phase,
    )


def simplex_weights(expected, covariance):
    """Maximize expected contrast divided by background standard deviation."""
    long_weight = np.linspace(0, 1, 201)
    candidates = np.column_stack((long_weight, 1 - long_weight))
    variance = np.einsum("ni,ij,nj->n", candidates, covariance, candidates)
    return candidates[np.argmax((candidates @ expected) ** 2 / variance)]


class Fusion:
    def __init__(self, training, gain):
        self.background_power = np.maximum(np.mean(abs(training) ** 2, axis=0), 1e-12)
        normalized = abs(training) ** 2 / self.background_power - 1
        covariance = np.cov(normalized, rowvar=False)
        covariance = 0.85 * covariance + 0.15 * np.diag(np.diag(covariance))
        covariance += 0.005 * np.eye(2)
        self.equal = (gain > 0).astype(float)
        self.equal /= self.equal.sum()
        self.energy = gain / gain.sum()
        self.background = self.equal / self.background_power
        self.background /= self.background.sum()
        self.joint = simplex_weights(gain / self.background_power, covariance)

        # A covariance-aware energy statistic. Whiten complex amplitudes,
        # then give more weight to directions with useful nominal signal energy.
        covariance_complex = training.T @ training.conj() / len(training)
        covariance_complex = (
            0.95 * covariance_complex + 0.05 * np.diag(self.background_power)
            + 0.005 * np.eye(2)
        )
        inverse_background = np.linalg.inv(covariance_complex)
        self.whitened = inverse_background @ np.diag(gain) @ inverse_background

    def scores(self, samples, distance):
        power = abs(samples) ** 2 / self.background_power - 1
        # Far mode locks to the long branch for every same-waveform comparison.
        if distance >= 1200:
            return dict.fromkeys(FUSION_METHODS, power[:, 0])
        return {
            "hard_switch": power[:, int(distance < 142.4)],
            "equal": power @ self.equal,
            "energy": power @ self.energy,
            "background": power @ self.background,
            "joint": power @ self.joint,
            "whitened": np.einsum("ni,ij,nj->n", samples.conj(), self.whitened, samples).real,
            "maximum": power[:, self.equal > 0].max(axis=1),
        }


class Detectors:
    def __init__(self, training, gains, distance, thermal_power):
        self.background_power = np.maximum(np.mean(abs(training) ** 2, axis=0), 1e-12)
        self.hard_channel = int(distance < 142.4)
        self.adjacent = Fusion(training[:, DUAL_SLICE], gains[0])
        self.late = Fusion(training[:, LATE_SLICE], gains[1])
        background_to_thermal = self.late.background_power / np.maximum(
            thermal_power[LATE_SLICE], 1e-12,
        )
        self.late_rule = "whitened" if background_to_thermal.max() > 2 else "equal"

    def scores(self, samples, distance):
        power = abs(samples) ** 2 / self.background_power - 1
        late_scores = self.late.scores(samples[:, LATE_SLICE], distance)
        late_scores["adaptive"] = late_scores[self.late_rule]
        return {
            "fixed_long_ppc": power[:, 0],
            "fixed_long_full": power[:, 1],
            "fixed_short": power[:, 2],
            "alternate": power[:, 3 + self.hard_channel],
            "dual_long": power[:, DUAL_SLICE.start],
            "dual_short": power[:, DUAL_SLICE.start + 1],
            "late_long": power[:, LATE_SLICE.start],
            "late_short": power[:, LATE_SLICE.start + 1],
            **self.adjacent.scores(samples[:, DUAL_SLICE], distance),
            **{f"late_{name}": score for name, score in late_scores.items()},
        }


def wilson(count, total):
    probability = count / total
    z = 1.96
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2))
    return max(0, center - half / denominator), min(1, center + half / denominator)


def evaluate(case, seed, null_count, target_count):
    rng = np.random.default_rng(seed)
    cell = RangeVelocityCell(case.distance, case.velocity)
    noise = cell.noise_factor()
    kernels = clutter_kernels(cell, case) if case.clutter else None
    transmit_voltage = np.ones(len(noise))
    if case.equal_average_power:
        dual_duration = sum(pulse.duration for pulse in DUAL.pulses)
        transmit_voltage[DUAL_SLICE.start:] = np.sqrt(LONG.duration / dual_duration)
        if kernels is not None:
            kernels *= transmit_voltage
    training = background(rng, BACKGROUND_TRAIN, noise, kernels, case.clutter)
    calibration = background(rng, null_count, noise, kernels, case.clutter)
    evaluation = background(rng, null_count, noise, kernels, case.clutter)
    targets = evaluation[:target_count] + target_trials(rng, cell, case, target_count) * transmit_voltage
    gains = cell.nominal_dual_gain(case.distance, case.velocity)
    detectors = Detectors(training, gains, case.distance, np.sum(abs(noise) ** 2, axis=1))
    calibration_scores = detectors.scores(calibration, case.distance)
    evaluation_scores = detectors.scores(evaluation, case.distance)
    target_scores = detectors.scores(targets, case.distance)
    rows = []
    detected = {}
    for name, calibration_score in calibration_scores.items():
        threshold = np.quantile(calibration_score, 1 - PFA, method="higher")
        false_alarms = int(np.count_nonzero(evaluation_scores[name] > threshold))
        detected[name] = target_scores[name] > threshold
        hits = int(detected[name].sum())
        # Diagnostic ROC interpolation at the same observed null count.
        # The deployed threshold above still comes from independent calibration.
        null_rank = null_count - int(np.floor(null_count * PFA)) - 1
        roc_threshold = np.partition(evaluation_scores[name], null_rank)[null_rank]
        pd_at_matched_pfa = np.mean(target_scores[name] > roc_threshold)
        pd_low, pd_high = wilson(hits, target_count)
        pfa_low, pfa_high = wilson(false_alarms, null_count)
        rows.append({
            "scenario": case.name, "range_m": case.distance, "velocity_mps": case.velocity,
            "method": name, "pd": hits / target_count, "pd_low": pd_low, "pd_high": pd_high,
            "roc_pd_at_matched_pfa": pd_at_matched_pfa,
            "pfa": false_alarms / null_count, "pfa_low": pfa_low, "pfa_high": pfa_high,
            "hits": hits, "target_trials": target_count, "false_alarms": false_alarms,
            "null_trials": null_count, "threshold": threshold,
            "adjacent_joint_long_weight": detectors.adjacent.joint[0],
            "late_joint_long_weight": detectors.late.joint[0],
            "late_rule": detectors.late_rule,
        })
    return rows, compare_paired(case, detected)


def compare_paired(case, detected):
    rows = []
    for candidate in ("joint", "whitened", "late_equal", "late_joint", "late_whitened",
                      "late_adaptive"):
        for baseline in COMPARE_WITH:
            difference = detected[candidate].astype(int) - detected[baseline]
            delta = difference.mean()
            # With few discordant outcomes the normal interval can collapse.
            # The sum of the two Wilson bounds gives a conservative fallback.
            candidate_only = int(np.count_nonzero(difference == 1))
            baseline_only = int(np.count_nonzero(difference == -1))
            if candidate_only + baseline_only < 10:
                candidate_low, candidate_high = wilson(candidate_only, difference.size)
                baseline_low, baseline_high = wilson(baseline_only, difference.size)
                low, high = candidate_low - baseline_high, candidate_high - baseline_low
            else:
                half_width = 1.96 * difference.std(ddof=1) / np.sqrt(difference.size)
                low, high = max(-1, delta - half_width), min(1, delta + half_width)
            rows.append({
                "scenario": case.name, "range_m": case.distance, "velocity_mps": case.velocity,
                "candidate": candidate, "baseline": baseline, "delta_pd": delta,
                "delta_low": low, "delta_high": high,
                "candidate_only": candidate_only, "baseline_only": baseline_only,
            })
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(output_dir, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, scenario in zip(axes, ("transition", "far")):
        methods = (*COMPARE_WITH, "joint", "late_adaptive")
        for method in methods:
            data = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            axis.plot([row["range_m"] for row in data], [row["pd"] for row in data],
                      marker=".", label=method.replace("_", " "))
        axis.set(title=scenario.capitalize(), xlabel="Range (m)", ylabel="Detection probability",
                 ylim=(-0.025, 1.025))
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "detection_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "benchmark_output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--null-trials", type=int, default=200_000)
    parser.add_argument("--target-trials", type=int, default=6_000)
    args = parser.parse_args()
    if args.null_trials < 20_000 or not 2 <= args.target_trials <= args.null_trials:
        parser.error("Require at least 20000 null trials and 2 to null-trials target trials.")

    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(experiment_cases())
    seeds = np.random.SeedSequence(args.seed).spawn(len(cases))
    rows = []
    paired = []
    for case, seed in zip(cases, seeds):
        case_rows, case_paired = evaluate(case, seed, args.null_trials, args.target_trials)
        rows.extend(case_rows)
        paired.extend(case_paired)
        print(f"{case.name:>20}  {case.distance:>5g} m  {case.velocity:>4g} m/s", flush=True)
    write_csv(args.output / "operating_points.csv", rows)
    write_csv(args.output / "paired_comparisons.csv", paired)
    save_comparison(args.output, rows)


if __name__ == "__main__":
    main()
