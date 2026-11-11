"""Compare a steeper LFM tail with linear LFM at equal transmit energy."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from radar_benchmark import (
    BACKGROUND_TRAIN, PFA, Case, background, clutter_kernels, target_trials, wilson, write_csv,
)
from radar_demo import Mode, plt, sample_echo
from radar_refinement import AdaptiveProjection, auxiliary_views
from radar_scene import CHANNELS, DWELL_PULSES, LATE_DUAL, LONG, TIME, RangeVelocityCell, View
from radar_scene import projection


TAIL_LONG = replace(LONG, tail_duration=1.6e-6, tail_bandwidth=12e6)
TAIL_MODE = Mode("Late dual with steeper tail", (TAIL_LONG, LATE_DUAL.pulses[1]))
LINEAR_CHANNELS = CHANNELS[-2:]
TAIL_CHANNELS = (
    replace(LINEAR_CHANNELS[0], name="tail_long", mode=TAIL_MODE, pulse=TAIL_LONG),
    replace(LINEAR_CHANNELS[1], name="tail_short", mode=TAIL_MODE),
)
WAVEFORMS = ("linear", "steeper_tail")


def study_cases():
    yield Case("strong_scatterer", 100, 0, 1, 0.38, edge=True)
    yield Case("strong_scatterer", 100, 1.95, 1, 0.38, edge=True)
    yield Case("transition", 142, 3.9, 1, 0.38)
    for clutter, reflection, scenario in ((0, 0, "noise"), (1, 0, "clutter"), (1, 0.38, "low")):
        for input_db in (0, 3, 6, 9):
            yield Case(f"250m_{scenario}", 250, 3.9, clutter, reflection,
                       amplitude_scale=10 ** (input_db / 20))
    for distance in (4000, 6000):
        yield Case("far", distance, 7.8, 1, 0.38, far_budget=True)
    yield Case("250m_low", 250, 3.9, 1, 0.38, amplitude_scale=10 ** (12 / 20))


def evaluate(case, seed, null_count, target_count):
    rng = np.random.default_rng(seed)
    modes = (LINEAR_CHANNELS, TAIL_CHANNELS)
    near = case.distance < 1200
    if near:
        views = tuple(
            auxiliary_views(case.distance, case.velocity, DWELL_PULSES, channels)
            for channels in modes
        )
    else:
        views = tuple((View(channels[0], case.distance, case.velocity),) for channels in modes)
    cell = RangeVelocityCell(case.distance, case.velocity, channels=(), extra_views=views[0] + views[1])
    noise_factor = cell.noise_factor()
    noise = noise_factor @ noise_factor.conj().T
    kernels = clutter_kernels(cell, case) if case.clutter else None

    def null_samples(count):
        return background(rng, count, noise_factor, kernels, case.clutter)

    groups = (slice(0, len(views[0])), slice(len(views[0]), len(cell.channels)))
    training = null_samples(BACKGROUND_TRAIN)
    if near:
        model = cell.path_response(case.distance, case.velocity)
        detectors = tuple(
            AdaptiveProjection(training[:, group], noise[group, group], model[group])
            for group in groups
        )

        def scores(samples):
            return np.column_stack([detector.scores(samples[:, group])[1]
                                    for detector, group in zip(detectors, groups)])
    else:
        def scores(samples):
            return abs(samples) ** 2

    calibration = scores(null_samples(null_count))
    threshold = np.quantile(calibration, 1 - PFA, axis=0, method="higher")
    evaluation = null_samples(null_count)
    evaluation_scores = scores(evaluation)
    target_scores = scores(evaluation[:target_count] + target_trials(rng, cell, case, target_count))
    rank = null_count - int(np.floor(null_count * PFA)) - 1
    roc_threshold = np.partition(evaluation_scores, rank, axis=0)[rank]
    rows = []
    for index, waveform in enumerate(WAVEFORMS):
        hits = np.count_nonzero(target_scores[:, index] > threshold[index])
        false_alarms = np.count_nonzero(evaluation_scores[:, index] > threshold[index])
        pd_low, pd_high = wilson(hits, target_count)
        rows.append({
            "scenario": case.name, "range_m": case.distance, "velocity_mps": case.velocity,
            "input_power_db": 20 * np.log10(case.amplitude_scale), "waveform": waveform,
            "pd": hits / target_count, "pd_low": pd_low, "pd_high": pd_high,
            "pfa": false_alarms / null_count,
            "roc_pd": np.mean(target_scores[:, index] > roc_threshold[index]),
            "false_alarms": false_alarms, "target_trials": target_count, "null_trials": null_count,
        })
    return rows


def pulse_response(channel, distance):
    echo = sample_echo(TIME, channel.pulse, distance)
    offsets = np.arange(-200, 201) * 0.5
    power = np.array([
        abs(np.vdot(projection(channel, distance + offset), echo)) ** 2
        for offset in offsets
    ])
    return offsets, power


def response_metrics(offsets, power):
    peak = len(offsets) // 2
    half = power[peak] / 2
    left = np.flatnonzero(power[:peak] <= half)[-1]
    right = peak + np.flatnonzero(power[peak:] <= half)[0]
    left_half = np.interp(half, power[left:left + 2], offsets[left:left + 2])
    right_half = np.interp(half, power[right - 1:right + 1][::-1], offsets[right - 1:right + 1][::-1])
    minima, _ = find_peaks(-power)
    left_minima = minima[minima < left_half / 0.5 + peak]
    right_minima = minima[minima > right_half / 0.5 + peak]
    sidelobe_db = ""
    if len(left_minima) and len(right_minima):
        sidelobes = np.r_[power[:left_minima[-1]], power[right_minima[0] + 1:]]
        if len(sidelobes):
            sidelobe_db = 10 * np.log10(max(sidelobes) / power[peak])
    return right_half - left_half, sidelobe_db, power[peak]


def save_comparison(path, curves, rows):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    labels = ("Linear chirp", "Steeper tail")
    colors = ("#47688d", "#b65336")
    for axis, distance in zip(axes[0], (100, 250, 4000)):
        for waveform, label, color in zip(WAVEFORMS, labels, colors):
            offsets, power = curves[waveform, distance]
            normalized = power / power[len(power) // 2]
            axis.plot(offsets, 10 * np.log10(np.maximum(normalized, 1e-8)),
                      label=label, color=color)
        axis.set(title=f"Long-pulse response at {distance} m", xlim=(-75, 75), ylim=(-45, 3),
                 xlabel="Range offset (m)")
    for axis, suffix, title in zip(axes[1], ("noise", "clutter", "low"),
                                   ("Thermal noise", "Ground clutter", "Clutter and multipath")):
        for waveform, label, color in zip(WAVEFORMS, labels, colors):
            selected = [r for r in rows if r["scenario"] == f"250m_{suffix}"
                        and r["waveform"] == waveform]
            axis.plot([r["input_power_db"] for r in selected], [r["roc_pd"] for r in selected],
                      marker="o", label=label, color=color)
        axis.axhline(0.90, color="0.65", linestyle=":")
        axis.set(title=f"250 m · {title}", xlabel="Input power relative to reference (dB)",
                 ylim=(-0.03, 1.03))
    axes[0, 0].set_ylabel("Normalized pulse compression (dB)")
    axes[1, 0].set_ylabel("Detection probability at matched Pfa")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    handles, captions = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, captions, loc="outside lower center", ncols=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "benchmark_output" / "refinement")
    parser.add_argument("--seed", type=int, default=64957)
    parser.add_argument("--null-trials", type=int, default=200_000)
    parser.add_argument("--target-trials", type=int, default=6000)
    args = parser.parse_args()
    if args.null_trials < 20_000 or not 2 <= args.target_trials <= args.null_trials:
        parser.error("Require at least 20000 null trials and 2 to null-trials target trials.")
    cases = list(study_cases())
    seeds = np.random.SeedSequence(args.seed).spawn(len(cases))
    rows = []
    for case, seed in zip(cases, seeds):
        rows.extend(evaluate(case, seed, args.null_trials, args.target_trials))
        print(f"{case.name:>17} {case.distance:>4g} m, {20*np.log10(case.amplitude_scale):g} dB",
              flush=True)
    response = []
    curves = {}
    for name, channels in (("linear", LINEAR_CHANNELS), ("steeper_tail", TAIL_CHANNELS)):
        for distance in (100, 250, 4000):
            offsets, power = pulse_response(channels[0], distance)
            curves[name, distance] = (offsets, power)
            width, sidelobe, peak = response_metrics(offsets, power)
            response.append({
                "waveform": name, "range_m": distance, "half_power_width_m": width,
                "peak_sidelobe_db": sidelobe, "target_cell_power": peak,
            })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tail_waveform_detection.csv", rows)
    write_csv(args.output / "tail_waveform_range_response.csv", response)
    save_comparison(args.output / "tail_waveform_comparison.png", curves, rows)


if __name__ == "__main__":
    main()
