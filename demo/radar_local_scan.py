"""Validate detection and localization when the target cell is unknown."""

import argparse
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

from radar_benchmark import (
    BACKGROUND_TRAIN, PFA, Case, Fusion, background, clutter_kernels,
    target_trials, wilson, write_csv,
)
from radar_demo import PRI, SPEED_OF_LIGHT
from radar_refinement import AdaptiveProjection, auxiliary_views
from radar_scene import CARRIER, CHANNELS, DUAL_SLICE, DWELL_PULSES, LATE_SLICE, RangeVelocityCell, View


METHODS = ("adjacent_whitened", "late_adaptive", "range_doppler_glrt", "range_doppler_amf")
SCORE_TRAIN = 8192
BATCH_SIZE = 8192
RANGE_STEP = 7.5
VELOCITY_STEP = SPEED_OF_LIGHT / (2 * CARRIER * PRI * DWELL_PULSES)


@dataclass(frozen=True)
class Candidate:
    distance: float
    velocity: float
    adjacent: np.ndarray
    late: np.ndarray
    auxiliary: np.ndarray


def build_patch(case, pulse_count):
    views = []
    index = {}
    candidates = []

    def indices(local_views):
        result = []
        for view in local_views:
            key = (view.channel.name, view.taper, round(view.distance, 8), round(view.velocity, 8))
            if key not in index:
                index[key] = len(views)
                views.append(view)
            result.append(index[key])
        return np.array(result)

    velocity_bins = pulse_count // DWELL_PULSES
    speed_offsets = np.arange(-velocity_bins, velocity_bins + 1) / velocity_bins
    for offset_range, offset_speed in product((-1, 0, 1), speed_offsets):
        distance = case.distance + RANGE_STEP * offset_range
        velocity = case.velocity + VELOCITY_STEP * offset_speed
        adjacent = indices(View(channel, distance, velocity) for channel in CHANNELS[DUAL_SLICE])
        late = indices(View(channel, distance, velocity) for channel in CHANNELS[LATE_SLICE])
        auxiliary = indices(auxiliary_views(distance, velocity, pulse_count))
        candidates.append(Candidate(distance, velocity, adjacent, late, auxiliary))
    bank = RangeVelocityCell(case.distance, case.velocity, channels=(), extra_views=views,
                             pulse_count=pulse_count)
    return bank, candidates


class PatchCell:
    def __init__(self, bank, candidate, training, noise):
        self.cell = candidate
        response = bank.path_response(candidate.distance, candidate.velocity)
        adjacent_gain = abs(np.diag(response[candidate.adjacent])) ** 2
        late_gain = abs(np.diag(response[candidate.late])) ** 2
        self.adjacent = Fusion(training[:, candidate.adjacent], adjacent_gain)
        self.late = Fusion(training[:, candidate.late], late_gain)
        late_thermal = noise.diagonal().real[candidate.late]
        self.late_rule = "whitened" if max(self.late.background_power / late_thermal) > 2 else "equal"
        aux = candidate.auxiliary
        self.auxiliary = AdaptiveProjection(training[:, aux], noise[np.ix_(aux, aux)], response[aux])

    def scores(self, samples):
        adjacent = self.adjacent.scores(samples[:, self.cell.adjacent], self.cell.distance)
        late = self.late.scores(samples[:, self.cell.late], self.cell.distance)
        glrt, adaptive = self.auxiliary.scores(samples[:, self.cell.auxiliary])
        return np.column_stack((adjacent["whitened"], late[self.late_rule], glrt, adaptive))


def draw_targets(rng, bank, case, count):
    locations = tuple(product((-6.0, -0.7, 6.0), (-1.55, 0.15, 1.55)))
    selection = rng.integers(len(locations), size=count)
    signal = np.empty((count, len(bank.channels)), dtype=complex)
    truth = np.empty((count, 2))
    for index, (distance, velocity) in enumerate(locations):
        selected = selection == index
        target_case = replace(case, distance=case.distance + distance,
                              velocity=case.velocity + velocity)
        signal[selected] = target_trials(rng, bank, target_case, np.count_nonzero(selected))
        truth[selected] = (target_case.distance, target_case.velocity)
    return signal, truth


def score_cube(detectors, samples):
    return np.stack([detector.scores(samples) for detector in detectors], axis=1)


def evaluate_patch(case, pulse_count, seed, null_count, target_count):
    rng = np.random.default_rng(seed)
    bank, candidates = build_patch(case, pulse_count)
    noise_factor = bank.noise_factor()
    noise = noise_factor @ noise_factor.conj().T
    active_modes = np.linalg.norm(noise_factor, axis=0) > 1e-8
    noise_factor = noise_factor[:, active_modes]
    kernels = clutter_kernels(bank, case)

    def null_samples(count):
        return background(rng, count, noise_factor, kernels, case.clutter)

    training = null_samples(BACKGROUND_TRAIN)
    detectors = [PatchCell(bank, candidate, training, noise) for candidate in candidates]
    scale_reference = score_cube(detectors, null_samples(SCORE_TRAIN))
    median, upper = np.quantile(scale_reference, [0.5, 0.99], axis=0)
    scale = np.maximum(upper - median, 1e-12)

    def scores(samples):
        return (score_cube(detectors, samples) - median) / scale

    def background_maxima():
        parts = []
        for start in range(0, null_count, BATCH_SIZE):
            count = min(BATCH_SIZE, null_count - start)
            parts.append(scores(null_samples(count)).max(axis=1))
        return np.concatenate(parts)

    threshold = np.quantile(background_maxima(), 1 - PFA, axis=0, method="higher")
    test_null = background_maxima()
    signal, truth = draw_targets(rng, bank, case, target_count)
    target_scores = scores(null_samples(target_count) + signal)
    candidate_index = target_scores.argmax(axis=1)
    candidate_values = target_scores.max(axis=1)
    locations = np.array([(candidate.distance, candidate.velocity) for candidate in candidates])
    errors = abs(locations[candidate_index] - truth[:, None, :])
    associated = np.all(errors <= (RANGE_STEP, VELOCITY_STEP), axis=2)
    rank = null_count - int(np.floor(null_count * PFA)) - 1
    roc_threshold = np.partition(test_null, rank, axis=0)[rank]
    rows = []
    for index, method in enumerate(METHODS):
        alarm = candidate_values[:, index] > threshold[index]
        hit = alarm & associated[:, index]
        false_alarms = np.count_nonzero(test_null[:, index] > threshold[index])
        pd_low, pd_high = wilson(np.count_nonzero(hit), target_count)
        pfa_low, pfa_high = wilson(false_alarms, null_count)
        hit_errors = errors[hit, index]
        rows.append({
            "scenario": case.name, "patch_range_m": case.distance,
            "patch_velocity_mps": case.velocity, "method": method,
            "pulses": pulse_count, "candidate_cells": len(candidates),
            "patch_pd": hit.mean(), "pd_low": pd_low, "pd_high": pd_high,
            "patch_pfa": false_alarms / null_count, "pfa_low": pfa_low, "pfa_high": pfa_high,
            "roc_pd": np.mean(
                (candidate_values[:, index] > roc_threshold[index]) & associated[:, index],
            ),
            "mislocalized_fraction": np.mean(alarm & ~associated[:, index]),
            "median_range_error_m": np.median(hit_errors[:, 0]) if len(hit_errors) else "",
            "median_velocity_error_mps": np.median(hit_errors[:, 1]) if len(hit_errors) else "",
            "false_alarms": false_alarms, "target_trials": target_count, "null_trials": null_count,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "benchmark_output" / "refinement")
    parser.add_argument("--seed", type=int, default=46277)
    parser.add_argument("--null-trials", type=int, default=200_000)
    parser.add_argument("--target-trials", type=int, default=6000)
    args = parser.parse_args()
    if args.null_trials < 20_000 or args.target_trials < 2:
        parser.error("Require at least 20000 null trials and 2 target trials.")
    cases = (
        (Case("strong_scatterer", 100, 0, 1, 0.38, edge=True), DWELL_PULSES),
        (Case("transition", 250, 3.9, 1, 0.38), DWELL_PULSES),
        (Case("transition", 250, 3.9, 1, 0.38), 256),
    )
    seeds = np.random.SeedSequence(args.seed).spawn(len(cases))
    rows = []
    for (case, pulse_count), seed in zip(cases, seeds):
        rows.extend(evaluate_patch(case, pulse_count, seed, args.null_trials, args.target_trials))
        print(f"Scanned local patch at {case.distance} m, {pulse_count} pulses ({case.name})",
              flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "local_scan_operating_points.csv", rows)


if __name__ == "__main__":
    main()
