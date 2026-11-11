"""Process a mixed-target ADC dwell into a range–velocity map and CFAR points."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import brentq

from radar_benchmark import PFA, circular_normal, write_csv
from radar_demo import PRI, SPEED_OF_LIGHT, plt
from radar_scene import (
    CARRIER, CHANNELS, DWELL_PULSES, LATE_DUAL, TIME,
    add_raw_path, projection, reflected_paths,
)


@dataclass(frozen=True)
class Target:
    distance: float
    velocity: float
    height: float
    voltage: float


TARGETS = (
    Target(100, 7.8, 8, 0.13),
    Target(180, -5.85, 13, 0.10),
    Target(1420, 15.6, 19, 0.045),
    Target(4000, -11.7, 18, 0.035),
    Target(7600, 19.5, 30, 0.031),
)


def mixed_dwell(seed):
    rng = np.random.default_rng(seed)
    raw = circular_normal(rng, (DWELL_PULSES, len(TIME)))
    for distance in np.r_[rng.uniform(65, 330, 25), rng.uniform(400, 8200, 45)]:
        speed = rng.normal(0, 0.45)
        subbands = circular_normal(rng, ()) + 0.28 * circular_normal(rng, 2)
        voltage = 0.11 * (100 / distance) ** 2
        add_raw_path(raw, LATE_DUAL, distance, speed, voltage, subbands)
    add_raw_path(raw, LATE_DUAL, 119, 1.15, 0.22, np.array([1.0, np.exp(0.9j)]))
    reflection = 0.38 * np.exp(-2.2j)
    for target in TARGETS:
        subbands = np.exp(1j * rng.uniform(-np.pi, np.pi, size=2))
        for distance, coefficient in reflected_paths(target.distance, target.height, reflection):
            add_raw_path(raw, LATE_DUAL, distance, target.velocity,
                         target.voltage * coefficient, subbands)
    return raw


def range_velocity_power(raw, ranges):
    velocities = np.fft.fftshift(np.fft.fftfreq(DWELL_PULSES, PRI)) * SPEED_OF_LIGHT / (2 * CARRIER)
    window = np.hanning(DWELL_PULSES)
    powers = []
    for channel in CHANNELS[-2:]:
        power = np.empty((len(velocities), len(ranges)))
        for start in range(0, len(ranges), 48):
            block = slice(start, start + 48)
            weights = np.array([projection(channel, distance) for distance in ranges[block]])
            pulse_compressed = raw @ weights.conj().T
            spectrum = np.fft.fftshift(
                np.fft.fft(window[:, None] * pulse_compressed, axis=0), axes=0,
            ) / np.linalg.norm(window)
            thermal = np.maximum(np.sum(abs(weights) ** 2, axis=1), 1e-12)
            power[:, block] = abs(spectrum) ** 2 / thermal
        # Interpolate from each RF center to the nominal common velocity grid.
        native = velocities * CARRIER / (CARRIER + channel.pulse.center_frequency)
        aligned = np.array([np.interp(velocities, native, column) for column in power.T]).T
        powers.append(aligned)
    # Use both normalized powers in the near region and the long branch outside it.
    long_power, short_power = powers
    score = np.where(ranges[None, :] < 1200, 0.5 * (long_power + short_power), long_power)
    return velocities, score


def local_cfar(score, ranges):
    """Compare CA and OS training against the same candidate map."""
    training_per_side = 12
    training_count = 2 * training_per_side
    order = 18
    ca_scale = training_count * (PFA ** (-1 / training_count) - 1)
    factors = training_count - np.arange(order)
    os_scale = brentq(lambda scale: np.prod(factors / (factors + scale)) - PFA, 0, 200)
    ca = np.full_like(score, np.nan)
    os = np.full_like(score, np.nan)
    for index, distance in enumerate(ranges):
        # Truncated long pulses have broader range responses in the near region.
        guard = 12 if distance < 1200 else 3
        # Use one-sided training near a grid edge when needed.
        left = np.arange(index - guard - training_per_side, index - guard)
        right = np.arange(index + guard + 1, index + guard + training_per_side + 1)
        if left[0] < 0:
            reference = np.arange(right[0], right[0] + training_count)
        elif right[-1] >= len(ranges):
            reference = np.arange(left[-1] - training_count + 1, left[-1] + 1)
        else:
            reference = np.r_[left, right]
        training = score[:, reference]
        ca[:, index] = ca_scale * training.mean(axis=1)
        os[:, index] = os_scale * np.partition(training, order - 1, axis=1)[:, order - 1]
    peaks = score == maximum_filter(score, size=(3, 3), mode="nearest")
    return {"CA": peaks & (score > ca), "OS": peaks & (score > os)}


def associate(ranges, velocities, detections):
    rows = []
    for method, mask in detections.items():
        speed_index, range_index = np.nonzero(mask)
        for target in TARGETS:
            range_error = ranges[range_index] - target.distance
            speed_error = velocities[speed_index] - target.velocity
            eligible = (abs(range_error) <= 15) & (abs(speed_error) <= 2.0)
            indices = np.flatnonzero(eligible)
            best = None
            if len(indices):
                cost = (range_error[indices] / 15) ** 2 + (speed_error[indices] / 2.0) ** 2
                best = indices[np.argmin(cost)]
            rows.append({
                "method": method, "target_range_m": target.distance,
                "target_velocity_mps": target.velocity, "hit": best is not None,
                "point_range_m": "" if best is None else ranges[range_index[best]],
                "point_velocity_mps": "" if best is None else velocities[speed_index[best]],
                "total_points": len(range_index),
            })
    return rows


def save_map(path, ranges, velocities, score, detections):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharey=True)
    for axis, x_limits in zip(axes, ((90, 340), (340, 8200))):
        image = axis.pcolormesh(
            ranges, velocities, 10 * np.log10(np.maximum(score, 1e-8)),
            shading="nearest", cmap="viridis", vmin=-4, vmax=28,
        )
        for method, marker, color in (("CA", "o", "white"), ("OS", "s", "#65d6fa")):
            velocity_index, range_index = np.nonzero(detections[method])
            axis.scatter(ranges[range_index], velocities[velocity_index], s=38,
                         marker=marker, facecolors="none", edgecolors=color,
                         linewidths=0.8, label=f"{method} points")
        axis.scatter([target.distance for target in TARGETS], [target.velocity for target in TARGETS],
                     marker="+", c="#ff875d", s=72, label="Target truth")
        axis.set(xlim=x_limits, ylim=(-25, 25), xlabel="Range (m)", ylabel="Radial velocity (m/s)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=axes, label="Power relative to thermal noise (dB)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "benchmark_output")
    parser.add_argument("--seed", type=int, default=9127)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranges = np.arange(90, 8200, 7.5)
    velocities, score = range_velocity_power(mixed_dwell(args.seed), ranges)
    detections = local_cfar(score, ranges)
    write_csv(args.output / "mixed_scene_points.csv", associate(ranges, velocities, detections))
    save_map(args.output / "mixed_scene.png", ranges, velocities, score, detections)
    print(f"Saved mixed_scene_points.csv and mixed_scene.png to {args.output}")


if __name__ == "__main__":
    main()
