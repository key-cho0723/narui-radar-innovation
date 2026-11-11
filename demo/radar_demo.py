"""Compare three radar transmit modes with one stationary point target per run."""

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import firwin

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPEED_OF_LIGHT = 299_792_458.0
SAMPLE_RATE = 100e6
PRI = 80e-6
RECOVERY = 0.15e-6
BANDWIDTH = 20e6
TARGET_RANGES = (100, 142, 1582, 4000)
ENERGY_SCALE = 1e6 / SAMPLE_RATE


@dataclass(frozen=True)
class Pulse:
    name: str
    start: float
    duration: float
    center_frequency: float
    tail_duration: float = 0.0
    tail_bandwidth: float = 0.0

    @property
    def end(self):
        return self.start + self.duration


@dataclass(frozen=True)
class Mode:
    name: str
    pulses: tuple[Pulse, ...]

    @property
    def receive_start(self):
        return max(pulse.end for pulse in self.pulses) + RECOVERY


LONG = Pulse("long", 0.0, 10e-6, -15e6)
SHORT = Pulse("short", LONG.end, 0.40e-6, 15e6)
MODES = (
    Mode("Fixed long", (LONG,)),
    Mode("Fixed short", (replace(SHORT, start=0.0),)),
    Mode("Dual", (LONG, SHORT)),
)
COLORS = {
    ("Fixed long", "long"): "#45638a",
    ("Fixed short", "short"): "#d27433",
    ("Dual", "long"): "#12867c",
    ("Dual", "short"): "#9d4377",
}


def sample_echo(time, pulse, distance):
    """Evaluate the delayed complex LFM directly at the ADC sample times."""
    delay = 2 * distance / SPEED_OF_LIGHT
    pulse_time = time - (pulse.start + delay)
    inside_pulse = (pulse_time >= 0) & (pulse_time < pulse.duration)
    local_time = pulse_time[inside_pulse]
    start_frequency = pulse.center_frequency - BANDWIDTH / 2
    if pulse.tail_duration:
        # Two contiguous sweeps with a shared boundary frequency and phase.
        body_duration = pulse.duration - pulse.tail_duration
        body_bandwidth = BANDWIDTH - pulse.tail_bandwidth
        body_time = np.minimum(local_time, body_duration)
        tail_time = np.maximum(local_time - body_duration, 0)
        phase_cycles = (
            start_frequency * local_time + body_bandwidth / body_duration * body_time**2 / 2
            + body_bandwidth * tail_time
            + pulse.tail_bandwidth / pulse.tail_duration * tail_time**2 / 2
        )
    else:
        phase_cycles = start_frequency * local_time + BANDWIDTH / pulse.duration * local_time**2 / 2
    phase = 2 * np.pi * phase_cycles
    echo = np.zeros(time.shape, dtype=complex)
    echo[inside_pulse] = np.exp(1j * phase)
    return echo


def visible_duration(mode, pulse, distance):
    """Return the continuous-time intersection of the echo and receive windows."""
    delay = 2 * distance / SPEED_OF_LIGHT
    start = np.maximum(mode.receive_start, pulse.start + delay)
    end = np.minimum(PRI, pulse.end + delay)
    return np.maximum(0, end - start)


def filter_channel(samples, oscillator, low_pass):
    # The centered convolution removes the FIR's constant group delay.
    return np.convolve(samples * oscillator, low_pass, mode="same")


def sample_energy(samples):
    return float(np.vdot(samples, samples).real)


def matched_power(observation, template):
    energy = sample_energy(template)
    if energy == 0:
        return np.nan
    return float(abs(np.vdot(template, observation)) ** 2 / energy)


def simulate(time, low_pass):
    """Run each mode and return checkpoint metrics and local range profiles."""
    rows = []
    profiles = {}
    for mode in MODES:
        # All configured gates align with ADC ticks. Compare integer ticks so
        # floating-point rounding cannot drop the first open sample.
        first_receive_sample = round(mode.receive_start * SAMPLE_RATE)
        gate = np.arange(time.size) >= first_receive_sample
        for distance in TARGET_RANGES:
            gated_echoes = {
                pulse.name: sample_echo(time, pulse, distance) * gate
                for pulse in mode.pulses
            }
            received = sum(gated_echoes.values())
            for pulse in mode.pulses:
                oscillator = np.exp(-2j * np.pi * pulse.center_frequency * time)
                observation = filter_channel(received, oscillator, low_pass)
                desired = gated_echoes[pulse.name]
                template = filter_channel(desired, oscillator, low_pass)
                sample_count = int(np.count_nonzero(desired))
                # Report energy in equivalent microseconds at unit sample power.
                rows.append({
                    "mode": mode.name,
                    "channel": pulse.name,
                    "range_m": distance,
                    "receive_start_us": mode.receive_start * 1e6,
                    "visible_us": float(visible_duration(mode, pulse, distance)) * 1e6,
                    "visible_samples": sample_count,
                    "filtered_energy_us": sample_energy(template) * ENERGY_SCALE,
                    "other_channel_energy_us": (
                        sample_energy(observation - template) * ENERGY_SCALE
                    ),
                    "matched_power_us": matched_power(observation, template) * ENERGY_SCALE,
                })

                if sample_count == 0:
                    continue
                # A local search at a fixed 1 m spacing, centered on the test range.
                candidates = distance + np.arange(-80, 81)
                power = np.empty(candidates.size)
                for index, candidate in enumerate(candidates):
                    replica = sample_echo(time, pulse, candidate) * gate
                    candidate_template = filter_channel(replica, oscillator, low_pass)
                    power[index] = matched_power(observation, candidate_template)
                profiles[(mode.name, pulse.name, distance)] = (candidates, power)
    return rows, profiles


def save_timing(output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(9, 5), layout="constrained")
    for axis, mode in zip(axes, MODES):
        for pulse in mode.pulses:
            axis.broken_barh(
                [(pulse.start * 1e6, pulse.duration * 1e6)], (0, 1),
                facecolors=COLORS[(mode.name, pulse.name)],
            )
            axis.text(
                (pulse.start + pulse.duration / 2) * 1e6, 1.06, pulse.name,
                ha="center", fontsize=9,
            )
        axis.broken_barh(
            [((mode.receive_start - RECOVERY) * 1e6, RECOVERY * 1e6)],
            (0, 1), facecolors="#d1d5db",
        )
        axis.broken_barh(
            [(mode.receive_start * 1e6, (PRI - mode.receive_start) * 1e6)],
            (0, 1), facecolors="#b6ddd7",
        )
        axis.text(
            mode.receive_start * 1e6 + 0.2, 0.5,
            f"RX from {mode.receive_start * 1e6:.2f} us", va="center",
        )
        axis.set(title=mode.name, xlim=(-0.4, 14), ylim=(0, 1.5), yticks=[])
    axes[-1].set_xlabel("Time since PRI start (us). All receive windows end at 80 us.")
    fig.savefig(output_dir / "timing.png", dpi=150)
    plt.close(fig)


def save_energy(output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, upper_range in zip(axes, (250, 12_000)):
        distances = np.linspace(0, upper_range, 1201)
        for mode in MODES:
            for pulse in mode.pulses:
                axis.plot(
                    distances, visible_duration(mode, pulse, distances) * 1e6,
                    color=COLORS[(mode.name, pulse.name)],
                    linestyle="--" if mode.name == "Dual" else "-",
                    label=f"{mode.name} / {pulse.name}",
                )
        axis.set(xlabel="Range (m)", ylabel="Visible energy (unit power x us)")
        axis.grid(alpha=0.25)
    axes[0].axvline(100, color="0.6", linewidth=1)
    axes[0].axvline(142, color="0.6", linewidth=1)
    axes[1].legend(loc="center")
    fig.savefig(output_dir / "visible_energy.png", dpi=150)
    plt.close(fig)


def save_profiles(output_dir, profiles):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, target_range in zip(axes.flat, TARGET_RANGES):
        for (mode, channel, distance), (candidates, power) in profiles.items():
            if distance != target_range:
                continue
            # Common absolute scale preserves energy differences between modes.
            power_us = power * ENERGY_SCALE
            axis.plot(
                candidates - distance, 10 * np.log10(np.maximum(power_us, 1e-8)),
                color=COLORS[(mode, channel)],
                linestyle="--" if mode == "Dual" else "-",
                label=f"{mode} / {channel}",
            )
        axis.axvline(0, color="0.65", linewidth=1)
        axis.set(
            title=f"Target at {target_range} m", xlabel="Candidate range error (m)",
            ylabel="Matched power (dB re 1 unit-power us)", ylim=(-60, 12),
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="lower center", fontsize=8)
    fig.savefig(output_dir / "pulse_compression.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "output",
        help="Directory for the CSV and PNG files (default: demo/output).",
    )
    output_dir = parser.parse_args().output
    output_dir.mkdir(parents=True, exist_ok=True)

    time = np.arange(round(PRI * SAMPLE_RATE)) / SAMPLE_RATE
    low_pass = firwin(numtaps=81, cutoff=15e6, fs=SAMPLE_RATE)
    rows, profiles = simulate(time, low_pass)
    with (output_dir / "checkpoints.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_timing(output_dir)
    save_energy(output_dir)
    save_profiles(output_dir, profiles)
    print(f"Saved checkpoints.csv and three PNGs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
