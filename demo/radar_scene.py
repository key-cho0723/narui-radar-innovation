"""Fast-time matched projections and slow-time point-scatterer responses."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.signal import firwin

from radar_demo import (
    LONG, MODES, PRI, RECOVERY, SAMPLE_RATE, SHORT, SPEED_OF_LIGHT,
    Mode, Pulse, filter_channel, sample_echo,
)


CARRIER = 15e9
DWELL_PULSES = 64
TIME = np.arange(round(PRI * SAMPLE_RATE)) / SAMPLE_RATE
SLOW_TIME = np.arange(DWELL_PULSES) * PRI
LOW_PASS = firwin(81, 15e6, fs=SAMPLE_RATE)
RADAR_HEIGHT = 6.0
FULL_DWELL = np.ones(DWELL_PULSES, dtype=bool)
EVEN_PULSES = np.arange(DWELL_PULSES) % 2 == 0


@dataclass(frozen=True)
class Channel:
    name: str
    mode: Mode
    pulse: Pulse
    active: np.ndarray
    use_gated_template: bool = True


@dataclass(frozen=True)
class View:
    channel: Channel
    distance: float
    velocity: float
    taper: Literal["hann", "rectangular"] = "hann"


FIXED_LONG, FIXED_SHORT, DUAL = MODES
LATE_SHORT = replace(SHORT, start=65e-6)
LATE_DUAL = Mode("Late dual", (LONG, LATE_SHORT))
CHANNELS = (
    Channel("fixed_long_ppc", FIXED_LONG, LONG, FULL_DWELL),
    Channel("fixed_long_full", FIXED_LONG, LONG, FULL_DWELL, False),
    Channel("fixed_short", FIXED_SHORT, FIXED_SHORT.pulses[0], FULL_DWELL),
    Channel("alternate_long", FIXED_LONG, LONG, EVEN_PULSES),
    Channel("alternate_short", FIXED_SHORT, FIXED_SHORT.pulses[0], ~EVEN_PULSES),
    Channel("dual_long", DUAL, LONG, FULL_DWELL),
    Channel("dual_short", DUAL, SHORT, FULL_DWELL),
    Channel("late_long", LATE_DUAL, LONG, FULL_DWELL),
    Channel("late_short", LATE_DUAL, LATE_SHORT, FULL_DWELL),
)
DUAL_SLICE = slice(5, 7)
LATE_SLICE = slice(7, 9)


def receive_gate(mode):
    gate = np.ones(TIME.size, dtype=bool)
    for pulse in mode.pulses:
        start = round(pulse.start * SAMPLE_RATE)
        end = round((pulse.end + RECOVERY) * SAMPLE_RATE)
        gate[start:end] = False
    return gate


def arrivals(pulse, distance, velocity, pulse_count=DWELL_PULSES):
    """Current and previous-PRI contributions on the receive time axis."""
    phase = path_voltage(pulse, distance)
    for pri_offset in (0, 1):
        echo = sample_echo(TIME + pri_offset * PRI, pulse, distance)
        if np.any(echo):
            yield phase * echo, steering(pulse, velocity, pri_offset, pulse_count)


def projection(channel, distance):
    """Return a raw-ADC projection vector equivalent to FIR then gated matching."""
    gate = receive_gate(channel.mode)
    oscillator = np.exp(-2j * np.pi * channel.pulse.center_frequency * TIME)
    echo = sample_echo(TIME, channel.pulse, distance)
    echo += sample_echo(TIME + PRI, channel.pulse, distance)
    replica = echo * gate if channel.use_gated_template else echo
    template = filter_channel(replica, oscillator, LOW_PASS)
    norm = np.linalg.norm(template)
    if norm == 0:
        return template
    # The symmetric centered FIR is its own adjoint. Move it from the ADC
    # samples onto the template, then apply the adjoint of downconversion.
    adjoint = np.convolve(template / norm, LOW_PASS, mode="same")
    return gate * oscillator.conj() * adjoint


def steering(pulse, velocity, pri_offset=0, pulse_count=DWELL_PULSES):
    wavelength = SPEED_OF_LIGHT / (CARRIER + pulse.center_frequency)
    time = SLOW_TIME if pulse_count == DWELL_PULSES else np.arange(pulse_count) * PRI
    return np.exp(
        4j * np.pi * velocity * (time + pulse.start - pri_offset * PRI) / wavelength
    )


def slow_time_template(channel, velocity, taper="hann", pulse_count=DWELL_PULSES):
    window = np.hanning(pulse_count) if taper == "hann" else np.ones(pulse_count)
    window *= np.resize(channel.active, pulse_count)
    return window * steering(channel.pulse, velocity, pulse_count=pulse_count) / np.linalg.norm(window)


def path_voltage(pulse, distance):
    """Complex carrier phase for a path with one-way-equivalent distance."""
    wavelength = SPEED_OF_LIGHT / (CARRIER + pulse.center_frequency)
    return np.exp(-4j * np.pi * distance / wavelength)


def reflected_paths(distance, target_height, reflection):
    """One- and two-reflection paths from the flat-ground image model."""
    image_distance = np.sqrt(distance**2 + 4 * RADAR_HEIGHT * target_height)
    one_reflection = (distance + image_distance) / 2
    return (
        (distance, 1.0),
        (one_reflection, 2 * reflection * distance / image_distance),
        (image_distance, (reflection * distance / image_distance) ** 2),
    )


class RangeVelocityCell:
    """Project a dwell onto one range–velocity cell for each comparator."""

    def __init__(self, distance, velocity, channels=CHANNELS, extra_views=(),
                 pulse_count=DWELL_PULSES):
        views = tuple(View(channel, distance, velocity) for channel in channels) + tuple(extra_views)
        self.channels = tuple(view.channel for view in views)
        self.pulse_count = pulse_count
        fast_cache = {}
        slow_cache = {}
        fast_weights = []
        slow_weights = []
        for view in views:
            fast_key = (view.channel.name, view.distance)
            if fast_key not in fast_cache:
                fast_cache[fast_key] = projection(view.channel, view.distance)
            slow_key = (view.channel.name, view.velocity, view.taper)
            if slow_key not in slow_cache:
                slow_cache[slow_key] = slow_time_template(
                    view.channel, view.velocity, view.taper, pulse_count,
                )
            fast_weights.append(fast_cache[fast_key])
            slow_weights.append(slow_cache[slow_key])
        self.fast_weights = np.array(fast_weights)
        self.slow_weights = np.array(slow_weights)

    def noise_factor(self):
        # Each channel uses the same raw white-noise dwell. The separable
        # Gram matrix gives its exact covariance after both projections.
        fast_covariance = self.fast_weights.conj() @ self.fast_weights.T
        slow_covariance = self.slow_weights.conj() @ self.slow_weights.T
        values, vectors = np.linalg.eigh(fast_covariance * slow_covariance)
        return vectors * np.sqrt(np.maximum(values, 0))

    def path_response(self, distance, velocity):
        """Return the response per processing channel and transmitted subband."""
        response = np.zeros((len(self.channels), 2), dtype=complex)
        for pulse_index, pulse_name in enumerate(("long", "short")):
            source_echoes = {}
            for index, channel in enumerate(self.channels):
                for pulse in channel.mode.pulses:
                    if pulse.name != pulse_name:
                        continue
                    if pulse not in source_echoes:
                        source_echoes[pulse] = list(
                            arrivals(pulse, distance, velocity, self.pulse_count),
                        )
                    for fast_echo, slow_echo in source_echoes[pulse]:
                        response[index, pulse_index] += (
                            np.vdot(self.fast_weights[index], fast_echo)
                            * np.vdot(self.slow_weights[index], slow_echo)
                        )
        return response

    def nominal_dual_gain(self, distance, velocity):
        response = self.path_response(distance, velocity)
        return tuple(np.abs(np.diag(response[group])) ** 2 for group in (DUAL_SLICE, LATE_SLICE))


def add_raw_path(raw, mode, distance, velocity, amplitude, subband_response):
    """Add one path to a raw ADC dwell before applying the receive gate."""
    for pulse in mode.pulses:
        band_index = int(pulse.name == "short")
        for echo, slow_echo in arrivals(pulse, distance, velocity, len(raw)):
            raw += amplitude * subband_response[band_index] * slow_echo[:, None] * echo
