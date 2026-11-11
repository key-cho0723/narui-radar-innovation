import unittest
from dataclasses import replace

import numpy as np

from radar_demo import BANDWIDTH, LONG, SAMPLE_RATE, sample_echo
from radar_refinement import noise_whitening
from radar_scene import (
    CHANNELS, DWELL_PULSES, LATE_DUAL, LATE_SHORT, TIME,
    RangeVelocityCell, add_raw_path, arrivals, receive_gate,
)


class RadarSceneTests(unittest.TestCase):
    def test_late_receive_gate_opens_on_exact_adc_ticks(self):
        gate = receive_gate(LATE_DUAL)
        for time_us, is_open in ((10.14, False), (10.15, True), (64.99, True),
                                 (65, False), (65.54, False), (65.55, True)):
            with self.subTest(time_us=time_us):
                self.assertEqual(gate[round(time_us * 1e-6 * SAMPLE_RATE)], is_open)

    def test_projected_cell_matches_direct_adc_dwell_including_previous_pri(self):
        for distance in (100, 2200, 4000):
            with self.subTest(distance=distance):
                velocity = 3.9
                channels = CHANNELS[-2:]
                cell = RangeVelocityCell(distance, velocity, channels)
                subband_response = np.array([0.7 + 0.2j, -0.5 + 0.4j])
                raw = np.zeros((DWELL_PULSES, len(TIME)), dtype=complex)
                add_raw_path(raw, LATE_DUAL, distance, velocity, 1, subband_response)
                compressed = raw @ cell.fast_weights.conj().T
                from_adc = np.sum(cell.slow_weights.conj().T * compressed, axis=0)
                from_cell = cell.path_response(distance, velocity) @ subband_response
                np.testing.assert_allclose(from_cell, from_adc, rtol=1e-11, atol=1e-11)

        # The short pulse straddles the PRI at 2200 m, and its 4 km echo
        # occurs entirely in the following PRI.
        self.assertEqual(len(list(arrivals(LATE_SHORT, 2200, 3.9))), 2)
        self.assertEqual(len(list(arrivals(LATE_SHORT, 4000, 3.9))), 1)

    def test_longer_dwell_has_correct_slow_time_gain(self):
        cell = RangeVelocityCell(250, 3.9, CHANNELS[-2:], pulse_count=128)
        raw = np.zeros((128, len(TIME)), dtype=complex)
        subband = np.array([1.0, 0.0])
        add_raw_path(raw, LATE_DUAL, 250, 3.9, 1, subband)
        raw_output = np.sum(cell.slow_weights.conj().T * (raw @ cell.fast_weights.conj().T), axis=0)
        np.testing.assert_allclose(raw_output, cell.path_response(250, 3.9) @ subband, atol=1e-10)

    def test_noise_whitening_handles_complex_covariance_and_singular_views(self):
        factor = np.array([[1.0, 1j], [0.2j, 2.0], [1.0, 1j]])
        covariance = factor @ factor.conj().T
        whitening = noise_whitening(covariance)
        self.assertEqual(whitening.shape, (3, 2))
        np.testing.assert_allclose(whitening.T @ covariance @ whitening.conj(), np.eye(2),
                                   atol=1e-12)

    def test_tail_sweep_keeps_unit_envelope_and_continuous_frequency(self):
        pulse = replace(LONG, tail_duration=1.6e-6, tail_bandwidth=12e6)
        boundary = pulse.duration - pulse.tail_duration
        samples = boundary + np.arange(-2, 3) / SAMPLE_RATE
        echo = sample_echo(samples, pulse, 0)
        np.testing.assert_allclose(abs(echo), 1)
        frequency = np.angle(echo[1:] * echo[:-1].conj()) * SAMPLE_RATE / (2 * np.pi)
        boundary_frequency = pulse.center_frequency + BANDWIDTH / 2 - pulse.tail_bandwidth
        self.assertLess(np.max(abs(frequency - boundary_frequency)), 120e3)


if __name__ == "__main__":
    unittest.main()
