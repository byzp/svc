import unittest

import numpy as np
from scipy.signal import chirp

from modules.F0Predictor.F0Predictor import F0Predictor
from modules.F0Predictor.PMF0Predictor import PMF0Predictor


class AutocorrelationPitchTests(unittest.TestCase):
    def setUp(self):
        self.predictor = PMF0Predictor()
        self.time = np.arange(44100) / 44100

    def test_pitch_range_and_frame_alignment(self):
        for frequency in (50, 80, 110, 220, 440, 880, 1100):
            with self.subTest(frequency=frequency):
                audio = np.sin(2 * np.pi * frequency * self.time)
                f0, uv = self.predictor.compute_f0_uv(audio)
                self.assertEqual(f0.shape, (len(audio) // 512,))
                self.assertEqual(f0.shape, uv.shape)
                self.assertGreater(uv[4:-4].mean(), 0.95)
                error_cents = np.abs(1200 * np.log2(f0[4:-4] / frequency))
                self.assertLess(np.median(error_cents), 3)

    def test_missing_fundamental(self):
        audio = sum(amplitude * np.sin(2 * np.pi * 180 * harmonic * self.time)
                    for harmonic, amplitude in ((2, 0.7), (3, 0.4), (4, 0.2)))
        f0, uv = self.predictor.compute_f0_uv(audio)
        self.assertGreater(uv.mean(), 0.95)
        self.assertLess(abs(np.median(f0) - 180), 1)

    def test_continuous_pitch_change(self):
        audio = chirp(self.time, 100, self.time[-1], 700)
        f0, _ = self.predictor.compute_f0_uv(audio)
        expected = 100 + 600 * np.arange(len(f0)) * 512 / 44100 / self.time[-1]
        error_cents = np.abs(1200 * np.log2(f0[4:-4] / expected[4:-4]))
        self.assertLess(np.quantile(error_cents, 0.95), 10)

    def test_uv_preserves_silence_after_interpolation(self):
        voiced = np.sin(2 * np.pi * 220 * self.time[:22050])
        audio = np.concatenate((np.zeros(11025), voiced, np.zeros(11025)))
        f0, uv = self.predictor.compute_f0_uv(audio)
        np.testing.assert_array_equal(uv[:15], 0)
        np.testing.assert_array_equal(uv[-15:], 0)
        np.testing.assert_array_equal(uv[30:55], 1)
        np.testing.assert_allclose(f0, 220, atol=2)

    def test_empty_short_silent_and_noise_inputs(self):
        rng = np.random.default_rng(42)
        for audio in (np.zeros(0), np.zeros(12), np.zeros(44100),
                      np.ones(44100), rng.normal(0, 0.1, 44100)):
            with self.subTest(samples=len(audio)):
                f0, uv = self.predictor.compute_f0_uv(audio)
                self.assertEqual(f0.shape, (len(audio) // 512,))
                np.testing.assert_array_equal(f0, 0)
                np.testing.assert_array_equal(uv, 0)
                self.assertIsInstance(self.predictor.compute_f0(audio), np.ndarray)

    def test_requested_length_and_invalid_input(self):
        audio = np.sin(2 * np.pi * 220 * self.time)
        frames = len(audio) // 512
        for length in (frames - 3, frames, frames + 3):
            f0, uv = self.predictor.compute_f0_uv(audio, p_len=length)
            self.assertEqual(f0.shape, (length,))
            self.assertEqual(uv.shape, (length,))
        for invalid in (np.zeros((2, 44100)), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                self.predictor.compute_f0(invalid)
        with self.assertRaises(ValueError):
            self.predictor.compute_f0(audio, p_len=frames + 4)

    def test_single_voiced_frame(self):
        f0, uv = F0Predictor.interpolate_f0([0, 220, 0])
        np.testing.assert_array_equal(f0, [220, 220, 220])
        np.testing.assert_array_equal(uv, [0, 1, 0])
        f0, uv = F0Predictor().post_process_f0([0, 220, 0], 6)
        np.testing.assert_array_equal(f0, [220] * 6)
        np.testing.assert_array_equal(uv, [0, 0, 1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
