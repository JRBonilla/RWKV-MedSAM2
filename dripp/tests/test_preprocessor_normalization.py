import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dripp.config as config
from dripp.preprocessor import Preprocessor


class PreprocessorNormalizationTests(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger(f"{__name__}.{self._testMethodName}")
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        self.preprocessor = Preprocessor(
            target_size=(4, 4),
            min_mask_size=1,
            dataset_logger=logger,
            dataset_name="synthetic",
            background_value=0,
        )

    def test_mri_volume_uses_one_mapping_across_slices(self):
        volume = np.array(
            [
                [[0, 10, 20], [30, 40, 50]],
                [[0, 10, 60], [70, 80, 90]],
            ],
            dtype=np.float32,
        )

        normalized = self.preprocessor._normalize_mri_volume(volume)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized[0, 0, 1]), float(normalized[1, 0, 1]), places=6)
        self.assertEqual(float(normalized[0, 0, 0]), 0.0)
        self.assertEqual(float(normalized[1, 0, 0]), 0.0)

    def test_mri_volume_foreground_is_zscored_after_clipping(self):
        volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        normalized = self.preprocessor._normalize_mri_volume(volume)
        foreground = volume != 0

        self.assertAlmostEqual(float(normalized[foreground].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[foreground].std()), 1.0, places=6)
        self.assertEqual(float(normalized[~foreground][0]), 0.0)

    def test_mri_channels_are_normalized_independently(self):
        channel_a = np.array([[[0, 1], [2, 3]], [[0, 4], [5, 6]]], dtype=np.float32)
        channel_b = np.array([[[0, 100], [300, 700]], [[0, 900], [1500, 2200]]], dtype=np.float32)

        normalized_a = self.preprocessor._normalize_mri_volume(channel_a)
        normalized_b = self.preprocessor._normalize_mri_volume(channel_b)

        for source, normalized in ((channel_a, normalized_a), (channel_b, normalized_b)):
            foreground = source != 0
            self.assertAlmostEqual(float(normalized[foreground].mean()), 0.0, places=6)
            self.assertAlmostEqual(float(normalized[foreground].std()), 1.0, places=6)

    def test_mri_volume_zeros_nonfinite_and_handles_empty_or_constant_data(self):
        invalid = np.array([[[0, np.nan], [np.inf, -np.inf]], [[1, 2], [3, 4]]], dtype=np.float32)
        normalized_invalid = self.preprocessor._normalize_mri_volume(invalid)
        self.assertTrue(np.isfinite(normalized_invalid).all())
        self.assertTrue(np.all(normalized_invalid[0] == 0))

        normalized_empty = self.preprocessor._normalize_mri_volume(np.zeros((2, 2, 2), dtype=np.float32))
        self.assertTrue(np.all(normalized_empty == 0))

        constant = np.full((2, 2, 2), 7, dtype=np.float32)
        constant[0, 0, 0] = 0
        normalized_constant = self.preprocessor._normalize_mri_volume(constant)
        self.assertTrue(np.all(normalized_constant == 0))

    def test_2d_mri_normalization_retains_existing_per_image_behavior(self):
        image = np.array([[0, 1], [2, 3]], dtype=np.float32)
        foreground = image > 0
        clipped = np.clip(
            image,
            np.percentile(image[foreground], 0.5),
            np.percentile(image[foreground], 99.5),
        )
        values = clipped[foreground]
        expected = (clipped - values.mean()) / values.std()

        actual = self.preprocessor._normalize_intensity(image, "mri")

        np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_ct_normalization_is_unchanged(self):
        preprocessor = Preprocessor(
            target_size=(2, 2),
            ct_window=(100, 50),
            global_ct_stats=(50, 25),
            min_mask_size=1,
            dataset_logger=self.preprocessor.dataset_logger,
            dataset_name="synthetic-ct",
        )
        preprocessor._active_ct_profile = preprocessor._legacy_ct_profile
        image = np.array([[-10, 0], [50, 120]], dtype=np.float32)

        actual = preprocessor._normalize_intensity(image, "ct")
        expected = (np.clip(image, 0, 100) - 50) / 25

        np.testing.assert_allclose(actual, expected.astype(np.float32))


class PreprocessorColorOutputTests(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger(f"{__name__}.{self._testMethodName}")
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        self.preprocessor = Preprocessor(
            target_size=(2, 2),
            min_mask_size=1,
            dataset_logger=logger,
            dataset_name="synthetic",
        )

    def test_rgb_raster_round_trip_preserves_channel_order(self):
        rgb = np.array(
            [
                [[255, 0, 0], [0, 0, 255]],
                [[0, 255, 0], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(config.OUTPUT_FORMATS, {"2d_image": ".png"}):
                path = self.preprocessor._save_image(rgb, "2d", 0, "rgb-test", output_dir)
            reopened = np.asarray(Image.open(path).convert("RGB"))

        np.testing.assert_array_equal(reopened, rgb)

    def test_rgb_video_frame_round_trip_preserves_channel_order(self):
        rgb = np.array(
            [
                [[255, 0, 0], [0, 0, 255]],
                [[255, 255, 0], [0, 255, 255]],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(config.OUTPUT_FORMATS, {"video_frame": ".png"}):
                path = self.preprocessor._save_image(rgb, "video", 0, "video-rgb-test", output_dir)
            reopened = np.asarray(Image.open(path).convert("RGB"))

        np.testing.assert_array_equal(reopened, rgb)

    def test_grayscale_raster_output_is_unchanged(self):
        grayscale = np.array([[0, 64], [128, 255]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(config.OUTPUT_FORMATS, {"2d_image": ".png"}):
                path = self.preprocessor._save_image(grayscale, "2d", 0, "gray-test", output_dir)
            reopened = np.asarray(Image.open(path).convert("L"))

        np.testing.assert_array_equal(reopened, grayscale)

    def test_medical_volume_output_does_not_apply_color_conversion(self):
        image = np.array([[0, 64], [128, 255]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(config.OUTPUT_FORMATS, {"2d_image": ".nii.gz"}):
                path = self.preprocessor._save_image(image, "2d", 0, "medical-test", output_dir)
            reopened = sitk.GetArrayFromImage(sitk.ReadImage(path))

        np.testing.assert_array_equal(reopened, image)


if __name__ == "__main__":
    unittest.main()
