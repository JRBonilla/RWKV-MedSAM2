"""
Test typed quality options and 3D component extent and slice-area filters.
"""

import configparser
import os
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from dripp import config
from dripp.helpers import (
    DEFAULT_PREPROCESSING_OPTIONS,
    get_preprocessing_options,
    normalize_3d_quality_policy,
    parse_preprocessing_options,
)
from dripp.preprocessor import Preprocessor


class QualityOptionTests(unittest.TestCase):
    """
    Check default resolution, typed parsing, and validation of preprocessing options.
    """

    def test_packaged_defaults_are_optional_and_active_config_is_enabled(self):
        """
        Verify that defaults are optional and the active configuration enables quality
        filters.

        Args:
            None.

        Returns:
            None.
        """
        parser = configparser.ConfigParser()
        parser.read(config.DEFAULT_CONFIG_PATH)
        self.assertEqual(parser.getint("preprocessing", "min_slice_area_px_3d"), 0)
        self.assertEqual(parser.getint("preprocessing", "min_qualified_slices_3d"), 0)
        self.assertFalse(
            parser.getboolean("preprocessing", "require_contiguous_qualified_slices_3d")
        )
        self.assertEqual(DEFAULT_PREPROCESSING_OPTIONS["min_component_voxels_3d"], 64)
        self.assertEqual(DEFAULT_PREPROCESSING_OPTIONS["mask_qc_downsample_3d"], 2)
        self.assertEqual(DEFAULT_PREPROCESSING_OPTIONS["min_slice_area_px_3d"], 32)
        self.assertAlmostEqual(
            DEFAULT_PREPROCESSING_OPTIONS["min_slice_area_fraction_3d"], 0.00153
        )
        self.assertEqual(DEFAULT_PREPROCESSING_OPTIONS["min_qualified_slices_3d"], 8)
        self.assertTrue(
            DEFAULT_PREPROCESSING_OPTIONS["require_contiguous_qualified_slices_3d"]
        )

    def test_numeric_options_and_subdataset_override_are_typed(self):
        """
        Verify numeric option types and subdataset override precedence.

        Args:
            None.

        Returns:
            None.
        """
        metadata = {
            "preprocessing_options": parse_preprocessing_options(
                "{min_component_voxels_3d: 80; mask_qc_downsample_3d: 2; "
                "min_slice_area_px_3d: 24; min_slice_area_fraction_3d: 0.01; "
                "min_qualified_slices_3d: 6; "
                "require_contiguous_qualified_slices_3d: true; "
                "share_masks_across_processed_images: true}"
                "[site-b] {min_qualified_slices_3d: 9}"
            )
        }
        options = get_preprocessing_options(metadata, "site-b", include_defaults=True)
        policy = normalize_3d_quality_policy(options, applied=True)
        self.assertEqual(policy["min_component_voxels_3d"], 80)
        self.assertEqual(policy["min_qualified_slices_3d"], 9)
        self.assertTrue(options["share_masks_across_processed_images"])
        self.assertEqual(policy["min_slice_area_fraction_3d"], 0.01)
        self.assertTrue(policy["applied"])
        self.assertEqual(policy["slice_axis_sampling_3d"], "full_resolution")

    def test_invalid_options_are_rejected(self):
        """
        Verify that unsupported or malformed preprocessing options are rejected.

        Args:
            None.

        Returns:
            None.
        """
        invalid_values = (
            "{min_component_voxels_3d: -1}",
            "{mask_qc_downsample_3d: 0}",
            "{min_slice_area_px_3d: -1}",
            "{min_slice_area_fraction_3d: 1.1}",
            "{min_qualified_slices_3d: -1}",
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_preprocessing_options(value)

        with self.assertRaisesRegex(ValueError, "requires"):
            normalize_3d_quality_policy(
                {
                    "min_component_voxels_3d": 0,
                    "mask_qc_downsample_3d": 2,
                    "min_slice_area_px_3d": 0,
                    "min_slice_area_fraction_3d": 0.0,
                    "min_qualified_slices_3d": 0,
                    "require_contiguous_qualified_slices_3d": True,
                }
            )


class ComponentExtentFilterTests(unittest.TestCase):
    """
    Check 3D component policy limits, axis selection, and slice-area sampling.
    """

    @staticmethod
    def _policy(**updates):
        """
        Build a validated quality policy with extent filters disabled unless overridden.

        Args:
            updates (dict): Policy or quality-counter fields overriding the fixture
                defaults.

        Returns:
            dict: Applied 3D quality policy containing the requested field overrides.
        """
        values = {
            "min_component_voxels_3d": 0,
            "mask_qc_downsample_3d": 2,
            "min_slice_area_px_3d": 0,
            "min_slice_area_fraction_3d": 0.0,
            "min_qualified_slices_3d": 0,
            "require_contiguous_qualified_slices_3d": False,
        }
        values.update(updates)

        return normalize_3d_quality_policy(values, applied=True)

    @staticmethod
    def _evaluate(labels, policy):
        """
        Evaluate a synthetic component-label volume under a quality policy.

        Args:
            labels (np.ndarray): Synthetic 3D connected-component label array.
            policy (dict): Validated 3D component quality thresholds and slice-axis
                sampling semantics.

        Returns:
            dict[int, str]: Rejected component labels mapped to quality-failure reasons.
        """
        image = sitk.GetImageFromArray(labels.astype(np.uint16))
        sizes = tuple(
            int(np.count_nonzero(labels == label))
            for label in range(1, int(labels.max()) + 1)
        )

        return Preprocessor._evaluate_3d_component_quality(image, sizes, policy)

    def test_all_extent_filters_can_be_disabled(self):
        """
        Verify that zero thresholds disable all component extent filters.

        Args:
            None.

        Returns:
            None.
        """
        labels = np.zeros((1, 2, 2), dtype=np.uint8)
        labels[0, 0, 0] = 1
        self.assertEqual(self._evaluate(labels, self._policy()), {})

    def test_factor_two_sampling_and_area_scaling(self):
        """
        Verify slice sampling and full-resolution area scaling.

        Args:
            None.

        Returns:
            None.
        """
        labels = np.zeros((10, 16, 16), dtype=np.uint8)
        labels[0:6, 0:8, 0:8] = 1
        policy = self._policy(
            mask_qc_downsample_3d=2,
            min_slice_area_px_3d=64,
            min_qualified_slices_3d=3,
            require_contiguous_qualified_slices_3d=True,
        )
        self.assertEqual(self._evaluate(labels, policy), {})

        shifted = np.zeros((10, 16, 16), dtype=np.uint8)
        shifted[1:6:2, 1:8, 1:8] = 1
        self.assertEqual(
            self._evaluate(shifted, policy)[1],
            "component_below_minimum_slice_area_3d",
        )

    def test_slice_area_slice_count_and_contiguity_reasons(self):
        """
        Verify distinct rejection reasons for area, slice count, and contiguity limits.

        Args:
            None.

        Returns:
            None.
        """
        area_labels = np.zeros((10, 16, 16), dtype=np.uint8)
        area_labels[0, :4, :4] = 1
        self.assertEqual(
            self._evaluate(area_labels, self._policy(min_slice_area_px_3d=32))[1],
            "component_below_minimum_slice_area_3d",
        )

        count_labels = np.zeros((10, 16, 16), dtype=np.uint8)
        count_labels[0:2, :4, :4] = 1
        self.assertEqual(
            self._evaluate(count_labels, self._policy(min_qualified_slices_3d=3))[1],
            "component_below_minimum_slices_3d",
        )

        scattered = np.zeros((10, 16, 16), dtype=np.uint8)
        scattered[[0, 4, 8], :4, :4] = 1
        policy = self._policy(
            min_qualified_slices_3d=3,
            require_contiguous_qualified_slices_3d=True,
        )
        self.assertEqual(
            self._evaluate(scattered, policy)[1],
            "component_below_minimum_contiguous_slices_3d",
        )
        noncontiguous_policy = self._policy(
            min_qualified_slices_3d=3,
            require_contiguous_qualified_slices_3d=False,
        )
        self.assertEqual(self._evaluate(scattered, noncontiguous_policy), {})

        contiguous = np.zeros((10, 16, 16), dtype=np.uint8)
        contiguous[0:5, :4, :4] = 1
        self.assertEqual(self._evaluate(contiguous, policy), {})

    def test_downsampling_preserves_the_sequence_axis(self):
        """
        Verify that downsampling preserves the component's slice sequence.

        Args:
            None.

        Returns:
            None.
        """
        labels = np.zeros((8, 16, 16), dtype=np.uint8)
        labels[:, :8, :8] = 1
        policy = self._policy(
            mask_qc_downsample_3d=2,
            min_slice_area_px_3d=16,
            min_qualified_slices_3d=8,
            require_contiguous_qualified_slices_3d=True,
        )
        self.assertEqual(self._evaluate(labels, policy), {})

    def test_highest_resolution_axis_matches_rwkv_heuristic(self):
        """
        Verify that the selected sequence axis matches the RWKV resolution heuristic.

        Args:
            None.

        Returns:
            None.
        """
        labels = np.zeros((8, 4, 10), dtype=np.uint8)
        labels[0, 0:3, 0:2] = 1
        policy = self._policy(
            mask_qc_downsample_3d=1,
            min_qualified_slices_3d=3,
            require_contiguous_qualified_slices_3d=True,
        )
        self.assertEqual(self._evaluate(labels, policy), {})

    def test_rejected_component_is_not_written_and_mixed_group_keeps_valid_one(self):
        """
        Verify that rejected components are skipped while valid components are saved.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as root:
            image = np.ones((10, 16, 16), dtype=np.float32)
            mask = np.zeros((10, 16, 16), dtype=np.uint8)
            mask[0:5, :4, :4] = 1
            mask[7, 10:12, 10:12] = 2
            image_path = os.path.join(root, "image.nii.gz")
            mask_path = os.path.join(root, "mask.nii.gz")
            image_dir = os.path.join(root, "images")
            mask_dir = os.path.join(root, "masks")
            os.makedirs(image_dir)
            os.makedirs(mask_dir)
            sitk.WriteImage(sitk.GetImageFromArray(image), image_path)
            sitk.WriteImage(sitk.GetImageFromArray(mask), mask_path)

            preprocessor = Preprocessor(
                target_size=(16, 16),
                min_component_voxels_3d=0,
                dataset_name="test",
            )
            metadata = preprocessor.preprocess_group(
                "default",
                "3D",
                [image_path],
                [mask_path],
                "mri",
                image_dir,
                mask_dir,
                "mixed",
                {},
                {
                    "min_component_voxels_3d": 0,
                    "mask_qc_downsample_3d": 2,
                    "min_slice_area_px_3d": 1,
                    "min_slice_area_fraction_3d": 0.0,
                    "min_qualified_slices_3d": 3,
                    "require_contiguous_qualified_slices_3d": True,
                },
            )
            self.assertEqual(len(metadata["proc_masks"]), 1)
            self.assertEqual(metadata["qc"]["components_3d_found"], 2)
            self.assertEqual(metadata["qc"]["components_3d_rejected"], 1)
            self.assertEqual(
                metadata["qc"]["components_3d_rejected_by_reason"],
                {"component_below_minimum_slices_3d": 1},
            )
            self.assertEqual(len(os.listdir(mask_dir)), 1)


if __name__ == "__main__":
    unittest.main()
