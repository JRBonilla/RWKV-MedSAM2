"""
Test teacher input normalization, shared-memory estimates, volume windows, and source isolation.
"""

import unittest

import numpy as np
import torch

from rwkv_medsam2.utils.teacher_refinement import (
    assert_teacher_shared_memory_capacity,
    assert_disjoint_sources,
    checkpoint_selection_score,
    extract_contiguous_volume_window,
    highest_resolution_axis,
    normalize_sam2_images,
    select_contiguous_foreground_window,
    teacher_prefetch_memory_requirement,
)


class TeacherInputNormalizationTests(unittest.TestCase):
    """
    Test byte-scale RGB normalization for teacher inputs.
    """

    def test_normalization_uses_255_scale_and_imagenet_statistics(self):
        """
        Verify teacher normalization uses byte-scale intensities and ImageNet statistics.

        Args:
            None.

        Returns:
            None.
        """
        image = torch.zeros(1, 3, 2, 2)
        image[:, 0] = 255.0 * 0.485
        image[:, 1] = 255.0 * 0.456
        image[:, 2] = 255.0 * 0.406
        normalized = normalize_sam2_images(image)
        self.assertTrue(
            torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6)
        )

    def test_normalization_rejects_non_rgb_input(self):
        """
        Verify teacher normalization rejects tensors without three RGB channels.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "channel dimension 3"):
            normalize_sam2_images(torch.zeros(1, 1, 8, 8))


class TeacherSharedMemoryTests(unittest.TestCase):
    """
    Test teacher prefetch memory estimates and capacity checks.
    """

    def test_compact_two_worker_configuration_fits_256_mib(self):
        """
        Verify compact two-worker prefetching fits a 256 MiB shared-memory budget.

        Args:
            None.

        Returns:
            None.
        """
        requirement = teacher_prefetch_memory_requirement(
            batch_size=3,
            sequence_length=8,
            height=512,
            width=512,
            num_workers=2,
            prefetch_factor=2,
        )
        self.assertEqual(requirement["batch_bytes"], 24 * 1024**2)
        self.assertEqual(requirement["queue_slots"], 5)
        self.assertEqual(requirement["required_bytes"], 150 * 1024**2)
        result = assert_teacher_shared_memory_capacity(
            batch_size=3,
            sequence_length=8,
            height=512,
            width=512,
            num_workers=2,
            prefetch_factor=2,
            available_bytes=256 * 1024**2,
        )
        self.assertEqual(result["available_bytes"], 256 * 1024**2)

    def test_float_worker_transport_is_rejected_for_256_mib(self):
        """
        Verify float32 worker transport exceeds the configured 256 MiB memory budget.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(RuntimeError, "Unsafe teacher DataLoader"):
            assert_teacher_shared_memory_capacity(
                batch_size=3,
                sequence_length=8,
                height=512,
                width=512,
                num_workers=2,
                prefetch_factor=2,
                image_itemsize=4,
                mask_itemsize=4,
                available_bytes=256 * 1024**2,
            )

    def test_invalid_worker_prefetch_is_rejected(self):
        """
        Verify rejection of invalid worker and prefetch settings.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "prefetch_factor"):
            teacher_prefetch_memory_requirement(
                batch_size=3,
                sequence_length=8,
                height=512,
                width=512,
                num_workers=2,
                prefetch_factor=0,
            )


class TeacherVolumeWindowTests(unittest.TestCase):
    """
    Test contiguous foreground-rich teacher volume windows.
    """

    def test_highest_resolution_axis_matches_rwkv_rule(self):
        """
        Verify teacher windows use the RWKV backbone's highest-resolution slicing rule.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(highest_resolution_axis((8, 20, 30)), 0)
        self.assertEqual(highest_resolution_axis((40, 8, 30)), 1)
        self.assertEqual(highest_resolution_axis((40, 30, 8)), 2)

    def test_window_is_contiguous_and_keeps_real_neighboring_slices(self):
        """
        Verify teacher windows contain consecutive slices from the original volume.

        Args:
            None.

        Returns:
            None.
        """
        image = np.zeros((12, 20, 30), dtype=np.float32)
        for index in range(image.shape[0]):
            image[index] = index

        mask = np.zeros_like(image, dtype=np.uint8)
        mask[3:11, 2:8, 2:8] = 1

        images, masks, axis, indices = extract_contiguous_volume_window(
            image, mask, seq_len=8
        )
        self.assertEqual(axis, 0)
        np.testing.assert_array_equal(indices, np.arange(3, 11))
        np.testing.assert_array_equal(images[:, 0, 0], np.arange(3, 11))
        self.assertTrue(np.all(masks.sum(axis=(1, 2)) > 0))

    def test_single_volume_pair_is_not_treated_as_one_repeated_slice(self):
        """
        Verify a volume pair is loaded as real neighboring slices rather than repeated input.

        Args:
            None.

        Returns:
            None.
        """
        mask = np.zeros((8, 16, 16), dtype=np.uint8)
        mask[:, 1:4, 1:4] = 1
        axis, indices = select_contiguous_foreground_window(mask, seq_len=8)
        self.assertEqual(axis, 0)
        np.testing.assert_array_equal(indices, np.arange(8))

    def test_empty_and_too_short_volumes_fail_explicitly(self):
        """
        Verify explicit failures for empty masks and volumes shorter than the window.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "no foreground"):
            select_contiguous_foreground_window(
                np.zeros((8, 16, 16), dtype=np.uint8), seq_len=8
            )
        with self.assertRaisesRegex(ValueError, "too short"):
            select_contiguous_foreground_window(
                np.ones((7, 16, 16), dtype=np.uint8), seq_len=8
            )


class TeacherValidationTests(unittest.TestCase):
    """
    Test source isolation and balanced checkpoint-selection scores.
    """

    def test_source_overlap_is_rejected(self):
        """
        Verify rejection of image-path overlap between training and validation.

        Args:
            None.

        Returns:
            None.
        """
        train = [{"sequence": [["/tmp/shared.png", "/tmp/train-mask.png"]]}]
        val = [{"sequence": [["/tmp/shared.png", "/tmp/val-mask.png"]]}]

        with self.assertRaisesRegex(RuntimeError, "source leakage"):
            assert_disjoint_sources(train, val)

    def test_source_group_overlap_is_rejected_even_when_image_paths_differ(self):
        """
        Verify rejection of shared source identities across different image paths.

        Args:
            None.

        Returns:
            None.
        """
        train = [
            {
                "source_identifier": "same-case",
                "sequence": [["/tmp/channel-a.png", "/tmp/train-mask.png"]],
            }
        ]
        val = [
            {
                "source_identifier": "same-case",
                "sequence": [["/tmp/channel-b.png", "/tmp/val-mask.png"]],
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "shared source groups"):
            assert_disjoint_sources(train, val)

    def test_equal_identifiers_in_different_datasets_are_not_leakage(self):
        """
        Verify dataset namespaces prevent false source-identity collisions.

        Args:
            None.

        Returns:
            None.
        """
        train = [
            {
                "dataset": "dataset-a",
                "subdataset": "default",
                "source_identifier": "0001",
                "sequence": [["/tmp/a.png", "/tmp/a-mask.png"]],
            }
        ]
        val = [
            {
                "dataset": "dataset-b",
                "subdataset": "default",
                "source_identifier": "0001",
                "sequence": [["/tmp/b.png", "/tmp/b-mask.png"]],
            }
        ]
        assert_disjoint_sources(train, val)

    def test_source_disjointness_accepts_distinct_images(self):
        """
        Verify distinct source images pass the train/validation isolation check.

        Args:
            None.

        Returns:
            None.
        """
        train = [{"sequence": [["/tmp/train.png", "/tmp/train-mask.png"]]}]
        val = [{"sequence": [["/tmp/val.png", "/tmp/val-mask.png"]]}]
        assert_disjoint_sources(train, val)

    def test_checkpoint_score_uses_fixed_threshold_and_balances_dimensions(self):
        """
        Verify checkpoint selection averages fixed-threshold Dice equally across dimensions.

        Args:
            None.

        Returns:
            None.
        """
        metrics = {
            "dice2d@0.5": 0.8,
            "dice3d@0.5": 0.4,
            "dice_best": 0.99,
            "iou_best": 0.98,
        }
        self.assertAlmostEqual(checkpoint_selection_score(metrics), 0.6)

    def test_checkpoint_score_handles_a_missing_dimension(self):
        """
        Verify checkpoint selection uses the available dimension when the other is absent.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(checkpoint_selection_score({"dice2d@0.5": 0.7}), 0.7)
        with self.assertRaisesRegex(ValueError, "fixed-threshold"):
            checkpoint_selection_score({"dice_best": 0.99})


if __name__ == "__main__":
    unittest.main()
