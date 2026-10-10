"""Metric/protocol edge cases. No GPU or checkpoint loading; not run during creation."""

import unittest
import numpy as np

from rwkv_medsam2.evaluation.shared.data import (
    add_counts,
    canonical_frame,
    canonical_mask,
    confusion,
    decode_mask,
    encode_mask,
    overlaps,
    support_indices,
    validate_case,
)
from rwkv_medsam2.evaluation.accuracy.report import (
    RunningMacro,
    hierarchical,
    source_means,
)


class AccuracyContracts(unittest.TestCase):
    """
    Test canonical preprocessing, support selection, overlap scores, and aggregation.
    """

    def test_empty_masks(self):
        """
        Verify perfect overlap for two empty masks and zero overlap for one empty mask.

        Args:
            None.

        Returns:
            None.
        """
        empty = np.zeros((2, 2), bool)
        self.assertEqual(overlaps(confusion(empty, empty)), {"dice": 1.0, "iou": 1.0})
        self.assertEqual(overlaps(confusion(~empty, empty)), {"dice": 0.0, "iou": 0.0})
        self.assertEqual(overlaps(confusion(empty, ~empty)), {"dice": 0.0, "iou": 0.0})

    def test_empty_query_false_positives_count(self):
        """
        Verify that false positives on empty query slices affect pooled overlap scores.

        Args:
            None.

        Returns:
            None.
        """
        positive = confusion(np.ones((2, 2), bool), np.ones((2, 2), bool))
        empty_slice = confusion(np.ones((2, 2), bool), np.zeros((2, 2), bool))
        scores = overlaps(add_counts([positive, empty_slice]))
        self.assertAlmostEqual(scores["dice"], 2 / 3)
        self.assertEqual(scores["iou"], 0.5)

    def test_support_never_consumes_all_positive_slices(self):
        """
        Verify that support selection leaves at least one positive query slice.

        Args:
            None.

        Returns:
            None.
        """
        for count in (2, 3, 4, 5, 20, 100):
            selected = support_indices(list(range(count)), "p", "case")
            self.assertGreaterEqual(len(selected), 1)
            self.assertLess(len(selected), count)
            self.assertEqual(selected, sorted(set(selected)))

        self.assertEqual(len(support_indices(list(range(20)), "p", "case")), 5)
        self.assertEqual(support_indices([0], "p", "case"), [])

    def test_support_not_affected_by_inventory_order(self):
        """
        Verify that support selection is independent of inventory ordering.

        Args:
            None.

        Returns:
            None.
        """
        forward = support_indices(list(range(50)), "p", "case")
        backward = support_indices(list(reversed(range(50))), "p", "case")
        self.assertEqual(forward, backward)

    def test_saved_mask_roundtrip(self):
        """
        Verify lossless serialization and restoration of a support mask.

        Args:
            None.

        Returns:
            None.
        """
        mask = np.zeros((512, 512), bool)
        mask[0, 511] = True
        mask[300:400, 25:31] = True
        np.testing.assert_array_equal(decode_mask(encode_mask(mask)), mask)

    def test_constant_volume_finite_and_normalized_once(self):
        """
        Verify finite canonical inputs for a constant-intensity volume.

        Args:
            None.

        Returns:
            None.
        """
        frame = canonical_frame(np.ones((9, 7), np.float32), [1.0, 1.0])
        self.assertEqual(frame.shape, (1, 3, 512, 512))
        expected = -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
        np.testing.assert_allclose(frame[0, :, 0, 0], expected, rtol=1e-6)
        self.assertTrue(np.isfinite(frame).all())

    def test_nearest_mask_keeps_binary(self):
        """
        Verify that nearest-neighbor mask resampling preserves binary labels.

        Args:
            None.

        Returns:
            None.
        """
        mask = canonical_mask(np.eye(8, dtype=bool))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (512, 512))

    def test_macro_does_not_overweight_more_targets_or_bigger_datasets(self):
        """
        Verify equal weighting across cases, sources, tasks, and datasets.

        Args:
            None.

        Returns:
            None.
        """
        rows = []
        for dataset, case, source, values in [
            ("A", "a", "s1", [0.0, 0.0, 0.0]),
            ("A", "b", "s2", [1.0]),
            ("B", "c", "s3", [1.0]),
        ]:
            for i, value in enumerate(values):
                rows.append(
                    {
                        "case": dict(
                            dataset=dataset,
                            task="organ",
                            source_id=source,
                            case_id=case,
                            target_id=str(i),
                        ),
                        "primary": {"dice": value},
                    }
                )

        aggregate = hierarchical(source_means(rows, "dice"))
        self.assertEqual(aggregate["macro"], 0.75)
        tracker = RunningMacro()

        for row in rows:
            value = tracker.update(row, "dice")

        self.assertEqual(value, 0.75)
        self.assertEqual(tracker.update(rows[-1], "dice"), 0.75)

    def test_unreviewed_inputs_refused(self):
        """
        Verify rejection of evaluation cases lacking reviewed provenance.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaises(ValueError):
            validate_case({"provenance_reviewed": False})


class RecordingContracts(unittest.TestCase):
    """
    Test pooled diagnostics, calibration records, and boundary conventions.
    """

    def test_confusion_generator_not_exhausted(self):
        """
        Verify that pooling consumes a confusion-count generator only once.

        Args:
            None.

        Returns:
            None.
        """
        rows = [
            {"tp": 1, "fp": 2, "fn": 3, "tn": 4},
            {"tp": 5, "fp": 6, "fn": 7, "tn": 8},
        ]
        self.assertEqual(
            add_counts(r for r in rows), {"tp": 6, "fp": 8, "fn": 10, "tn": 12}
        )

    def test_slice_generator_keeps_empty_slices(self):
        """
        Verify that slice summaries retain empty slices from generator inputs.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.accuracy.records import slice_summary

        result = slice_summary(
            r
            for r in [
                {"tp": 1, "fp": 0, "fn": 0, "tn": 3},
                {"tp": 0, "fp": 2, "fn": 0, "tn": 2},
            ]
        )
        self.assertEqual(result["empty_query_slices_with_false_positives"], 1)
        self.assertEqual(result["empty_query_false_positive_pixels"], 2)

    def test_zero_logit_calibration(self):
        """
        Verify calibration statistics for zero logits and balanced target labels.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.accuracy.records import (
            probability_record,
            pool_probability,
        )

        record = probability_record(np.zeros((2, 2)), np.eye(2, dtype=bool))
        pooled = pool_probability([record])
        self.assertAlmostEqual(pooled["brier"], 0.25)
        self.assertAlmostEqual(pooled["ece20"], 0.0)
        self.assertAlmostEqual(pooled["log_loss"], np.log(2))

    def test_surface_identity_and_empty_conventions(self):
        """
        Verify boundary metrics for identical masks and empty-mask cases.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.accuracy.surfaces import boundary_metrics

        target = np.zeros((8, 8), bool)
        target[2:5, 2:5] = True
        same = boundary_metrics(target, target, [1.0, 1.0])
        self.assertEqual(same["hd95"], 0.0)
        self.assertEqual(same["nsd"]["2.0"], 1.0)
        empty = boundary_metrics(np.zeros_like(target), target, [1.0, 1.0])
        self.assertIsNone(empty["hd95"])
        self.assertEqual(empty["nsd"]["2.0"], 0.0)


class ArchiveContracts(unittest.TestCase):
    """
    Test lossless prediction archives and identity validation.
    """

    def test_archive_roundtrip_and_identity(self):
        """
        Verify archived mask/logit round trips and frozen-identity validation.

        Args:
            None.

        Returns:
            None.
        """
        import tempfile
        import zipfile
        from pathlib import Path
        from rwkv_medsam2.evaluation.accuracy.records import (
            PredictionArchive,
            verify_archive,
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = dict(
                run_id="run",
                unit_key="case:box",
                case_sha256="casehash",
                sequence_condition="positive_frames",
                frame_ids=[7],
            )
            archive = PredictionArchive(
                root / "predictions/case.zip", identity, save_logits=True
            )
            logits = np.full((512, 512), -1.0, np.float32)
            logits[4:8, 5:9] = 1.0
            mask = logits >= 0
            archive.frame(7, mask, logits)
            record = dict(identity, frames=1, prediction_archive=archive.commit())
            path = verify_archive(root / "cases/case.json", record)

            with zipfile.ZipFile(path) as stream:
                recovered = np.unpackbits(
                    np.frombuffer(stream.read("masks/000007.bin"), np.uint8)
                ).reshape(512, 512)
                np.testing.assert_array_equal(recovered, mask)

            record["run_id"] = "different-run"
            with self.assertRaises(ValueError):
                verify_archive(root / "cases/case.json", record)


class SequenceConditionContracts(unittest.TestCase):
    """
    Test frozen prompts, frame membership, and dense sequence indices.
    """

    def common_case(self):
        """
        Build a frozen volume fixture with one support and two positive query slices.

        Args:
            None.

        Returns:
            dict: Eligible case with shared prompts and canonical-input identities.
        """
        return dict(
            dim=3,
            eligible=True,
            transform={"frames": 7},
            positive_frames=3,
            positive_frame_ids=[1, 3, 5],
            support_frames=[3],
            prompts={"3": {"box_xy": [[1, 1], [2, 2]], "mask_sha256": "mask"}},
            encoder_input_sha256=["input"] * 7,
            shared_case_sha256="common",
        )

    def test_positive_inputs_have_no_empty_frames_and_dense_memory_indices(self):
        """
        Verify positive-only frame membership and dense SAM memory indices.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.shared.data import condition_case, sequence_plan

        case = condition_case(self.common_case(), "positive_frames")
        ids, mapping, order = sequence_plan(case)
        self.assertEqual(ids, [1, 3, 5])
        self.assertEqual(mapping, {1: 0, 3: 1, 5: 2})
        self.assertEqual(order, [3, 1, 5])

    def test_robustness_preserves_prompts_and_common_case(self):
        """
        Verify that full-scan evaluation preserves frozen prompts and common-case identity.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.shared.data import condition_case, sequence_plan

        positive = condition_case(self.common_case(), "positive_frames")
        full = condition_case(positive, "full_scan")

        for key in (
            "prompts",
            "support_frames",
            "shared_case_sha256",
            "encoder_input_sha256",
        ):
            self.assertEqual(positive[key], full[key])

        self.assertEqual(sequence_plan(full)[0], list(range(7)))
        self.assertEqual(positive["frame_ids"], [1, 3, 5])

    def test_full_scan_cannot_be_relabelled_as_positive(self):
        """
        Verify rejection of a full-scan result relabelled as positive-only evaluation.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.shared.data import condition_case, sequence_plan

        case = condition_case(self.common_case(), "full_scan")
        case["sequence_condition"] = "positive_frames"

        with self.assertRaises(ValueError):
            sequence_plan(case)

    def test_two_dimensional_cases_are_not_repeated_in_robustness(self):
        """
        Verify that full-scan robustness excludes independent 2D cases.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.shared.data import condition_case

        case = dict(self.common_case(), dim=2)
        self.assertEqual(
            condition_case(case, "positive_frames")["sequence_condition"],
            "independent_2d",
        )
        self.assertFalse(condition_case(case, "full_scan")["eligible"])


if __name__ == "__main__":
    unittest.main()
