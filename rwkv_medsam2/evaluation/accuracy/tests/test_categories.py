"""
Test audited reporting categories, shared-source bootstraps, and frozen-input cache parity.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from rwkv_medsam2.evaluation.accuracy.cohorts import (
    POLICY,
    report_views,
    validate_reporting,
)
from rwkv_medsam2.evaluation.accuracy.categorized import build_reports
from rwkv_medsam2.evaluation.accuracy.inputs import InputCache
from rwkv_medsam2.evaluation.accuracy.report import paired_interval
from rwkv_medsam2.evaluation.shared.data import freeze_case
from rwkv_medsam2.evaluation.accuracy.tests.test_overlap import fixture


def reporting(category="none_identified", validation=False, reserved=False):
    """
    Build reporting metadata for training, validation, and reserved-pool overlap.

    Args:
        category (str): Training-overlap category assigned to the reporting fixture.
        validation (bool): Whether the fixture reports validation overlap.
        reserved (bool): Whether the fixture reports reserved-pool overlap.

    Returns:
        dict: Audited reporting fixture with group, reasons, and exposure flags.
    """
    return dict(
        policy=POLICY,
        group=(
            "main"
            if category == "none_identified" and not validation
            else "supplementary"
        ),
        training_overlap=category,
        validation_overlap=validation,
        reserved_pool_only=reserved,
        source_history_uncertain=True,
        altered_scan_copy=False,
        audit_sha256="audit",
        limitations="Unknown external pretraining",
        reasons=[] if category == "none_identified" and not validation else ["match"],
    )


class CategoryTests(unittest.TestCase):
    """
    Test overlap-based report categories and shared-source bootstrap weighting.
    """

    def test_unused_pool_is_flagged_but_not_called_training_overlap(self):
        """
        Verify reporting treatment of reserved data that was never used for training.

        Args:
            None.

        Returns:
            None.
        """
        c = dict(reporting=reporting(reserved=True))
        self.assertIn("main", report_views(c))
        self.assertIn("reserved_pool_only", report_views(c))
        self.assertNotIn("supplementary", report_views(c))
        c["reporting"]["group"] = "supplementary"

        with self.assertRaises(ValueError):
            validate_reporting(c)

    def test_validation_overlap_excludes_main_even_without_training_match(self):
        """
        Verify that validation overlap places a case in supplementary reporting.

        Args:
            None.

        Returns:
            None.
        """
        c = dict(reporting=reporting(validation=True))
        self.assertIn("supplementary", report_views(c))
        self.assertIn("validation_overlap", report_views(c))
        self.assertNotIn("main", report_views(c))

    def test_reports_separate_categories_and_PAIP_without_new_predictions(self):
        """
        Verify categorized and PAIP-filtered reports reuse existing predictions.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, rows, paths = [], {}, {}
            for i, (ds, cat, score) in enumerate(
                [
                    ("A", "none_identified", 0.5),
                    ("A", "exact_image_or_scan", 1.0),
                    ("PAIP2019", "none_identified", 0.9),
                ]
            ):
                c = dict(
                    dataset=ds,
                    task="organ",
                    source_id=str(i),
                    case_id=str(i),
                    target_id=str(i),
                    dim=2,
                    modality="histology",
                    reporting=reporting(cat),
                    eligible=True,
                    modes=["box"],
                )
                row = dict(
                    case=c,
                    mode="box",
                    unit_key=str(i),
                    primary=dict(dice=score, iou=score),
                    all_frame_assisted=dict(dice=score, iou=score),
                )
                cases.append(c)
                rows[str(i)] = row
                paths[("fixture", str(i))] = root / "cases" / f"{i}.json"

            summary = build_reports({"fixture": rows}, cases, paths, Mock(), root, 10)
            views = summary["views"]
            self.assertEqual(
                views["main/without_PAIP"]["models"]["fixture"]["2d/box"]["dice"][
                    "macro"
                ],
                0.5,
            )
            self.assertEqual(
                views["main/with_PAIP"]["models"]["fixture"]["2d/box"]["dice"]["macro"],
                0.7,
            )
            self.assertEqual(
                views["training_exact_image_or_scan/with_PAIP"]["models"]["fixture"][
                    "2d/box"
                ]["dice"]["macro"],
                1.0,
            )
            self.assertEqual(len(rows), 3)
            self.assertTrue((root / "categorized-scores.csv").exists())

    def test_cross_dataset_duplicates_share_bootstrap_draws(self):
        """
        Verify shared bootstrap draws for source aliases across datasets.

        Args:
            None.

        Returns:
            None.
        """
        left = []
        right = []

        for ds in ("A", "B"):
            for i in range(4):
                c = dict(
                    dataset=ds,
                    task="organ",
                    source_id=str(i),
                    global_source_id=f"global:{i}",
                    case_id=str(i),
                )
                key = ds + str(i)
                value = i / 10 if ds == "A" else -i / 10
                left.append(dict(case=c, unit_key=key, primary={"dice": value}))
                right.append(dict(case=c, unit_key=key, primary={"dice": 0.0}))

        result = paired_interval(left, right, "dice", 100)
        self.assertEqual(result["source_clusters"], 4)
        self.assertEqual(result["ci95"], [0.0, 0.0])

    def test_freeze_cache_is_bitwise_equivalent_with_reporting_metadata(self):
        """
        Verify cached and uncached case freezing produce identical audited records.

        Args:
            None.

        Returns:
            None.
        """
        for dim in (2, 3):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                frozen = fixture(root, dim)
                keys = (
                    "case_id",
                    "target_id",
                    "dataset",
                    "task",
                    "source_id",
                    "source_identity_status",
                    "source_note",
                    "modality",
                    "image",
                    "mask",
                    "dim",
                    "label",
                    "target_scope",
                    "split",
                    "provenance_reviewed",
                    "preprocessing_note",
                    "axis_zyx",
                    "reverse_slices",
                    "anatomical_volume",
                    "geometry_note",
                    "spacing_verified_mm",
                    "spacing_provenance_note",
                )
                c = {k: frozen[k] for k in keys if k in frozen}
                c["reporting"] = reporting()
                expected = freeze_case(c, root, "test")
                cache = dict(inputs=InputCache(1024**2), hashes={}, encoder_hashes={})
                self.assertEqual(
                    expected, freeze_case(c, root, "test", freeze_cache=cache)
                )
                self.assertEqual(
                    expected, freeze_case(c, root, "test", freeze_cache=cache)
                )
                self.assertEqual(expected["reporting"], c["reporting"])


if __name__ == "__main__":
    unittest.main()
