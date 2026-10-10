"""Temporal stacks use spatial frame metrics, never volumetric distances."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rwkv_medsam2.evaluation.accuracy.tests.test_overlap import (
    fixture,
    prediction_result,
)
from rwkv_medsam2.evaluation.accuracy.surfaces import (
    analyze,
    boundary_metrics,
    stored_grid,
)
from rwkv_medsam2.evaluation.shared.data import (
    atomic_json,
    case_kind,
    condition_case,
    digest,
    file_hash,
    sequence_plan,
    validate_case,
    validate_result,
)


class VideoAccuracyTests(unittest.TestCase):
    """
    Test video provenance, frame-level surfaces, spatial units, and reporting.
    """

    def test_explicit_video_provenance_and_forward_order_required(self):
        """
        Verify temporal sequences require reviewed video provenance and forward order.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            case = fixture(Path(tmp), video=True)
            validate_case(case)
            self.assertEqual(case_kind(case), "video")

            for change in (
                {"anatomical_volume": True},
                {"reverse_slices": True},
                {"temporal_axis_note": ""},
                {"dim": 2},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_case(dict(case, **change))

            full = condition_case(case, "full_scan")
            ids, _, order = sequence_plan(full)
            self.assertEqual(ids, list(range(6)))
            self.assertEqual(
                order,
                sorted(full["support_frames"])
                + [t for t in ids if t not in full["support_frames"]],
            )

    def test_both_conditions_compute_only_2d_query_frame_distances(self):
        """
        Verify both video conditions measure boundaries on 2D query frames.

        Args:
            None.

        Returns:
            None.
        """
        for condition in ("positive_frames", "full_scan"):
            with self.subTest(
                condition=condition
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case = condition_case(fixture(root, video=True), condition)
                row, path = prediction_result(root, case)
                validate_result(row, case, "box", "test-run")
                shapes = []

                def checked(pred, truth, spacing):
                    """
                    Assert that video boundary distances use only two spatial dimensions.

                    Args:
                        pred (np.ndarray): Binary prediction mask.
                        truth (np.ndarray): Binary reference mask used to check boundary metrics.
                        spacing (Sequence[float]): Spatial pixel or voxel spacing in array-axis order.

                    Returns:
                        dict: Boundary metrics computed after the dimensionality checks.
                    """
                    self.assertEqual(pred.ndim, 2)
                    self.assertEqual(len(spacing), 2)
                    shapes.append(pred.shape)

                    return boundary_metrics(pred, truth, spacing)

                with patch(
                    "rwkv_medsam2.evaluation.accuracy.surfaces.boundary_metrics",
                    side_effect=checked,
                ):
                    result = analyze(row, path, case, root)

                query = [
                    t for t in case["frame_ids"] if t not in case["support_frames"]
                ]
                self.assertEqual(len(shapes), len(query))
                self.assertEqual([r["frame"] for r in result["per_slice"]], query)
                self.assertEqual(result["scope"], "mean_query_video_frame_surfaces")
                self.assertEqual(result["units"], "stored_pixels")
                self.assertEqual(result["spacing"], [1.0, 1.0])
                wrong = dict(row, case=dict(row["case"], sequence_kind=None))

                with self.assertRaisesRegex(ValueError, "sequence kind"):
                    validate_result(wrong, case, "box", "test-run")

    def test_video_spacing_never_uses_time_spacing(self):
        """
        Verify temporal spacing never enters video boundary-distance calculations.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            case = fixture(Path(tmp), video=True)
            physical = dict(case, spacing_verified_mm=True)
            with self.assertRaisesRegex(ValueError, "spacing_hw_mm"):
                stored_grid(physical, case["transform"])

            physical["spacing_hw_mm"] = [1.2, 0.7]
            transform = dict(
                case["transform"], geometry={"spacing_xyz": [999.0, 888.0, 777.0]}
            )
            self.assertEqual(stored_grid(physical, transform), ([1.2, 0.7], "mm"))

    def test_summary_csv_and_tensorboard_identify_video(self):
        """
        Verify JSON, CSV, and TensorBoard reports identify video results.

        Args:
            None.

        Returns:
            None.
        """
        from rwkv_medsam2.evaluation.accuracy.report import summarize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = condition_case(fixture(root, video=True), "full_scan")
            case["modes"] = ["box"]
            manifest = root / "frozen"
            manifest.mkdir()
            atomic_json(manifest / "manifest.json", {"fixture": True})
            identity = dict(
                model="fixture",
                manifest_sha256=file_hash(manifest / "manifest.json"),
                source_sha256="test",
                environment={},
            )
            shard = root / "runs/fixture/shard-0000-of-0001"
            row, path = prediction_result(shard, case, run_id=digest(identity))
            row["diagnostics"] = dict(
                query=dict(
                    precision=1.0,
                    recall=1.0,
                    specificity=1.0,
                    relative_size_error=0.0,
                    target_pixels=1,
                )
            )
            atomic_json(path, row)
            atomic_json(
                shard / "job.json",
                dict(
                    identity=identity,
                    run_id=digest(identity),
                    shards=1,
                    shard=0,
                    expected=[row["unit_key"]],
                ),
            )
            meta = dict(
                profile="test",
                sequence_condition="full_scan",
                scope="test",
                metric_grid="test",
            )
            args = SimpleNamespace(
                manifest=manifest,
                runs=root / "runs",
                output=root / "report",
                models=["fixture"],
                bootstrap=2,
            )

            with patch(
                "rwkv_medsam2.evaluation.accuracy.runner.read_frozen",
                return_value=(meta, [case]),
            ):
                summarize(args)

            result = json.loads((args.output / "summary.json").read_text())
            self.assertIn("full_scan/video/box", result["models"]["fixture"])
            self.assertIn(",video,", (args.output / "dataset-scores.csv").read_text())
            events = EventAccumulator(str(args.output / "tensorboard")).Reload()
            self.assertTrue(
                any("/video/box/dice" in tag for tag in events.Tags()["scalars"])
            )
            self.assertFalse(any("/3d/" in tag for tag in events.Tags()["scalars"]))
