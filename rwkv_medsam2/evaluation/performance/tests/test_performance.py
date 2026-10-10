"""CPU-only regression tests, written but not executed during implementation."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rwkv_medsam2.evaluation.performance.data import local_case, make_plan, read_plan
from rwkv_medsam2.evaluation.shared.prediction import predict_frame


class PerformancePlanTests(unittest.TestCase):
    """
    Test portable performance panels, balanced trial schedules, and plan integrity.
    """

    def test_portable_paths_reject_escape_and_other_roots(self):
        """
        Verify portable input paths cannot escape the declared dataset root.

        Args:
            None.

        Returns:
            None.
        """
        c = dict(image="/data/images/a.nii.gz", mask="masks/a.nii.gz")
        moved = local_case(c, Path("/gamma-data"), Path("/data"))
        self.assertEqual(moved["image"], "/gamma-data/images/a.nii.gz")
        self.assertEqual(moved["mask"], "/gamma-data/masks/a.nii.gz")
        self.assertEqual(c["image"], "/data/images/a.nii.gz")

        for image in ("../outside", "/elsewhere/image"):
            with self.assertRaises(ValueError):
                local_case(dict(c, image=image), Path("/gamma-data"), Path("/data"))

    def test_source_dedup_tied_lengths_balanced_trials_and_tamper_rejection(self):
        """
        Verify source deduplication, length strata, balanced trials, and plan integrity.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text("{}")
            cases = []

            for i in range(14):
                image = f"image-{i}"
                mask = f"mask-{i}"
                (root / image).touch()
                (root / mask).touch()
                cases.append(
                    dict(
                        unit_id=str(i),
                        case_id=str(i),
                        target_id="organ",
                        dataset="dataset",
                        task="task",
                        source_id=str(i // 2),
                        source_identity_status="verified",
                        eligible=True,
                        dim=3,
                        image=image,
                        mask=mask,
                        frame_ids=[0, 2, 4],
                        positive_frame_ids=[0, 2, 4],
                        support_frames=[2],
                        transform={"frames": 6},
                        modes=["box", "mask"],
                        sequence_condition="positive_frames",
                    )
                )

            meta = dict(
                protocol_sha256="protocol",
                sequence_condition="positive_frames",
                preparation_environment={"packages": {}},
            )
            args = SimpleNamespace(
                manifest=root,
                data_root=root,
                source_root=Path("/data"),
                models=["rwkv_medsam2_final_nodistill", "rwkv_medsam2_final_distill"],
                host_cache_gib=3.0,
                output=root / "plan.json",
            )
            with patch(
                "rwkv_medsam2.evaluation.accuracy.runner.read_frozen",
                return_value=(meta, cases),
            ), patch(
                "rwkv_medsam2.evaluation.accuracy.runner.engine_files", return_value={}
            ):
                make_plan(args)

            plan = read_plan(args.output)
            main = [
                c for c in plan["cases"] if plan["panel_labels"][c["unit_id"]]["main"]
            ]
            self.assertEqual(len(main), 5)
            self.assertEqual(len({c["source_id"] for c in main}), 5)
            self.assertEqual(
                {plan["panel_labels"][c["unit_id"]]["length_bin"] for c in main},
                {"short"},
            )
            self.assertEqual(sum(v["stress"] for v in plan["panel_labels"].values()), 1)

            for c in plan["cases"]:
                for mode in c["modes"]:
                    jobs = [
                        t
                        for t in plan["trials"]
                        if t["unit_id"] == c["unit_id"] and t["mode"] == mode
                    ]
                    blocks = [
                        [
                            t["model"]
                            for t in jobs
                            if t["kind"] == "prediction" and t["block"] == b
                        ]
                        for b in (0, 1)
                    ]
                    self.assertEqual(blocks[0], blocks[1][::-1])

                    for model in args.models:
                        self.assertEqual(
                            sum(
                                t["repetitions"]
                                for t in jobs
                                if t["model"] == model and t["kind"] == "prediction"
                            ),
                            10,
                        )
                        self.assertEqual(
                            sum(
                                t["kind"] == "memory"
                                for t in jobs
                                if t["model"] == model
                            ),
                            3,
                        )

            plan["cases"][0]["support_frames"] = [0]
            args.output.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                read_plan(args.output)


class SharedCoreTests(unittest.TestCase):
    """
    Test shared prediction indices and temporal-memory arguments.
    """

    def test_dense_frame_index_and_memory_pass_unchanged(self):
        # Sparse stored slice IDs must not accidentally become SAM frame indices.
        """
        Verify the shared prediction core retains dense indices and temporal-memory arguments.

        Args:
            None.

        Returns:
            None.
        """
        calls = {}

        class Model:
            """
            Record shared prediction calls without constructing a segmentation model.
            """

            def forward_image(self, frame):
                """
                Record the image tensor passed through the shared prediction core.

                Args:
                    frame (torch.Tensor): Normalized image frame passed to the encoder.

                Returns:
                    str: Synthetic encoded-feature marker.
                """
                calls["frame"] = frame

                return "encoded"

            def _prepare_backbone_features(self, encoded):
                """
                Return synthetic features with the shared core's expected tuple layout.

                Args:
                    encoded (object): Synthetic backbone result passed to the feature-preparation stub.

                Returns:
                    tuple: Placeholder backbone output, features, positions, and feature sizes.
                """
                return None, "features", "positions", "sizes"

            def track_step(self, **kwargs):
                """
                Record tracking arguments and return a synthetic mask result.

                Args:
                    **kwargs (dict): Tracking arguments recorded for dense-index and temporal-memory
                        assertions.

                Returns:
                    dict: Synthetic tracking output.
                """
                calls.update(kwargs)

                return {"result": "mask"}

        memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        points = {"point_coords": "frozen"}
        out = predict_frame(Model(), "image", points, None, memory, 1, True, 3)
        self.assertEqual(out, {"result": "mask"})
        self.assertIs(calls["output_dict"], memory)
        self.assertIs(calls["point_inputs"], points)
        self.assertEqual(calls["frame_idx"], 1)
        self.assertEqual(calls["num_frames"], 3)
        self.assertFalse(calls["track_in_reverse"])
        self.assertTrue(calls["run_mem_encoder"])


if __name__ == "__main__":
    unittest.main()
