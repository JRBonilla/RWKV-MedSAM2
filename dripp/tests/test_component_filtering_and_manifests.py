import json
import logging
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

import dripp.dataset as dataset_module
from dripp.dataset import SegmentationDataset, atomic_write_json
from dripp.preprocessor import Preprocessor


class ComponentVoxelFilterTests(unittest.TestCase):
    def _run_volume(self, voxel_count, threshold):
        with tempfile.TemporaryDirectory() as root:
            image = np.ones((1, 8, 8), dtype=np.float32)
            mask = np.zeros((1, 8, 8), dtype=np.uint8)
            mask.reshape(-1)[:voxel_count] = 1
            image_path = os.path.join(root, "image.nii.gz")
            mask_path = os.path.join(root, "mask.nii.gz")
            image_dir = os.path.join(root, "images")
            mask_dir = os.path.join(root, "masks")
            os.makedirs(image_dir)
            os.makedirs(mask_dir)
            sitk.WriteImage(sitk.GetImageFromArray(image), image_path)
            sitk.WriteImage(sitk.GetImageFromArray(mask), mask_path)

            preprocessor = Preprocessor(
                target_size=(8, 8), min_mask_size=1,
                min_component_voxels_3d=threshold, dataset_name="test",
            )
            return preprocessor.preprocess_group(
                "default", "3D", [image_path], [mask_path], "mri",
                image_dir, mask_dir, f"case_{voxel_count}_{threshold}", {},
                {
                    "min_slice_area_px_3d": 0,
                    "min_slice_area_fraction_3d": 0.0,
                    "min_qualified_slices_3d": 0,
                    "require_contiguous_qualified_slices_3d": False,
                },
            )

    def test_63_voxels_rejected_and_64_voxels_accepted(self):
        rejected = self._run_volume(63, 64)
        accepted = self._run_volume(64, 64)
        self.assertEqual(rejected["proc_masks"], [])
        self.assertEqual(rejected["qc"]["components_3d_rejected"], 1)
        self.assertEqual(len(accepted["proc_masks"]), 1)
        self.assertEqual(accepted["proc_masks"][0]["dimension"], 3)
        self.assertEqual(accepted["proc_masks"][0]["image_index"], 0)

    def test_single_slice_64_voxel_component_is_accepted(self):
        result = self._run_volume(64, 64)
        self.assertEqual(len(result["proc_masks"]), 1)
        self.assertEqual(result["qc"]["components_3d_found"], 1)

    def test_zero_disables_filter(self):
        result = self._run_volume(63, 0)
        self.assertEqual(len(result["proc_masks"]), 1)
        self.assertEqual(result["qc"]["components_3d_rejected"], 0)

    def test_negative_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero or greater"):
            Preprocessor(min_component_voxels_3d=-1)

    def test_2d_and_video_pairings_are_recorded_at_save_time(self):
        preprocessor = Preprocessor(target_size=(2, 2), min_mask_size=1)
        preprocessor._reset_group_tracking()
        preprocessor._track_image_output("image.png", "2d", 7)
        preprocessor._track_mask_output("mask.png", "2d", 7, "organ")
        preprocessor._track_image_output("frame.png", "video", 9)
        preprocessor._track_mask_output("frame_mask.png", "video", 9, "organ")
        result = preprocessor.get_group_tracking_metadata()
        self.assertEqual(result["proc_masks"][0]["image_index"], 0)
        self.assertEqual(result["proc_masks"][1]["image_index"], 1)
        self.assertEqual([item["dimension"] for item in result["proc_masks"]], [2, 2])

    def test_2d_filter_reuses_component_areas_for_qc(self):
        preprocessor = Preprocessor(target_size=(2, 2), min_mask_size=3)
        preprocessor._reset_group_tracking()
        kept = preprocessor._filter_small_components([
            np.ones((2, 2), dtype=np.uint8),
            np.ones((1, 2), dtype=np.uint8),
        ])
        tracking = preprocessor.get_group_tracking_metadata()
        self.assertEqual(len(kept), 1)
        self.assertEqual(tracking["qc"]["components_2d_found"], 2)
        self.assertEqual(tracking["qc"]["components_2d_rejected"], 1)
        self.assertIn("component_below_minimum_size_2d", tracking["reasons"])


class PairingRegressionTests(unittest.TestCase):
    def _run_2d_group(self, mask_path):
        preprocessor = Preprocessor(
            target_size=(4, 4), min_mask_size=1, dataset_name="fixture",
        )
        mask = np.ones((4, 4), dtype=np.uint8)
        preprocessor.is_significant_crop = False
        image = np.ones((4, 4), dtype=np.float32)
        with mock.patch.object(
            preprocessor, "load_mask", return_value=(mask, None, None),
        ), mock.patch.object(
            preprocessor, "load_image", return_value=(image, None),
        ), mock.patch.object(
            preprocessor, "_crop", side_effect=lambda value, _bbox: value,
        ), mock.patch.object(
            preprocessor, "_resize", side_effect=lambda value, is_mask: value,
        ), mock.patch.object(
            preprocessor, "_normalize_intensity",
            side_effect=lambda value, *_args, **_kwargs: value,
        ), mock.patch.object(
            preprocessor, "_save_mask",
        ) as save_mask, mock.patch.object(
            preprocessor, "_save_image",
        ), mock.patch(
            "dripp.preprocessor.match_mask_class", return_value="organ",
        ):
            preprocessor.preprocess_2d(
                "site", ["/tmp/case_a.png", "/tmp/case_b.png"],
                [mask_path], "default", "/tmp", "/tmp", "group",
                (0, 4, 0, 4), {}, {
                    "mask_stem_strategy": "stem",
                    "save_2d_masks_with_source_stem": False,
                    "skip_unmatched_2d_images": False,
                },
            )
        return preprocessor, save_mask

    def test_stem_pairing_is_recorded_when_source_stem_filename_is_disabled(self):
        _preprocessor, save_mask = self._run_2d_group("/tmp/case_b.png")
        self.assertEqual(save_mask.call_count, 1)
        self.assertEqual(save_mask.call_args.kwargs["pairing_index"], 1)

    def test_unmatched_multi_image_mask_is_not_assigned_to_image_zero(self):
        preprocessor, save_mask = self._run_2d_group("/tmp/no_match.png")
        self.assertEqual(save_mask.call_count, 0)
        tracking = preprocessor.get_group_tracking_metadata()
        self.assertEqual(tracking["qc"]["unmatched_2d_masks"], 1)
        self.assertIn("unmatched_2d_mask", tracking["reasons"])



class FakePreprocessor:
    target_size = (8, 8)
    ct_profiles = None
    min_mask_size = 100
    min_component_voxels_3d = 64
    dataset_name = "fixture"
    background_value = 0

    def __init__(self):
        self.dataset_logger = logging.getLogger("fake-preprocessor")
        self.main_logger = self.dataset_logger
        self._tracking = {}

    @staticmethod
    def _short_id(value):
        return value[:8]

    @staticmethod
    def _qc(**updates):
        value = {
            "empty_masks": 0,
            "components_2d_found": 1,
            "components_2d_rejected": 0,
            "components_3d_found": 0,
            "components_3d_rejected": 0,
        }
        value.update(updates)
        return value

    @staticmethod
    def _policy():
        return {
            "min_component_voxels_3d": 64,
            "mask_qc_downsample_3d": 2,
            "min_slice_area_px_3d": 32,
            "min_slice_area_fraction_3d": 0.00153,
            "min_qualified_slices_3d": 8,
            "require_contiguous_qualified_slices_3d": True,
            "slice_axis_sampling_3d": "full_resolution",
            "applied": True,
        }

    def preprocess_group(self, sub_name, pipeline, images, masks, modality,
                         image_dir, mask_dir, composite_id, classes, options):
        image_path = os.path.join(image_dir, "out_img000.png")
        mask_record = {
            "path": os.path.join(mask_dir, "out_img000_mask000.png"),
            "class": "organ", "image_index": 0, "dimension": 2,
        }
        if "failed" in composite_id:
            self._tracking = {
                "proc_images": [], "proc_masks": [], "qc": self._qc(),
                "reasons": [], "stage": "processing",
            }
            raise RuntimeError("synthetic processing failure")
        if "partial" in composite_id:
            self._tracking = {
                "proc_images": [image_path], "proc_masks": [mask_record],
                "qc": self._qc(), "reasons": [], "stage": "write",
            }
            raise OSError("synthetic late write failure")
        if "empty" in composite_id:
            return {
                "proc_images": [image_path], "proc_masks": [],
                "qc": self._qc(empty_masks=1, components_2d_found=0),
                "reasons": ["empty_mask"], "stage": "processing",
            }
        if "small" in composite_id:
            return {
                "proc_images": [image_path], "proc_masks": [],
                "qc": self._qc(components_2d_rejected=1),
                "reasons": ["component_below_minimum_size_2d"],
                "stage": "processing",
            }
        if "multi" in composite_id:
            images_out = [
                os.path.join(image_dir, "out_img000_modality0.nii.gz"),
                os.path.join(image_dir, "out_img001_modality1.nii.gz"),
            ]
            return {
                "proc_images": images_out,
                "proc_masks": [dict(mask_record, dimension=3)],
                "image_niftis": images_out,
                "mask_niftis": [mask_record["path"]],
                "qc": self._qc(components_2d_found=0, components_3d_found=1),
                "quality_policy_3d": self._policy(),
                "reasons": [], "stage": "write", "volume_shape": [1, 8, 8],
            }
        return {
            "proc_images": [image_path], "proc_masks": [mask_record],
            "qc": self._qc(), "reasons": [], "stage": "write",
            "quality_policy_3d": self._policy(),
            "resize_shape": [8, 8],
        }

    def get_group_tracking_metadata(self):
        return self._tracking


class ManifestResultTests(unittest.TestCase):
    def test_nested_manifest_statuses_metadata_pairings_and_legacy_filter(self):
        with tempfile.TemporaryDirectory() as root:
            groups_dir = os.path.join(root, "groups")
            output_root = os.path.join(root, "output")
            os.makedirs(groups_dir)
            groups = {
                "dataset_metadata": {"license": "fixture"},
                "subdatasets": [{
                    "name": "site-a", "modality": "mri", "pipeline": "2D",
                    "tasks": ["segment"], "classes": ["organ"],
                    "additional_metadata": {"scanner": "fixture"},
                    "train": [
                        {"identifier": name, "images": [{"path": f"/{name}.png"}],
                         "masks": [{"path": f"/{name}_mask.png"}],
                         "additional_metadata": {"source": name}}
                        for name in ("accepted", "empty", "small", "failed", "partial", "multi")
                    ],
                    "test": [],
                }],
            }
            with open(os.path.join(groups_dir, "fixture_groups.json"), "w") as handle:
                json.dump(groups, handle)

            with mock.patch.object(dataset_module, "GROUPS_DIR", groups_dir), \
                    mock.patch.object(dataset_module, "BASE_PROC", output_root), \
                    mock.patch.object(dataset_module.config, "BASE_PROC", output_root):
                dataset = SegmentationDataset("fixture", {"modalities": ["mri"]})
                dataset.process(FakePreprocessor())

            dataset_dir = os.path.join(output_root, "fixture")
            with open(os.path.join(dataset_dir, "fixture_groups.json")) as handle:
                nested = json.load(handle)
            with open(os.path.join(dataset_dir, "groupings.json")) as handle:
                legacy = json.load(handle)
            with open(os.path.join(dataset_dir, "preprocessing_report.json")) as handle:
                report = json.load(handle)

            self.assertEqual(nested["dataset_metadata"], groups["dataset_metadata"])
            self.assertEqual(nested["preprocessing_schema_version"], 3)
            subdataset = nested["subdatasets"][0]
            self.assertEqual(subdataset["tasks"], ["segment"])
            self.assertEqual(subdataset["classes"], ["organ"])
            self.assertEqual(subdataset["additional_metadata"]["scanner"], "fixture")
            entries = subdataset["train"]
            by_source = {}
            for entry in entries:
                by_source.setdefault(entry.get("source_identifier", entry["identifier"]), []).append(entry)
            self.assertEqual(by_source["accepted"][0]["preprocessing"]["status"], "accepted")
            self.assertEqual(by_source["empty"][0]["preprocessing"]["reason"], "empty_mask")
            self.assertEqual(by_source["small"][0]["preprocessing"]["status"], "rejected")
            self.assertEqual(by_source["failed"][0]["preprocessing"]["status"], "failed")
            self.assertEqual(by_source["partial"][0]["preprocessing"]["status"], "partial")
            self.assertEqual(by_source["failed"][0]["proc_images"], [])
            self.assertEqual(by_source["small"][0]["proc_masks"], [])
            self.assertEqual(len(by_source["multi"]), 2)
            self.assertTrue(all(e["proc_masks"][0]["image_index"] == 0 for e in by_source["multi"]))
            self.assertTrue(all(e["proc_masks"][0]["dimension"] == 3 for e in by_source["multi"]))
            self.assertTrue(all(
                e["preprocessing"]["quality_policy_3d"]["applied"]
                for e in by_source["multi"]
            ))
            self.assertTrue(all(e["preprocessing"]["status"] in {"accepted", "partial"} for e in legacy))
            self.assertEqual(report["group_counts"], {
                "attempted": 6, "accepted": 2, "rejected": 2,
                "failed": 1, "partial": 1,
            })
            self.assertEqual(len(report["issues"]), 4)
            self.assertEqual(report["schema_version"], 3)
            self.assertIn(FakePreprocessor._policy(), report["quality_policies_3d"])

    def test_shared_2d_masks_are_copied_to_every_split_modality(self):
        dataset = object.__new__(SegmentationDataset)
        dataset.preprocessor = FakePreprocessor()
        entry = {
            "identifier": "qubiq",
            "images": [{"path": "/raw/modalities.nii.gz"}],
            "masks": [{"path": "/raw/consensus.png"}],
        }
        metadata = {
            "proc_images": [f"/processed/modality{index}.png" for index in range(4)],
            "proc_masks": [
                {
                    "path": "/processed/consensus.png", "class": "tumor",
                    "image_index": 0, "dimension": 2,
                }
            ],
        }
        status = {
            "status": "accepted", "reason": None, "stage": None,
            "output_counts": {"images": 4, "masks": 1},
        }
        shared = dataset._successful_entries(
            entry, "qubiq", metadata, status, {
                "split_processed_images_by_modality": True,
                "share_masks_across_processed_images": True,
            },
        )
        self.assertEqual(len(shared), 4)
        self.assertTrue(all(len(item["proc_masks"]) == 1 for item in shared))
        self.assertTrue(all(
            item["proc_masks"][0]["image_index"] == 0 for item in shared
        ))

        separate = dataset._successful_entries(
            entry, "qubiq", metadata, status, {
                "split_processed_images_by_modality": True,
                "share_masks_across_processed_images": False,
            },
        )
        self.assertEqual(
            [len(item["proc_masks"]) for item in separate], [1, 0, 0, 0]
        )


    def test_atomic_write_preserves_previous_manifest_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "manifest.json")
            with open(path, "w") as handle:
                handle.write('{"old":true}')
            with mock.patch.object(dataset_module.json, "dump", side_effect=TypeError("boom")):
                with self.assertRaises(TypeError):
                    atomic_write_json(path, {"new": True})
            with open(path) as handle:
                self.assertEqual(handle.read(), '{"old":true}')
            self.assertEqual(os.listdir(root), ["manifest.json"])


if __name__ == "__main__":
    unittest.main()
