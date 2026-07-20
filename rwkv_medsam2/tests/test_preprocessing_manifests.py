import json
import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

from rwkv_medsam2.utils import preprocessing


class ExplicitPairingTests(unittest.TestCase):
    def test_explicit_pairing_dimension_status_skip_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_dir = os.path.join(root, "fixture")
            os.makedirs(dataset_dir)
            explicit_images = ["/tmp/first.bin", "/tmp/second.bin"]
            legacy_image = "/tmp/legacy_img007.png"
            manifest = {
                "subdatasets": [{
                    "name": "site", "modality": "mri", "tasks": ["segment"],
                    "classes": ["organ"],
                    "train": [
                        {
                            "identifier": "explicit", "split": "train",
                            "preprocessing": {"status": "accepted"},
                            "proc_images": explicit_images,
                            "proc_masks": [{
                                "path": "/tmp/explicit_mask.bin", "class": "organ",
                                "image_index": 1, "dimension": 2,
                            }],
                        },
                        {
                            "identifier": "legacy", "split": "train",
                            "proc_images": [legacy_image],
                            "proc_masks": [{
                                "path": "/tmp/legacy_img007_mask000_%organ%_comp000.png",
                                "class": "organ",
                            }],
                        },
                        {
                            "identifier": "malformed", "split": "train",
                            "preprocessing": {"status": "accepted"},
                            "proc_images": ["/tmp/only.png"],
                            "proc_masks": [{
                                "path": "/tmp/bad.png", "class": "organ",
                                "image_index": 4, "dimension": 2,
                            }],
                        },
                        {
                            "identifier": "rejected", "split": "train",
                            "preprocessing": {"status": "rejected"},
                            "proc_images": ["/tmp/rejected.png"],
                            "proc_masks": [{
                                "path": "/tmp/rejected_mask.png", "class": "organ",
                                "image_index": 0, "dimension": 2,
                            }],
                        },
                    ],
                    "test": [],
                }],
            }
            with open(os.path.join(dataset_dir, "fixture_groups.json"), "w") as handle:
                json.dump(manifest, handle)

            with mock.patch.object(preprocessing, "_mask_has_fg_2d", return_value=True):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pairs = preprocessing.get_pairings(root, ["fixture"], split="train")

            self.assertEqual(len(pairs), 2)
            by_image = {pair["pair"][0]: pair for pair in pairs}
            self.assertEqual(by_image["/tmp/second.bin"]["dim"], 2)
            self.assertEqual(by_image[legacy_image]["dim"], 2)
            self.assertNotIn("/tmp/rejected.png", by_image)
            self.assertTrue(any("image_index=4" in str(item.message) for item in caught))


class DrippPolicyTrustTests(unittest.TestCase):
    QUALITY_PARAMS = {
        "min_voxels": 64,
        "min_slices_any_axis": 8,
        "min_slice_area_px": 32,
        "downsample": 2,
        "min_frac_2d_area": 0.00153,
    }

    @staticmethod
    def _policy(**updates):
        policy = {
            "min_component_voxels_3d": 64,
            "mask_qc_downsample_3d": 2,
            "min_slice_area_px_3d": 32,
            "min_slice_area_fraction_3d": 0.00153,
            "min_qualified_slices_3d": 8,
            "require_contiguous_qualified_slices_3d": True,
            "applied": True,
        }
        policy.update(updates)
        return policy

    @staticmethod
    def _write_manifest(root, entries):
        dataset_dir = os.path.join(root, "fixture")
        os.makedirs(dataset_dir)
        manifest = {
            "preprocessing_schema_version": 2,
            "subdatasets": [{
                "name": "site", "modality": "mri", "tasks": ["segment"],
                "classes": ["organ"], "train": entries, "test": [],
            }],
        }
        with open(
            os.path.join(dataset_dir, "fixture_groups.json"), "w"
        ) as handle:
            json.dump(manifest, handle)

    @classmethod
    def _entry(cls, identifier, policy):
        return {
            "identifier": identifier,
            "split": "train",
            "preprocessing": {
                "status": "accepted", "quality_policy_3d": policy,
            },
            "proc_images": [f"/tmp/{identifier}.nii.gz"],
            "proc_masks": [{
                "path": f"/tmp/{identifier}_mask.nii.gz",
                "class": "organ", "image_index": 0, "dimension": 3,
            }],
        }

    def test_sufficient_policy_skips_all_mask_stat_io(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_manifest(
                root, [self._entry("trusted", self._policy())]
            )
            with mock.patch.object(
                preprocessing, "load_maskstats_cache",
                side_effect=AssertionError("trusted masks must not be read"),
            ), mock.patch.object(
                preprocessing, "compute_mask_stats_3d",
                side_effect=AssertionError("trusted masks must not be scanned"),
            ):
                pairs = preprocessing.get_pairings(
                    root, ["fixture"], split="train",
                    quality_params=self.QUALITY_PARAMS,
                )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["pair"][1], "/tmp/trusted_mask.nii.gz")

    def test_legacy_malformed_and_weaker_policies_use_fallback(self):
        policies = [
            {},
            self._policy(applied=False),
            self._policy(min_component_voxels_3d=63),
            self._policy(mask_qc_downsample_3d=1),
            self._policy(min_slice_area_px_3d=31),
            self._policy(min_slice_area_fraction_3d=0.00152),
            self._policy(min_qualified_slices_3d=7),
            self._policy(require_contiguous_qualified_slices_3d=False),
            self._policy(mask_qc_downsample_3d="2"),
        ]
        entries = [
            self._entry(f"fallback_{index}", policy)
            for index, policy in enumerate(policies)
        ]
        stats = {
            entry["proc_masks"][0]["path"]: {"fixture": True}
            for entry in entries
        }
        with tempfile.TemporaryDirectory() as root:
            self._write_manifest(root, entries)
            with mock.patch.object(
                preprocessing, "load_maskstats_cache", return_value=stats,
            ) as load_stats, mock.patch.object(
                preprocessing, "evaluate_quality_from_stats",
                return_value=(True, {"reason": "ok"}),
            ) as evaluate:
                pairs = preprocessing.get_pairings(
                    root, ["fixture"], split="train",
                    quality_params=self.QUALITY_PARAMS,
                )
        self.assertEqual(len(pairs), len(policies))
        self.assertEqual(load_stats.call_count, len(policies))
        self.assertEqual(evaluate.call_count, len(policies))


class CacheSignatureTests(unittest.TestCase):
    def test_manifest_fingerprint_changes_after_manifest_metadata_changes(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_dir = os.path.join(root, "fixture")
            os.makedirs(dataset_dir)
            path = os.path.join(dataset_dir, "fixture_groups.json")
            with open(path, "w") as handle:
                handle.write('{"value":1}')
            first = preprocessing._fingerprint_dripp_manifests(root)
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            second = preprocessing._fingerprint_dripp_manifests(root)
            self.assertNotEqual(first, second)

    def test_foreground_fraction_changes_cache_signature(self):
        values = {
            "version": preprocessing.DATASET_CACHE_VERSION,
            "out_dir": "/tmp/out", "split": "both", "seq_len": 8,
            "min_fg_frames_in_window": 2, "truncate_val_test": False,
            "val_frac": 0.1, "seed": 42, "aug_probs": (0.15, 0.25, 0.5),
            "quality_params": {}, "tasks_file_fingerprint": None,
            "dripp_manifests_fingerprint": "manifest",
        }
        first = preprocessing.DatasetSignature(fg_min_pixels_frac=0.1, **values)
        second = preprocessing.DatasetSignature(fg_min_pixels_frac=0.2, **values)
        self.assertNotEqual(first.to_hash(), second.to_hash())

    def test_cache_version_change_invalidates_signature(self):
        values = {
            "out_dir": "/tmp/out", "split": "both", "seq_len": 8,
            "min_fg_frames_in_window": 2, "truncate_val_test": False,
            "val_frac": 0.1, "seed": 42, "aug_probs": (0.15, 0.25, 0.5),
            "quality_params": {}, "tasks_file_fingerprint": None,
            "fg_min_pixels_frac": 0.0002,
            "dripp_manifests_fingerprint": "manifest",
        }
        old = preprocessing.DatasetSignature(version="v1.07", **values)
        current = preprocessing.DatasetSignature(
            version=preprocessing.DATASET_CACHE_VERSION, **values
        )
        self.assertNotEqual(old.to_hash(), current.to_hash())

    def test_cached_dataset_reconstruction_preserves_configured_foreground_fraction(self):
        with tempfile.TemporaryDirectory() as root:
            fake = SimpleNamespace(
                sequences=[], transform=None, max_frames_per_sequence=8,
                min_fg_frames_in_window=2, truncate=False,
            )
            config = SimpleNamespace(
                dripp=SimpleNamespace(output_dir=root, tasks_file=None),
                sampler=SimpleNamespace(
                    seq_len=8, min_fg_frames_in_window=2,
                    fg_min_pixels_frac=0.123,
                ),
                training=SimpleNamespace(val_frac=0.1, seed=42),
                prompt=None,
            )
            constructed = []

            def make_dataset(sequences, **kwargs):
                result = SimpleNamespace(sequences=sequences, **kwargs)
                constructed.append(result)
                return result

            with mock.patch.object(
                preprocessing, "_load_cached_datasets", return_value=(fake, fake, fake),
            ), mock.patch.object(preprocessing, "SegmentationSequenceDataset", side_effect=make_dataset):
                preprocessing.load_datasets(config=config, cache_root=root)

            self.assertEqual(len(constructed), 3)
            self.assertTrue(all(ds.fg_min_pixels_frac == 0.123 for ds in constructed))


if __name__ == "__main__":
    unittest.main()
