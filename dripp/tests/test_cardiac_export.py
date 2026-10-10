"""
Test cardiac phase ordering, canonical geometry, and overlapping binary mask exports.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "preprocessing"
    / "fix_orientation_and_export.py"
)
SPEC = importlib.util.spec_from_file_location("cardiac_export", SCRIPT)
cardiac_export = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys

sys.modules[SPEC.name] = cardiac_export
SPEC.loader.exec_module(cardiac_export)


class CardiacExportTests(unittest.TestCase):
    """
    Check cardiac phase grouping and preservation of image/mask geometry.
    """

    def frame(self, name, position, trigger):
        """
        Construct a cardiac DICOM frame fixture with slice position and trigger time.

        Args:
            name (str): Synthetic source filename identifying this frame.
            position (float): Synthetic slice position along the acquisition direction.
            trigger (float): Synthetic cardiac trigger time used to rank acquisition
                phases.

        Returns:
            DicomFrame: Frame record for cardiac phase-ordering tests.
        """
        return cardiac_export.DicomFrame(Path(name), position, trigger, int(trigger))

    def test_phase_split_uses_trigger_rank_and_assigns_singletons(self):
        """
        Verify trigger-ranked cardiac phases and assignment of singleton slices.

        Args:
            None.

        Returns:
            None.
        """
        frames = [
            self.frame("z0_early.dcm", 0.0, 100.0),
            self.frame("z0_late.dcm", 0.0, 700.0),
            self.frame("z1_early.dcm", 10.0, 110.0),
            self.frame("z1_late.dcm", 10.0, 710.0),
            self.frame("z2_late.dcm", 20.0, 705.0),
        ]
        phases = cardiac_export.split_cardiac_phases(frames)
        self.assertEqual(
            [frame.path.name for frame in phases[0]], ["z0_early.dcm", "z1_early.dcm"]
        )
        self.assertEqual(
            [frame.path.name for frame in phases[1]],
            ["z0_late.dcm", "z1_late.dcm", "z2_late.dcm"],
        )

    def test_canonical_orientation_preserves_image_mask_geometry(self):
        """
        Verify that canonical orientation preserves matching image and mask geometry.

        Args:
            None.

        Returns:
            None.
        """
        array = np.zeros((9, 16, 16), dtype=np.uint8)
        array[:, 4:12, 5:13] = 1
        image = cardiac_export.sitk.GetImageFromArray(array)
        image.SetSpacing((1.3, 1.3, 10.0))
        image.SetDirection(
            (
                -0.1045921,
                -0.7331921,
                -0.6719299,
                0.3463809,
                -0.6601881,
                0.6664623,
                -0.9322450,
                -0.1630370,
                0.3230142,
            )
        )
        mask = cardiac_export.sitk.GetImageFromArray(array)
        mask.CopyInformation(image)

        oriented_image = cardiac_export._canonicalize_orientation(image)
        oriented_mask = cardiac_export._canonicalize_orientation(mask)

        orienter = cardiac_export.sitk.DICOMOrientImageFilter()
        self.assertEqual(
            orienter.GetOrientationFromDirectionCosines(oriented_image.GetDirection()),
            "LPS",
        )
        self.assertEqual(oriented_mask.GetSize(), oriented_image.GetSize())
        self.assertEqual(oriented_mask.GetSpacing(), oriented_image.GetSpacing())
        self.assertEqual(oriented_mask.GetOrigin(), oriented_image.GetOrigin())
        self.assertEqual(oriented_mask.GetDirection(), oriented_image.GetDirection())
        np.testing.assert_array_equal(
            cardiac_export.sitk.GetArrayFromImage(oriented_mask),
            cardiac_export.sitk.GetArrayFromImage(oriented_image),
        )

    def test_separate_binary_masks_preserve_overlapping_regions(self):
        """
        Verify that independent binary mask exports preserve overlapping target regions.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            grouped = {}

            for index in range(2):
                stem = f"IM-0001-{index:04d}"
                frames.append(self.frame(root / f"{stem}.dcm", float(index), 100.0))
                endo = np.zeros((5, 5), dtype=np.uint8)
                endo[1:4, 1:4] = 255
                epi = np.ones((5, 5), dtype=np.uint8) * 255
                bloodpool = np.zeros((5, 5), dtype=np.uint8)
                bloodpool[2:4, 2:4] = 255
                myocardium = np.logical_and(epi != 0, endo == 0).astype(np.uint8) * 255
                paths = {}

                for name, array in {
                    "endocardium": endo,
                    "epicardium": epi,
                    "bloodpool": bloodpool,
                    "myocardium": myocardium,
                }.items():
                    path = root / f"{stem}-{name}.png"
                    Image.fromarray(array).save(path)
                    paths[name] = path

                grouped[stem] = paths

            masks = cardiac_export._phase_masks(frames, grouped, (5, 5))
            self.assertEqual(set(masks), set(cardiac_export.CLASS_NAMES))
            self.assertTrue(
                np.logical_and(masks["endocardium"], masks["epicardium"]).any()
            )
            self.assertTrue(
                np.logical_and(masks["bloodpool"], masks["endocardium"]).any()
            )
            self.assertEqual(masks["bloodpool"].shape, (2, 5, 5))


if __name__ == "__main__":
    unittest.main()
