import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dripp.output_filenames import (
    OutputFilenameError,
    render_image_filename,
    render_mask_filename,
    validate_output_filenames,
)


class OutputFilenameTests(unittest.TestCase):
    def test_default_image_filename_matches_existing_format(self):
        self.assertEqual(
            render_image_filename("2d", 3, "a1b2c3d4", ".png"),
            "a1b2c3d4_img003.png",
        )
        self.assertEqual(
            render_image_filename("video", 3, "a1b2c3d4", ".png", "modality0"),
            "a1b2c3d4_frame0003_~modality0~.png",
        )

    def test_default_mask_filename_matches_existing_format(self):
        self.assertEqual(
            render_mask_filename("2d", 3, 2, 5, "a1b2c3d4", ".png", "tumor"),
            "a1b2c3d4_img003_mask002_%tumor%_comp005.png",
        )
        self.assertEqual(
            render_mask_filename("3d", 3, 2, 5, "a1b2c3d4", ".nii.gz", "tumor", label_value=7),
            "a1b2c3d4_img003_mask002_%tumor%_label007_comp005.nii.gz",
        )

    def test_custom_segment_order_and_separator(self):
        self.assertEqual(
            render_image_filename(
                "2d",
                3,
                "a1b2c3d4",
                ".png",
                image_segments=["image_number", "short_id"],
                separator="-",
            ),
            "img003-a1b2c3d4.png",
        )

    def test_validation_rejects_unsafe_separator_and_duplicates(self):
        with self.assertRaises(OutputFilenameError):
            validate_output_filenames(["short_id"], ["short_id"], "/")
        with self.assertRaises(OutputFilenameError):
            validate_output_filenames(["short_id", "short_id"], ["short_id"], "_")


if __name__ == "__main__":
    unittest.main()
