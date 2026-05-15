import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dripp.output_structure import (
    DEFAULT_GROUP_FOLDER_TEMPLATE,
    OutputStructureError,
    render_output_dirs,
    validate_group_folder_template,
    validate_leaf_folder,
)


class OutputStructureTests(unittest.TestCase):
    def test_default_template_matches_existing_subdataset_layout(self):
        dirs = render_output_dirs(
            "F:/Preprocessed",
            "ExampleCT",
            {"split": "train", "identifier": "case001"},
            "ct",
            "liver",
            "train_case001_slice03",
        )

        self.assertEqual(
            dirs["img_out_dir"],
            "F:/Preprocessed/ExampleCT/ct/liver/train/case001/slice03/images",
        )
        self.assertEqual(
            dirs["mask_out_dir"],
            "F:/Preprocessed/ExampleCT/ct/liver/train/case001/slice03/masks",
        )

    def test_default_template_omits_missing_subdataset_segment(self):
        dirs = render_output_dirs(
            "F:/Preprocessed",
            "ExampleCT",
            {"split": "test", "identifier": "case002"},
            "ct",
            None,
            "test_case002",
        )

        self.assertEqual(
            dirs["img_out_dir"],
            "F:/Preprocessed/ExampleCT/ct/test/case002/images",
        )

    def test_id_parts_strips_split_prefix(self):
        dirs = render_output_dirs(
            "F:/Preprocessed",
            "ExampleCT",
            {"split": "train", "identifier": "case001"},
            "ct",
            "default",
            "train_case001_partA",
        )

        self.assertIn("/train/case001/partA/images", dirs["img_out_dir"])
        self.assertNotIn("/train/train/", dirs["img_out_dir"])

    def test_custom_template_and_leaf_folders(self):
        dirs = render_output_dirs(
            "F:/Preprocessed",
            "ExampleCT",
            {"split": "train", "identifier": "case001"},
            "ct",
            "default",
            "train_case001",
            "{dataset}/{split}/{modality}/{id}",
            "imgs",
            "labels",
        )

        self.assertEqual(
            dirs["img_out_dir"],
            "F:/Preprocessed/ExampleCT/train/ct/case001/imgs",
        )
        self.assertEqual(
            dirs["mask_out_dir"],
            "F:/Preprocessed/ExampleCT/train/ct/case001/labels",
        )

    def test_template_validation_rejects_unsafe_values(self):
        invalid_templates = [
            "{dataset}/{unknown}",
            "../{dataset}",
            "{dataset}//{id}",
            "/outputs/{dataset}",
        ]
        for template in invalid_templates:
            with self.subTest(template=template):
                with self.assertRaises(OutputStructureError):
                    validate_group_folder_template(template)

    def test_leaf_folder_validation_rejects_unsafe_values(self):
        invalid_leaf_folders = ["", "../labels", "nested/labels", "{split}", "/labels"]
        for folder in invalid_leaf_folders:
            with self.subTest(folder=folder):
                with self.assertRaises(OutputStructureError):
                    validate_leaf_folder(folder, "Masks Folder")

    def test_default_template_is_valid(self):
        self.assertEqual(
            validate_group_folder_template(DEFAULT_GROUP_FOLDER_TEMPLATE),
            DEFAULT_GROUP_FOLDER_TEMPLATE,
        )


if __name__ == "__main__":
    unittest.main()
