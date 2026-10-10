"""
Test consistent split assignment for related groups with the same identifier.
"""

import unittest
from unittest.mock import patch

from dripp.indexer import build_groupings


class RelatedGroupSplitTests(unittest.TestCase):
    """
    Check that related image/mask groups remain on one dataset split.
    """

    def test_groups_with_same_identifier_stay_on_same_split(self):
        """
        Verify that related groups with the same identifier share a dataset split.

        Args:
            None.

        Returns:
            None.
        """
        base = "/data/research/Cardiac MRI/Fixed_Niftis_v2"
        patients = ("SCD0000101", "SCD0000201")
        images = []
        masks = []

        for patient in patients:
            for phase in ("00", "01"):
                name = f"{patient}_phase{phase}"
                images.append(
                    {"path": f"{base}/{name}/{name}_image.nii.gz", "id": name}
                )
                for class_name in (
                    "bloodpool",
                    "myocardium",
                    "epicardium",
                    "endocardium",
                ):
                    masks.append(
                        {
                            "path": f"{base}/{name}/{name}_{class_name}_mask.nii.gz",
                            "id": f"{name}_{class_name}",
                        }
                    )

        metadata = {
            "train_ratio": 0.5,
            "modalities": ["MRI"],
            "grouping_strategy": "regex-folder",
            "grouping_regex": (
                r"images: ^(?P<id>SCD\d{7})_phase(?P<phase>\d{2})/"
                r"(?P=id)_phase(?P=phase)_image\.nii(?:\.gz)?$; masks: "
                r"^(?P<id>SCD\d{7})_phase(?P<phase>\d{2})/"
                r"(?P=id)_phase(?P=phase)_(?:bloodpool|myocardium|epicardium|endocardium)"
                r"_mask\.nii(?:\.gz)?$"
            ),
            "mask_classes": (
                "bloodpool=bloodpool|myocardium=myocardium|"
                "epicardium=epicardium|endocardium=endocardium"
            ),
            "segmentation_tasks": (
                "cardiac_substructures=bloodpool,endocardium,epicardium,myocardium"
            ),
        }
        index_data = {
            "images_train": images,
            "images_test": [],
            "mask_files": masks,
        }

        with patch(
            "dripp.indexer.random.random", side_effect=(0.1, 0.9)
        ) as random_value:
            result = build_groupings(index_data, base, metadata)

        subdataset = result["subdatasets"][0]
        self.assertEqual(random_value.call_count, 2)
        self.assertEqual(
            {group["identifier"] for group in subdataset["train"]},
            {"SCD0000101"},
        )
        self.assertEqual(
            {group["identifier"] for group in subdataset["test"]},
            {"SCD0000201"},
        )
        self.assertTrue(
            all(
                len(group["masks"]) == 4
                for group in subdataset["train"] + subdataset["test"]
            )
        )


if __name__ == "__main__":
    unittest.main()
