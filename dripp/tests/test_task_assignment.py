"""
Test inherited class definitions and task registration for grouped subdatasets.
"""

import unittest
from unittest.mock import patch

from dripp.helpers import match_tasks_for_class, parse_segmentation_tasks
from dripp.indexer import GLOBAL_TASKS, build_groupings, register_group_tasks


class TaskAssignmentTests(unittest.TestCase):
    """
    Check inherited class mappings and task memberships across subdatasets.
    """

    def grouped(self, names, mask_classes="organ=1", tasks="segment=organ"):
        """
        Build synthetic grouped metadata with configurable classes and tasks.

        Args:
            names (list[str]): Subdataset names for the task-assignment grouping
                fixture.
            mask_classes (str): Class-to-rule assignment expression for the metadata
                fixture.
            tasks (str): Task-to-class assignment expression for the metadata fixture.

        Returns:
            dict: Grouping result containing subdatasets, inherited class definitions,
                and task memberships.
        """
        base = "/fixture"
        metadata = {
            "train_ratio": 1.0,
            "modalities": ["MRI"],
            "grouping_strategy": "regex-file",
            "grouping_regex": ", ".join(
                f"[MRI: {name}] {{images: ^(?P<id>01)_{name}\\.nii$; "
                f"masks: ^(?P<id>01)_{name}_mask\\.nii$}}"
                for name in names
            ),
            "mask_classes": mask_classes,
            "segmentation_tasks": tasks,
        }
        index = {
            "images_train": [
                {"path": f"{base}/01_{name}.nii", "id": "01"} for name in names
            ],
            "images_test": [],
            "mask_files": [
                {"path": f"{base}/01_{name}_mask.nii", "id": "01"} for name in names
            ],
        }

        return build_groupings(index, base, metadata)

    def test_default_classes_inherit_into_all_affected_named_subsets(self):
        """
        Verify inheritance of default classes into every affected named subdataset.

        Args:
            None.

        Returns:
            None.
        """
        cases = {
            "Neonatal": ["T1", "T2"],
            "NCI": ["PROSTATE-DIAGNOSIS", "Prostate-3T"],
            "ROSE-1": ["DVC", "SVC", "SVC_DVC"],
            "WBC": ["Dataset-1", "Dataset-2"],
            "WMH": ["Amsterdam", "Singapore", "Utrecht"],
        }
        for dataset, names in cases.items():
            with self.subTest(dataset=dataset), patch.dict(
                GLOBAL_TASKS, {}, clear=True
            ):
                groups = self.grouped(names)
                self.assertEqual({s["name"] for s in groups["subdatasets"]}, set(names))
                for sub in groups["subdatasets"]:
                    self.assertEqual(sub["tasks"], ["segment"])
                    self.assertEqual(sub["classes"], {"organ": "1"})

                register_group_tasks(dataset, groups, {"segment": ["organ"]})
                self.assertEqual(
                    GLOBAL_TASKS["segment"]["datasets"][dataset], set(names)
                )

    def test_explicit_subset_classes_override_default(self):
        """
        Verify that explicit subdataset class definitions override defaults.

        Args:
            None.

        Returns:
            None.
        """
        groups = self.grouped(
            ["T1", "T2"],
            "[default] {organ=1}, [T2] {lesion=2}",
            "segment=organ|tumor=lesion",
        )
        subs = {s["name"]: s for s in groups["subdatasets"]}
        self.assertEqual(subs["T1"]["tasks"], ["segment"])
        self.assertEqual(subs["T2"]["tasks"], ["tumor"])
        self.assertEqual(subs["T2"]["classes"], {"lesion": "2"})

    def test_heart_spelling_variant_keeps_original_mask_class(self):
        """
        Verify task matching of a spelling variant while retaining the original class
        name.

        Args:
            None.

        Returns:
            None.
        """
        tasks = parse_segmentation_tasks("cardiac_substructures=left_atrium")
        groups = self.grouped(
            ["Heart"], "[Heart] {left atrium=1}", "cardiac_substructures=left_atrium"
        )
        self.assertEqual(groups["subdatasets"][0]["tasks"], ["cardiac_substructures"])
        self.assertEqual(groups["subdatasets"][0]["classes"], {"left atrium": "1"})

        with patch.dict(GLOBAL_TASKS, {}, clear=True):
            register_group_tasks("MSD", groups, tasks)
            self.assertEqual(
                GLOBAL_TASKS["cardiac_substructures"]["datasets"], {"MSD": {"Heart"}}
            )
            self.assertIn(
                "left atrium", GLOBAL_TASKS["cardiac_substructures"]["classes"]
            )

    def test_distinct_classes_are_not_merged(self):
        """
        Verify that separate target classes remain distinct during task assignment.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(
            match_tasks_for_class("right atrium", {"cardiac": ["left_atrium"]}), []
        )

    def test_unmatched_tasks_fail_before_saving_empty_assignments(self):
        """
        Verify failure for unmatched tasks before empty assignments are saved.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(
            ValueError, "No segmentation tasks match subdataset 'T1'"
        ):
            self.grouped(["T1"], tasks="segment=unrelated")


if __name__ == "__main__":
    unittest.main()
