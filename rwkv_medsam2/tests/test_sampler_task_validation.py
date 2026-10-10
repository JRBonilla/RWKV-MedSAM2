"""
Test balanced-sampler rejection of stale task metadata and valid epoch sampling.
"""

import unittest

from rwkv_medsam2.dataset import BalancedTaskBatchSampler


class SamplerTaskValidationTests(unittest.TestCase):
    """
    Test task-membership validation and balanced sampling across epochs.
    """

    TASKS = {
        "brain_regions": {
            "classes": ["organ"],
            "datasets": {"Neonatal": ["T1", "T2"]},
        }
    }

    @staticmethod
    def record(subset="T1", tasks=None):
        """
        Build a dataset record with configurable subdataset and task memberships.

        Args:
            subset (str): Neonatal subdataset label assigned to the fixture record.
            tasks (list[str] | None): Explicit task memberships, or None to use the fixture
                default.

        Returns:
            dict: Neonatal volume metadata for balanced-sampler validation.
        """
        return {
            "dataset": "Neonatal",
            "subdataset": subset,
            "dim": 3,
            "tasks": ["brain_regions"] if tasks is None else tasks,
        }

    def test_stale_empty_task_cache_is_rejected(self):
        """
        Verify rejection of cached sequence records with no valid task memberships.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "Neonatal/T1: 1"):
            BalancedTaskBatchSampler([self.record(tasks=[])], self.TASKS, batch_size=1)

    def test_stale_default_only_catalog_is_rejected(self):
        """
        Verify rejection of stale task catalogs lacking eligible subdataset assignments.

        Args:
            None.

        Returns:
            None.
        """
        catalog = {"brain_regions": {"datasets": {"Neonatal": ["default"]}}}
        with self.assertRaisesRegex(ValueError, "no eligible task assignments"):
            BalancedTaskBatchSampler([self.record()], catalog, batch_size=1)

    def test_one_invalid_record_cannot_hide_among_valid_records(self):
        """
        Verify one invalid task assignment is detected among otherwise valid records.

        Args:
            None.

        Returns:
            None.
        """
        with self.assertRaisesRegex(ValueError, "Neonatal/T2: 1"):
            BalancedTaskBatchSampler(
                [self.record(), self.record("T2", [])],
                self.TASKS,
                batch_size=1,
            )

    def test_corrected_records_are_sampled_across_epochs(self):
        """
        Verify valid records remain represented across sampler epochs.

        Args:
            None.

        Returns:
            None.
        """
        sampler = BalancedTaskBatchSampler(
            [self.record(), self.record("T2")],
            self.TASKS,
            batch_size=1,
            seed=42,
        )
        for epoch in (0, 19):
            sampler.set_epoch(epoch)
            self.assertEqual({i for batch in sampler for i, task in batch}, {0, 1})


if __name__ == "__main__":
    unittest.main()
