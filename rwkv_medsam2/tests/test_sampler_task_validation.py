import unittest

from rwkv_medsam2.dataset import BalancedTaskBatchSampler


class SamplerTaskValidationTests(unittest.TestCase):
    TASKS = {"brain_regions": {
        "classes": ["organ"], "datasets": {"Neonatal": ["T1", "T2"]},
    }}

    @staticmethod
    def record(subset="T1", tasks=None):
        return {"dataset": "Neonatal", "subdataset": subset, "dim": 3,
                "tasks": ["brain_regions"] if tasks is None else tasks}

    def test_stale_empty_task_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Neonatal/T1: 1"):
            BalancedTaskBatchSampler([self.record(tasks=[])], self.TASKS, batch_size=1)

    def test_stale_default_only_catalog_is_rejected(self):
        catalog = {"brain_regions": {"datasets": {"Neonatal": ["default"]}}}
        with self.assertRaisesRegex(ValueError, "no eligible task assignments"):
            BalancedTaskBatchSampler([self.record()], catalog, batch_size=1)

    def test_one_invalid_record_cannot_hide_among_valid_records(self):
        with self.assertRaisesRegex(ValueError, "Neonatal/T2: 1"):
            BalancedTaskBatchSampler(
                [self.record(), self.record("T2", [])], self.TASKS, batch_size=1,
            )

    def test_corrected_records_are_sampled_across_epochs(self):
        sampler = BalancedTaskBatchSampler(
            [self.record(), self.record("T2")], self.TASKS, batch_size=1, seed=42,
        )
        for epoch in (0, 19):
            sampler.set_epoch(epoch)
            self.assertEqual({i for batch in sampler for i, task in batch}, {0, 1})


if __name__ == "__main__":
    unittest.main()
