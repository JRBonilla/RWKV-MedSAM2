from pathlib import Path
import tempfile
import unittest

from rwkv_medsam2.evaluation.accuracy.runner import output_lock


class OutputLockTests(unittest.TestCase):
    def test_exclusive_owner_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'shard'
            with output_lock(folder):
                self.assertTrue((folder/'.writer.claim/owner.json').exists())
                with self.assertRaisesRegex(RuntimeError,'Another writer'):
                    with output_lock(folder):
                        self.fail('Second writer entered')
            self.assertFalse((folder/'.writer.claim').exists())
            with output_lock(folder):pass

    def test_exception_releases_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.assertRaisesRegex(ValueError,'fixture'):
                with output_lock(folder):raise ValueError('fixture')
            with output_lock(folder):pass

if __name__=='__main__':unittest.main()
