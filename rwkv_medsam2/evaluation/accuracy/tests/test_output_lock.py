"""
Test exclusive evaluation-output ownership and claim release after exceptions.
"""

from pathlib import Path
import tempfile
import unittest

from rwkv_medsam2.evaluation.accuracy.runner import output_lock


class OutputLockTests(unittest.TestCase):
    """
    Test exclusive output ownership and release on normal or exceptional exit.
    """

    def test_exclusive_owner_and_release(self):
        """
        Verify exclusive output ownership and claim release after normal exit.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "shard"
            with output_lock(folder):
                self.assertTrue((folder / ".writer.claim/owner.json").exists())
                with self.assertRaisesRegex(RuntimeError, "Another writer"):
                    with output_lock(folder):
                        self.fail("Second writer entered")

            self.assertFalse((folder / ".writer.claim").exists())
            with output_lock(folder):
                pass

    def test_exception_releases_claim(self):
        """
        Verify that exceptions release the evaluation-output ownership claim.

        Args:
            None.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, "fixture"):
                with output_lock(folder):
                    raise ValueError("fixture")
            with output_lock(folder):
                pass


if __name__ == "__main__":
    unittest.main()
