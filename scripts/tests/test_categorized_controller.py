"""Campaign resume must reject stale identities and incomplete scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_categorized_accuracy import verified_completion
from rwkv_medsam2.evaluation.shared.data import PROFILE, digest, file_hash
from rwkv_medsam2.evaluation.shared.models import CONTROLS


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        (self.folder/'manifest.json').write_text('{}')
        self.meta = dict(profile=PROFILE, cases_sha256='cases', protocol_sha256='protocol',
                         sequence_condition='positive_frames')
        self.cases = [dict(unit_id='target', dataset='fixture', case_id='case',
                           eligible=True, modes=['box'])]
        self.env = {'python': 'fixture'}
        self.snapshot = {'sha256': 'source'}
        self.checkpoint = {'path': 'weights', 'sha256': 'checkpoint'}
        identity = dict(profile=PROFILE, manifest_sha256=file_hash(self.folder/'manifest.json'),
                        cases_sha256='cases', protocol_sha256='protocol', model='fixture',
                        checkpoint=self.checkpoint, initialization_assets={}, source_sha256='source',
                        environment=self.env, controls=CONTROLS,
                        dynamic_multimask_via_stability=False, dtype='float32', attention_backend='math',
                        frame_batch=1, state_policy='all_frame_GPU', seed=42,
                        sequence_condition='positive_frames', recording_schema=3, save_logits=False)
        self.job = dict(run_id=digest(identity), identity=identity, shard=0, shards=1,
                        expected=['target:box'])
        self.complete = dict(run_id=self.job['run_id'], completed=1, expected=1,
                             surfaces_completed=1, surfaces_requested=True)
        self.save()

    def save(self):
        (self.folder/'job.json').write_text(json.dumps(self.job))
        (self.folder/'complete.json').write_text(json.dumps(self.complete))

    def verify(self):
        return verified_completion(self.folder, self.folder, 'fixture', self.checkpoint,
                                   self.env, self.snapshot, 1, 0,
                                   frozen=(self.meta, self.cases), assets={})

    def test_matching_completed_job_is_reused(self):
        self.assertTrue(self.verify())

    def test_missing_completion_returns_to_runner(self):
        (self.folder/'complete.json').unlink()
        self.assertFalse(self.verify())

    def test_changed_source_or_checkpoint_is_rejected(self):
        for identity in (self.snapshot, self.checkpoint):
            with self.subTest(identity=identity), patch.dict(identity, sha256='changed'):
                with self.assertRaisesRegex(ValueError, 'identity or coverage'):
                    self.verify()

    def test_incomplete_surfaces_are_rejected(self):
        self.complete['surfaces_completed'] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, 'identity or coverage'):
            self.verify()

    def test_changed_shard_membership_is_rejected(self):
        self.job['expected'] = ['other:box']
        self.save()
        with self.assertRaisesRegex(ValueError, 'identity or coverage'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
