"""CPU regression checks for the isolated Gamma campaign controller."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('campaign', ROOT / 'deploy/gamma/performance-campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_resume_rejects_changed_identity_and_incomplete_repetitions(self):
        from rwkv_medsam2.evaluation.shared.data import digest
        trial = dict(trial_id='id', model='model', unit_id='unit', kind='prediction', repetitions=2)
        case = dict(unit_id='unit', original='frozen inputs')
        row = dict(status='complete', trial=trial, plan_sha256='plan', identity_sha256='identity',
                   case_sha256=digest(case), repetitions=[{}, {}])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trial.json'
            def check(value):
                path.write_text(json.dumps(value))
                return campaign.validate_completed(path, trial, {'plan_sha256': 'plan'}, {'model': 'identity'}, {'unit': case})
            self.assertEqual(check(row), row)
            with self.assertRaises(ValueError):
                check(dict(row, identity_sha256='other checkpoint or runtime'))
            with self.assertRaises(ValueError):
                check(dict(row, repetitions=[{}]))
            with self.assertRaises(ValueError):
                check(dict(row, case_sha256='changed prompts'))

    def test_overview_uses_completed_jobs_and_readable_text(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            writer = campaign.Overview(folder)
            writer.update(dict(condition='positive', state='running', completed=3, total=10),
                          {'corrected_hiera': 10}, {'corrected_hiera': 3})
            writer.close()
            events = EventAccumulator(str(folder)); events.Reload()
            scalar = events.Scalars('Completion percent/Overall')[0]
            self.assertEqual(scalar.step, 3)
            self.assertAlmostEqual(scalar.value, 30)
            text = events.Tensors('Progress/text_summary')[0].tensor_proto.string_val[0].decode()
            self.assertIn('Hiera (corrected)', text)
            self.assertIn('Completed 3 / 10', text)



if __name__ == '__main__':
    unittest.main()
