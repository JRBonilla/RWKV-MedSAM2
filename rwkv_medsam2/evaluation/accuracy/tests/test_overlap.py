"""CPU integration coverage for cached inputs and spawned surface workers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import SimpleITK as sitk

from rwkv_medsam2.evaluation.accuracy.inputs import InputCache
from rwkv_medsam2.evaluation.accuracy.records import PredictionArchive
from rwkv_medsam2.evaluation.accuracy.surfaces import analyze
from rwkv_medsam2.evaluation.accuracy.surface_workers import SurfacePool, process_surface
from rwkv_medsam2.evaluation.shared.data import (
    add_counts, atomic_json, canonical_mask, condition_case, confusion, digest,
    file_hash, freeze_case, overlaps, read_case, sequence_plan)


def fixture(root, dim=3, axis=0, reverse=False, video=False):
    image = root / ('image.png' if dim == 2 else 'image.nii.gz')
    mask = root / ('mask.png' if dim == 2 else 'mask.nii.gz')
    if dim == 2:
        x = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
        y = np.zeros_like(x)
        y[6:15, 8:20] = 1
        Image.fromarray(x).save(image)
        Image.fromarray(y).save(mask)
    else:
        x = np.arange(6 * 16 * 20, dtype=np.float32).reshape(6, 16, 20)
        y = np.zeros_like(x, dtype=np.uint8)
        y[1:5, 5:12, 7:15] = 1
        for path, array in ((image, x), (mask, y)):
            v = sitk.GetImageFromArray(array)
            v.SetSpacing((.7, 1.2, 2.5))
            sitk.WriteImage(v, str(path))
    case = dict(case_id='case', target_id='organ', dataset='test', task='segmentation',
                source_id='source', source_identity_status='unresolved', source_note='Test fixture',
                modality='synthetic', image=str(image), mask=str(mask), dim=dim, label=1,
                target_scope='semantic', split='test', provenance_reviewed=True,
                preprocessing_note='Unit-test fixture only')
    if dim == 3:
        case.update(axis_zyx=axis, reverse_slices=reverse, anatomical_volume=True,
                    geometry_note='Synthetic axis', spacing_verified_mm=True,
                    spacing_provenance_note='Synthetic fixture spacing')
    if video:
        case.update(sequence_kind='video', anatomical_volume=False, spacing_verified_mm=False,
                    temporal_axis_note='Synthetic forward time axis')
    return freeze_case(case, root, 'test')


def prediction_result(root, case, mode='box', run_id='test-run'):
    result_path = root / 'cases' / (case['unit_id'] + '-' + mode + '.json')
    ids, mapping, _ = sequence_plan(case)
    _, target, _ = read_case(case, root)
    identity = dict(run_id=run_id, unit_key=case['unit_id'] + ':' + mode,
                    case_sha256=digest(case), sequence_condition=case['sequence_condition'], frame_ids=ids)
    archive = PredictionArchive(root / 'predictions' / (result_path.stem + '.zip'), identity)
    frames = []
    for t in ids:
        truth = canonical_mask(target[t])
        pred = np.zeros_like(truth)
        pred[2:, 2:] = truth[:-2, :-2]
        archive.frame(t, pred, None)
        frames.append(dict(frame=t, sequence_index=mapping[t], support=t in case['support_frames'],
                           **confusion(pred, truth)))
    query = frames if case['dim'] == 2 else [r for r in frames if not r['support']]
    positive_queries = [r for r in query if r['frame'] in case['positive_frame_ids']]
    row = dict(identity, model='fixture', mode=mode, status='complete', encoder_inputs_verified=True,
               case={k: case[k] for k in ('unit_id', 'case_id', 'target_id', 'dataset', 'task',
                                          'source_id', 'source_identity_status', 'target_scope', 'modality', 'dim')},
               frames=len(ids), stored_frames=len(target), shared_case_sha256=case['shared_case_sha256'],
               support_frames=case['support_frames'], query_frames=len(query), per_frame=frames,
               prediction_archive=archive.commit())
    for name, values in (('primary', query), ('all_frame_assisted', frames),
                         ('positive_query_overlap', positive_queries)):
        counts = add_counts(values)
        row[name] = dict(counts, **overlaps(counts))
    if case.get('sequence_kind') == 'video':
        row['case']['sequence_kind'] = 'video'
    atomic_json(result_path, row)
    return row, result_path


class InputCacheTests(unittest.TestCase):
    def test_exact_uncached_parity_across_axes_and_reverse(self):
        for dim, axis, reverse in ((2, 0, False), (3, 0, False), (3, 1, True), (3, 2, False)):
            with self.subTest(dim=dim, axis=axis, reverse=reverse), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case = fixture(root, dim, axis, reverse)
                cache = InputCache(1024**2)
                expected = read_case(case, root)
                actual = cache.read_case(case, root)
                for x, y in zip(actual[:2], expected[:2]):
                    np.testing.assert_array_equal(x, y)
                    self.assertFalse(x.flags.writeable)
                self.assertEqual(actual[2], expected[2])
                cache.read_case(case, root)
                self.assertEqual(cache.stats, dict(hash_reads=2, array_loads=2, array_hits=2))
                other_label = dict(case, label=2)
                cache.read_case(other_label, root)
                self.assertEqual(cache.stats['array_loads'], 3)  # image reused across targets

    def test_surface_never_decodes_image_and_checks_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = fixture(root)
            cache = InputCache(1024**2)
            with patch('rwkv_medsam2.evaluation.accuracy.inputs._image', side_effect=AssertionError('image decoded')):
                target, transform = cache.read_target(case, root)
                self.assertEqual(transform, case['transform'])
                self.assertEqual(target.shape, (6, 16, 20))
                cache.read_target(case, root)
            self.assertEqual(cache.stats['array_loads'], 1)
            with Path(case['image']).open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                cache.read_target(case, root)

    def test_cache_budget_and_geometry_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = fixture(root)
            cache = InputCache(1)
            cache.read_case(case, root)
            self.assertEqual(cache.bytes, 0)
            cache.read_case(case, root)
            self.assertEqual(cache.stats['array_loads'], 4)
            wrong = dict(case, transform=dict(case['transform'], frames=99))
            with self.assertRaisesRegex(ValueError, 'frame count'):
                cache.read_target(wrong, root)

    def test_fractional_and_nonfinite_annotations_are_still_rejected(self):
        for value in (.5, float('nan')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case = fixture(root)
                image = sitk.ReadImage(case['mask'])
                array = sitk.GetArrayFromImage(image).astype(np.float32)
                array[0, 0, 0] = value
                changed = sitk.GetImageFromArray(array)
                changed.CopyInformation(image)
                sitk.WriteImage(changed, case['mask'])
                case['file_sha256']['mask'] = file_hash(case['mask'])
                # Some ITK NIfTI readers replace NaN pixels with zero. Inject
                # decoded values here to exercise our finite-value guard itself.
                with patch('rwkv_medsam2.evaluation.accuracy.inputs._volume',
                           return_value=(array, case['transform']['geometry'])):
                    with self.assertRaisesRegex(ValueError, 'finite integers'):
                        InputCache().read_target(case, root)


class SurfaceOverlapTests(unittest.TestCase):
    def test_accuracy_runner_drains_cpu_jobs_and_resumes_without_inference(self):
        self._runner_drains_and_resumes(video=False)

    def test_video_runner_keeps_metadata_and_resumes(self):
        self._runner_drains_and_resumes(video=True)

    def _runner_drains_and_resumes(self, video):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from rwkv_medsam2.evaluation.accuracy.runner import run
        from rwkv_medsam2.evaluation.shared.models import CONTROLS
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = fixture(root, 3 if video else 2, video=video)
            case['modes'] = ['box']
            template, _ = prediction_result(root/'reference', case)
            manifest = root/'frozen'
            manifest.mkdir()
            atomic_json(manifest/'manifest.json', {'fixture': True})
            atomic_json(root/'checkpoints.json', {})
            env = {'packages': {p: 'test' for p in ('numpy', 'Pillow', 'SimpleITK')}}
            meta = dict(cases_sha256='cases', protocol_sha256='protocol', preparation_environment=env,
                        sequence_condition='positive_frames')
            args = SimpleNamespace(manifest=manifest, shards=1, shard=0, model='fixture',
                checkpoints=root/'checkpoints.json', device='cuda:0', data_root=root, output=root/'runs',
                save_logits=False, qualitative_cases=0, input_cache_gib=.01, surface_workers=1,
                surface_cache_gib=.01)
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: 'CPU test double',
                get_device_capability=lambda: (0, 0)), version=SimpleNamespace(cuda=None), __version__='test')
            fake_model = SimpleNamespace(**CONTROLS,
                sam_mask_decoder=SimpleNamespace(dynamic_multimask_via_stability=False))
            def evaluate(model, frozen, mode, data_root, device, progress, archive, input_cache=None):
                _, target, _ = input_cache.read_case(frozen, data_root)
                for t in frozen['frame_ids']:
                    truth = canonical_mask(target[t])
                    prediction = np.zeros_like(truth)
                    prediction[2:, 2:] = truth[:-2, :-2]
                    archive.frame(t, prediction, None)
                scores = {k: v for k, v in template.items() if k not in
                    ('run_id', 'status', 'unit_key', 'model', 'mode', 'case', 'case_sha256', 'prediction_archive')}
                scores['diagnostics'] = dict(query={}, probability_query=dict(brier=0., log_loss=0., ece20=0.))
                return scores, None
            with ExitStack() as stack:
                stack.enter_context(patch('rwkv_medsam2.evaluation.accuracy.runner.read_frozen', return_value=(meta, [case])))
                for name, value in [('environment', env), ('checkpoint_identity', {'sha256': 'fixture'}),
                                    ('source_snapshot', {'sha256': 'test'}), ('initialization_assets', {}),
                                    ('configure_runtime', fake_torch), ('load_model', fake_model)]:
                    stack.enter_context(patch('rwkv_medsam2.evaluation.shared.models.' + name, return_value=value))
                inference = stack.enter_context(patch('rwkv_medsam2.evaluation.accuracy.runner.evaluate_case', side_effect=evaluate))
                run(args)
                self.assertEqual(inference.call_count, 1)
                shard = args.output/'fixture/shard-0000-of-0001'
                complete = json.loads((shard/'complete.json').read_text())
                self.assertEqual(complete['surfaces_completed'], 1)
                self.assertTrue((shard/'predictions-complete.json').exists())
                inference.side_effect = AssertionError('Resume should not rerun inference')
                run(args)
                self.assertEqual(inference.call_count, 1)
            events = EventAccumulator(str(args.output/'tensorboard/fixture/shard-0000-of-0001'))
            events.Reload()
            self.assertEqual(events.Scalars('boundary/progress/fraction')[-1].value, 1.)
            if video:
                self.assertTrue(any('/video/' in tag for tag in events.Tags()['scalars']))
                self.assertFalse(any('/3d/' in tag for tag in events.Tags()['scalars']))

    def test_standalone_report_reuses_inline_results_and_logs_progress(self):
        from types import SimpleNamespace
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from rwkv_medsam2.evaluation.accuracy.surfaces import surfaces
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'frozen'
            manifest.mkdir()
            atomic_json(manifest/'manifest.json', {'test': True})
            case = fixture(root, 2)
            identity = dict(model='fixture', manifest_sha256=file_hash(manifest/'manifest.json'))
            shard = root/'runs/fixture/shard-0000-of-0001'
            row, path = prediction_result(shard, case, run_id=digest(identity))
            atomic_json(shard/'job.json', dict(identity=identity, run_id=digest(identity),
                                             expected=[row['unit_key']]))
            process_surface(row, path, case, root, shard/'surfaces/cases'/path.name)
            args = SimpleNamespace(manifest=manifest, runs=root/'runs', data_root=root,
                                   output=root/'report', workers=2, cache_gib=.01)
            with patch('rwkv_medsam2.evaluation.accuracy.runner.read_frozen',
                       return_value=({'sequence_condition': 'positive_frames'}, [case])):
                surfaces(args)
            summary = json.loads((args.output/'summary.json').read_text())
            self.assertEqual(summary['reused_inline_results'], 1)
            self.assertEqual(summary['coverage']['fixture'], dict(expected=1, completed=1))
            events = EventAccumulator(str(args.output/'tensorboard'))
            events.Reload()
            self.assertEqual(events.Scalars('boundary/progress/completed')[-1].value, 1.)
            self.assertTrue(any(tag.endswith('/hd95') for tag in events.Tags()['scalars']))

    def test_mask_only_metrics_match_full_reader_all_conditions(self):
        for dim, condition in ((2, 'positive_frames'), (3, 'positive_frames'), (3, 'full_scan')):
            with self.subTest(dim=dim, condition=condition), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case = condition_case(fixture(root, dim), condition)
                row, path = prediction_result(root, case)
                actual = analyze(row, path, case, root)
                _, target, transform = read_case(case, root)
                with patch.object(InputCache, 'read_target', return_value=(target, transform)):
                    expected = analyze(row, path, case, root)
                self.assertEqual(actual, expected)

    def test_spawn_queue_resume_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = fixture(root)
            row, path = prediction_result(root, case)
            serial = root / 'serial.json'
            process_surface(row, path, case, root, serial)
            pool = SurfacePool(2, 1024**2)
            try:
                destinations = [root / f'parallel-{i}.json' for i in range(pool.capacity)]
                for output in destinations:
                    pool.submit(row, path, case, root, output)
                with self.assertRaisesRegex(RuntimeError, 'queue is full'):
                    pool.submit(row, path, case, root, root/'overflow.json')
                # Parent can do independent work while spawned workers compute.
                (root/'parent-continued').write_text('GPU producer need not wait for each surface result')
                done = []
                while pool.pending:
                    done.extend(pool.collect(block=True))
                self.assertEqual(len(done), 4)
                for output in destinations:
                    self.assertEqual(json.loads(output.read_text()), json.loads(serial.read_text()))
                pool.submit(row, path, case, root, destinations[0])
                self.assertTrue(pool.collect(block=True)[0]['reused'])
                # A corrupt archive must be rejected even when a surface result exists.
                archive = root/'predictions'/row['prediction_archive']['filename']
                with archive.open('ab') as f:
                    f.write(b'corruption')
                pool.submit(row, path, case, root, destinations[0])
                with self.assertRaisesRegex(ValueError, 'Prediction archive changed'):
                    pool.collect(block=True)
            finally:
                pool.close()

    def test_saved_surface_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = fixture(root, 2)
            row, path = prediction_result(root, case)
            destination = root/'surface.json'
            process_surface(row, path, case, root, destination)
            saved = json.loads(destination.read_text())
            saved['hd95'] = 999
            atomic_json(destination, saved)
            with self.assertRaisesRegex(ValueError, 'surface result changed'):
                process_surface(row, path, case, root, destination)


if __name__ == '__main__':
    unittest.main()
