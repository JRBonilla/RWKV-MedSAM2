"""Resumable serial Gamma performance campaign; no GPU work until main()."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from rwkv_medsam2.evaluation.accuracy.runner import engine_files, output_lock
from rwkv_medsam2.evaluation.performance.data import read_plan, local_case
from rwkv_medsam2.evaluation.performance.runner import checkpoint_for, telemetry
from rwkv_medsam2.evaluation.shared.data import atomic_json, digest, file_hash, case_kind
from rwkv_medsam2.evaluation.shared.models import environment, source_snapshot, initialization_assets, CONTROLS

LABELS = {'corrected_hiera': 'Hiera (corrected)', 'rwkv_medsam2_final_distill': 'RWKV distilled',
          'rwkv_medsam2_final_nodistill': 'RWKV non-distilled',
          'oxford_medical_sam2': 'Oxford', 'uoft_medsam2': 'UofT'}


class Overview:
    """Write TensorBoard synchronously between workers; no background logging thread."""
    def __init__(self, folder):
        from tensorboard.summary.writer.record_writer import RecordWriter
        from tensorboard.compat.proto.event_pb2 import Event
        folder.mkdir(parents=True, exist_ok=True)
        self.stream = (folder / f'events.out.tfevents.{time.time_ns()}.{socket.gethostname()}').open('xb')
        self.writer = RecordWriter(self.stream)
        self.writer.write(Event(wall_time=time.time(), file_version='brain.Event:2').SerializeToString())

    def update(self, state, totals, completed):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        lines = [f"**{state['condition']} — {state['state']}**", '',
                 f"Completed {state['completed']:,} / {state['total']:,} measurement jobs.", '',
                 '| Model | Completed | Total |', '|---|---:|---:|']
        for model in totals:
            lines.append(f'| {LABELS[model]} | {completed[model]:,} | {totals[model]:,} |')
        if state.get('current'):
            t = state['current']; lines += ['', f"Current: {LABELS[t['model']]} / {t['dataset']} / {t['kind']} / {t['mode']}"]
        values = [Summary.Value(tag='Completion percent/Overall', simple_value=100 * state['completed'] / state['total'])]
        values += [Summary.Value(tag='Completion percent/' + LABELS[m], simple_value=100 * completed[m] / totals[m]) for m in totals]
        values += [Summary.Value(tag='Progress/text_summary', tensor=TensorProto(dtype=7,
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
            string_val=['\n'.join(lines).encode()]),
            metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name='text')))]
        self.writer.write(Event(wall_time=time.time(), step=state['completed'], summary=Summary(value=values)).SerializeToString())
        self.writer.flush()

    def close(self):
        self.writer.close()


def validate_completed(path, trial, plan, identities, cases):
    row = json.loads(path.read_text())
    if (row.get('status') != 'complete' or row.get('trial') != trial
            or row.get('plan_sha256') != plan['plan_sha256']
            or row.get('identity_sha256') != identities[trial['model']]
            or row.get('case_sha256') != digest(cases[trial['unit_id']])):
        raise ValueError(f'Completed trial does not match this frozen run: {path}')
    if trial['kind'] == 'verify':
        if row.get('verification', {}).get('status') != 'passed':
            raise ValueError(f'Failed parity verification: {path}')
    elif len(row.get('repetitions', [])) != trial['repetitions']:
        raise ValueError(f'Incomplete measurement repetitions: {path}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('all', 'positive', 'full_scan'), default='all')
    args = parser.parse_args()
    if args.condition == 'all':
        for index, condition in enumerate(('positive', 'full_scan'), 1):
            print(f'FINAL BENCHMARK: stage {index}/2 — {condition}', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--condition', condition], check=True)
        print('FINAL BENCHMARK COMPLETE. Both fixed stages finished.', flush=True)
        return
    package = Path('/campaign')
    for relative, expected in json.loads((package / 'package-files.json').read_text()).items():
        if file_hash(package / relative) != expected:
            raise ValueError(f'Package file changed: {relative}')
    plan_path = package / 'plans' / (args.condition + '.json')
    plan = read_plan(plan_path)
    if plan['engine_files'] != engine_files():
        raise ValueError('Frozen engine differs from installed code')
    env = environment()
    for name in ('numpy', 'Pillow', 'SimpleITK'):
        if env['packages'][name] != plan['preparation_environment']['packages'][name]:
            raise ValueError(f'Input library differs from accuracy preparation: {name}')
    spec_path = package / 'checkpoints.json'
    spec = json.loads(spec_path.read_text())
    bundle = Path('/bundle')
    print('Checking five model checkpoints and all selected input files...', flush=True)
    checkpoints = {m: checkpoint_for(m, spec, bundle) for m in plan['models']}
    checked_files = {}
    for case in plan['cases']:
        mapped = local_case(case, Path('/gamma-data'), Path('/data'))
        for key in ('image', 'mask'):
            path = mapped[key]; expected = case['file_sha256'][key]
            if path in checked_files:
                if checked_files[path] != expected:
                    raise ValueError(f'Conflicting input hash: {path}')
            else:
                if file_hash(path) != expected:
                    raise ValueError(f'Missing or changed input: {path}')
                checked_files[path] = expected
                if len(checked_files) % 100 == 0:
                    print(f'  Verified {len(checked_files):,} input files...', flush=True)
    print(f"Verified {len(checked_files):,} input files; {len(plan['cases'])} targets; "
          f"{len({c['dataset'] for c in plan['cases']})} datasets.", flush=True)
    repo = Path('/workspace')
    folder = repo / 'artifacts/gamma/performance-campaign' / (args.condition + '-' + plan['plan_sha256'][:12])
    folder.mkdir(parents=True, exist_ok=True)
    cases = {c['unit_id']: c for c in plan['cases']}
    jobs = plan['trials']
    totals = Counter(t['model'] for t in jobs)
    completed = Counter()
    with output_lock(repo / 'artifacts/gamma/performance-lock'):
        snapshot = source_snapshot(bundle)
        import csv
        gpu = telemetry()
        if len(gpu['gpus']) != 2 or gpu['compute_processes']:
            raise RuntimeError('Require two visible Gamma GPUs with no active compute processes')
        hardware = [[part.strip() for part in row[:4]] for row in csv.reader(gpu['gpus'])]
        identities = {}
        for m in plan['models']:
            identity = dict(plan_sha256=plan['plan_sha256'], checkpoint=checkpoints[m],
                source_sha256=snapshot['sha256'], environment=env, device='cuda:1',
                initialization_assets=initialization_assets(m, bundle), controls=CONTROLS,
                visible_gpu_identity=hardware, platform=platform.platform())
            identities[m] = digest(identity)
        campaign = dict(plan_sha256=plan['plan_sha256'], checkpoints_sha256=file_hash(spec_path),
            expected=[t['trial_id'] for t in jobs], pilot_unit=None, plan_path=str(plan_path),
            environment=env, scientific_protocol_certified=False, driver='resumable_gamma_v1')
        campaign_path = folder / 'campaign.json'
        if campaign_path.exists() and json.loads(campaign_path.read_text()) != campaign:
            raise ValueError('Campaign identity changed; keep old results and review before restarting')
        atomic_json(campaign_path, campaign)
        atomic_json(folder / 'plan.json', plan)
        done = set()
        for t in jobs:
            path = folder / 'trials' / (t['trial_id'] + '.json')
            if path.exists():
                validate_completed(path, t, plan, identities, cases)
                done.add(t['trial_id']); completed[t['model']] += 1
        overview = Overview(folder / 'tensorboard-overview')
        def progress(state, trial=None):
            current = dict(trial, dataset=cases[trial['unit_id']]['dataset']) if trial else None
            status = dict(condition=args.condition, state=state, completed=len(done), total=len(jobs),
                          current=current, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(folder / 'status.json', status)
            overview.update(status, totals, completed)
        try:
            progress('running')
            print(f"Starting/resuming: {len(done):,}/{len(jobs):,} jobs complete. GPU 1; models run serially.", flush=True)
            print(f'Results: {folder}', flush=True)
            for index, trial in enumerate(jobs):
                if trial['trial_id'] in done:
                    continue
                case = cases[trial['unit_id']]
                progress('running', trial)
                print(f"[{index + 1:,}/{len(jobs):,}] {LABELS[trial['model']]} | {case['dataset']} | "
                      f"{case_kind(case)}, {len(case['frame_ids'])} frames | {trial['mode']} | {trial['kind']} block {trial['block'] + 1}", flush=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                log = folder / 'logs' / (trial['trial_id'] + '-' + stamp + '.log')
                log.parent.mkdir(exist_ok=True)
                command = [sys.executable, '-m', 'rwkv_medsam2.evaluation.performance', '_worker',
                    '--plan', str(plan_path), '--trial', trial['trial_id'], '--checkpoints', str(spec_path),
                    '--bundle', '/bundle', '--data-root', '/gamma-data', '--device', 'cuda:1', '--output', str(folder)]
                with log.open('x') as stream:
                    code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
                if code:
                    failure = dict(trial=trial, returncode=code, log=str(log), completed=len(done))
                    atomic_json(folder / 'failure-history' / (stamp + '.json'), failure)
                    progress('stopped after error', trial)
                    print('\n'.join(log.read_text().splitlines()[-60:]), flush=True)
                    raise SystemExit(code)
                validate_completed(folder / 'trials' / (trial['trial_id'] + '.json'), trial, plan, identities, cases)
                done.add(trial['trial_id']); completed[trial['model']] += 1
            atomic_json(folder / 'complete.json', dict(plan_sha256=plan['plan_sha256'], completed=len(done)))
            progress('measurements complete')
        finally:
            overview.close()
    report = folder / 'report'
    if not (report / 'summary.json').exists():
        if report.exists():
            report.rename(folder / ('report-interrupted-' + str(time.time_ns())))
        subprocess.run([sys.executable, '-m', 'rwkv_medsam2.evaluation.performance', 'summarize',
                        '--runs', str(folder), '--output', str(report)], check=True)
    print(f'STAGE COMPLETE: {args.condition}. Report: {report}', flush=True)


if __name__ == '__main__':
    main()
