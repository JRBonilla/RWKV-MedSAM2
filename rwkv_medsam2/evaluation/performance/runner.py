"""Gamma performance CLI. Importing this module never launches GPU work."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

from ..shared.data import atomic_json, digest, file_hash
from .data import local_case, make_plan, read_plan

ROOT = Path(__file__).resolve().parents[3]


def telemetry():
    def query(columns, kind='gpu'):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={columns}',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=20, check=True)
        return result.stdout.strip().splitlines()
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        gpu_columns='index,uuid,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used',
        gpus=query('index,uuid,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used'),
        process_columns='gpu_uuid,pid,process_name,used_gpu_memory',
        compute_processes=query('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps'))


def rss():
    # Current resident bytes versus the process-lifetime high-water mark.
    resident = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    return dict(current_bytes=resident, lifetime_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def checkpoint_for(name, spec, bundle_root):
    from ..shared.models import checkpoint_identity
    spec = json.loads(json.dumps(spec))
    if bundle_root:
        manifest = json.loads((bundle_root / 'manifest.json').read_text())
        entry = manifest['checkpoints'].get(name)
        if entry is None or entry['checkpoint_sha256'] != spec['models'][name]['sha256']:
            raise ValueError(f'{name}: bundle is missing the explicitly selected checkpoint/hash')
        spec['models'][name]['checkpoint'] = str((bundle_root / entry['bundle_path']).resolve())
    return checkpoint_identity(name, spec)


def verification_path(output, trial):
    key = digest([trial['model'], trial['unit_id'], trial['mode']])
    return output / 'verification' / (key + '.json')


def worker(args):
    from ..accuracy.runner import engine_files, evaluate_case
    from ..shared.models import (configure_runtime, environment,
                                  initialization_assets, load_model, source_snapshot, CONTROLS)
    from .core import prepare_host, predict, prediction_hash, seed_case, parameter_counts
    plan = read_plan(args.plan)
    if plan['engine_files'] != engine_files():
        raise ValueError('Code differs from the frozen plan; regenerate from a matching accuracy manifest')
    matches = [t for t in plan['trials'] if t['trial_id'] == args.trial]
    if len(matches) != 1:
        raise ValueError('Unknown trial')
    trial = matches[0]
    frozen = next(c for c in plan['cases'] if c['unit_id'] == trial['unit_id'])
    case = local_case(frozen, args.data_root, args.source_root)
    target = args.output / 'trials' / (args.trial + '.json')
    if target.exists():
        raise FileExistsError(target)
    spec = json.loads(args.checkpoints.read_text())
    checkpoint = checkpoint_for(trial['model'], spec, args.bundle)
    assets = initialization_assets(trial['model'], args.bundle)
    env = environment()
    for package in ('numpy', 'Pillow', 'SimpleITK'):
        if env['packages'][package] != plan['preparation_environment']['packages'][package]:
            raise ValueError(f'{package} differs from accuracy input preparation')
    snapshot = source_snapshot(args.bundle)
    before = telemetry()
    if len(before['gpus']) < 2:
        raise RuntimeError('Expose both Gamma GPUs so competing compute can be checked on both cards')
    if before['compute_processes']:
        raise RuntimeError('GPU compute was already active before model loading; trial not measured')
    hardware = [[part.strip() for part in row[:4]] for row in csv.reader(before['gpus'])]
    identity = dict(plan_sha256=plan['plan_sha256'], checkpoint=checkpoint,
                    source_sha256=snapshot['sha256'], environment=env,
                    device=args.device, initialization_assets=assets, controls=CONTROLS,
                    visible_gpu_identity=hardware, platform=platform.platform())
    identity_sha = digest(identity)
    atomic_json(args.output / 'identities' / (identity_sha + '.json'), identity)
    atomic_json(args.output / 'sources' / (snapshot['sha256'] + '.json'), snapshot)
    torch = configure_runtime(args.device)
    if 'GTX 1070' not in torch.cuda.get_device_name() or torch.cuda.get_device_capability() != (6, 1):
        raise ValueError('This Gamma plan requires a GTX 1070; do not run it on the shared A100')
    model = load_model(trial['model'], checkpoint, args.device, bundle_root=args.bundle)
    settings = {k: getattr(model, k) for k in CONTROLS}
    settings.update({k: getattr(model, k, None) for k in
                    ('num_maskmem', 'max_cond_frames_in_attn', 'memory_temporal_stride_for_eval',
                     'directly_add_no_mem_embed', 'image_size')})
    settings['dynamic_multimask_via_stability'] = model.sam_mask_decoder.dynamic_multimask_via_stability
    model_parameters = parameter_counts(model)
    started = time.perf_counter()
    prepared = prepare_host(case, trial['mode'], plan['host_cache_bytes'])
    preparation_seconds = time.perf_counter() - started
    verify_path = verification_path(args.output, trial)
    expected_hash = None
    if trial['kind'] != 'verify':
        reference = json.loads(verify_path.read_text())
        if reference['identity_sha256'] != identity_sha or reference['status'] != 'passed':
            raise ValueError('Local parity verification belongs to a different run')
        expected_hash = reference['prediction_sha256']
    from torch.nn.attention import SDPBackend, sdpa_kernel
    repetitions, verification = [], None
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast(device_type='cuda', enabled=False))
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        if trial['kind'] == 'verify':
            # Accuracy checks/hashes/calibration run only in this untimed process.
            class DigestOnlyArchive:
                def frame(self, t, prediction, logits):
                    pass
            scores, _ = evaluate_case(model, case, trial['mode'], Path('.'), args.device,
                                      lambda *_: None, DigestOnlyArchive())
            seed_case(torch, case, trial['mode'])
            outputs = predict(model, prepared, args.device)
            actual = prediction_hash(outputs, prepared['order'])
            del outputs
            if actual != scores['prediction_sha256']:
                raise ValueError('Prepared performance path differs from accuracy predictions')
            verification = dict(status='passed', identity_sha256=identity_sha,
                prediction_sha256=actual, encoder_inputs_verified=scores['encoder_inputs_verified'],
                scope='same_machine_binary_output_parity_only', primary=scores['primary'],
                all_frame_assisted=scores['all_frame_assisted'])
            atomic_json(verify_path, verification)
        elif trial['kind'] == 'encoder':
            # One frame resident at a time. All selected frames contribute equally
            # within their case; source/dataset weighting happens in the report.
            per_frame = {str(t): [] for t in prepared['ids']}
            for t in prepared['ids']:
                image = prepared['images'][t].to(args.device)
                for _ in range(plan['warmups']):
                    features = model.forward_image(image)
                    del features
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(trial['repetitions']):
                    start.record()
                    features = model.forward_image(image)
                    end.record()
                    end.synchronize()
                    per_frame[str(t)].append(float(start.elapsed_time(end)))
                    del features
                del image
            repetitions = [dict(repetition=i, milliseconds_per_frame=sum(v[i] for v in per_frame.values()) / len(per_frame))
                           for i in range(trial['repetitions'])]
        else:
            for _ in range(plan['warmups']):
                seed_case(torch, case, trial['mode'])
                outputs = predict(model, prepared, args.device)
                del outputs
            torch.cuda.synchronize()
            gc.collect()
            if trial['kind'] == 'memory':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cpu_before = rss()
                baseline = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats()
            for i in range(trial['repetitions']):
                seed_case(torch, case, trial['mode'])
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = predict(model, prepared, args.device)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                # Validation occurs after timing, and repeats must match the
                # untimed accuracy-reference output on this same GPU/backend.
                if trial['kind'] == 'memory':
                    peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
                    cpu_after = rss()
                actual = prediction_hash(outputs, prepared['order'])
                if actual != expected_hash:
                    raise ValueError('Repeated output differs from local accuracy reference')
                del outputs
                if trial['kind'] == 'prediction':
                    n = len(prepared['ids'])
                    repetitions.append(dict(repetition=i, seconds=elapsed,
                        milliseconds_per_frame=elapsed * 1000 / n, frames_per_second=n / elapsed))
                else:
                    repetitions.append(dict(repetition=i, baseline_gpu_bytes=baseline, peak_gpu_bytes=peak,
                        delta_peak_gpu_bytes={k: peak[k] - baseline[k] for k in peak},
                        cpu_before=cpu_before, cpu_after=cpu_after,
                        cpu_peak_scope='process lifetime including load/preparation/warmups; not isolated prediction RSS'))
    after = telemetry()
    record = dict(status='complete', trial=trial, identity_sha256=identity_sha,
        plan_sha256=plan['plan_sha256'], case={k: frozen[k] for k in
            ('unit_id', 'dataset', 'task', 'source_id', 'case_id', 'target_id', 'dim', 'modality')},
        sequence_condition=case['sequence_condition'], case_sha256=digest(frozen),
        frame_ids=prepared['ids'], frames=len(prepared['ids']), supports=len(prepared['support']),
        queries=1 if case['dim'] == 2 else len(prepared['ids']) - len(prepared['support']),
        panel=plan['panel_labels'][case['unit_id']], repetitions=repetitions,
        verification=verification, parameters=model_parameters, effective_settings=settings,
        preparation_seconds_excluded=preparation_seconds, prepared_budget_bytes=prepared['prepared_budget_bytes'],
        output_representation='host boolean 512x512 masks for all processed slices',
        telemetry_before=before, telemetry_after=after,
        telemetry_scope='boundary snapshots only; transient interference is not automatically detected',
        platform=platform.platform(), scientific_protocol_certified=False)
    if trial['kind'] == 'encoder':
        record['per_frame_milliseconds'] = per_frame
    atomic_json(target, record)
    print(json.dumps(dict(trial=args.trial, status='complete')), flush=True)


def run(args):
    """Explicit future launch: serial fresh processes, no automatic resume/skip."""
    from ..accuracy.runner import engine_files, output_lock
    from ..shared.models import environment
    plan = read_plan(args.plan)
    if plan['engine_files'] != engine_files():
        raise ValueError('Performance plan source mismatch')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; preserve previous trials for review')
    # Device indices are meaningful only if both GPUs are visible in the container.
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Expose both Gamma GPUs; select one using --device')
    spec = json.loads(args.checkpoints.read_text())
    for name in plan['models']:
        checkpoint_for(name, spec, args.bundle)
    args.output.mkdir(parents=True)
    jobs = plan['trials']
    if args.pilot_unit:
        if args.pilot_unit not in {c['unit_id'] for c in plan['cases']}:
            raise ValueError('Pilot unit must be in the frozen plan')
        jobs = [t for t in jobs if t['unit_id'] == args.pilot_unit]
    atomic_json(args.output / 'campaign.json', dict(plan_sha256=plan['plan_sha256'],
        checkpoints_sha256=file_hash(args.checkpoints), expected=[t['trial_id'] for t in jobs],
        pilot_unit=args.pilot_unit, plan_path=str(args.plan), environment=environment(),
        scientific_protocol_certified=False))
    atomic_json(args.output / 'plan.json', plan)
    # Close/flush monitoring events before each worker: no background TensorBoard
    # writer is kept alive during a measurement subprocess.
    from torch.utils.tensorboard import SummaryWriter
    with output_lock(ROOT / 'artifacts/gamma/performance-lock'):
        for index, trial in enumerate(jobs):
            cmd = [sys.executable, '-m', 'rwkv_medsam2.evaluation.performance', '_worker',
                   '--plan', str(args.plan), '--trial', trial['trial_id'],
                   '--checkpoints', str(args.checkpoints), '--output', str(args.output),
                   '--data-root', str(args.data_root), '--source-root', str(args.source_root),
                   '--device', args.device]
            if args.bundle:
                cmd += ['--bundle', str(args.bundle)]
            print(json.dumps(dict(index=index + 1, total=len(jobs), **trial)), flush=True)
            log = args.output / 'logs' / (trial['trial_id'] + '.log')
            log.parent.mkdir(exist_ok=True)
            with log.open('x') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(args.output / 'failed.json', dict(trial=trial, returncode=result.returncode,
                                                           log=str(log), completed=index))
                raise RuntimeError(f'Trial failed; campaign stopped. See {log}')
            row = json.loads((args.output / 'trials' / (trial['trial_id'] + '.json')).read_text())
            writer = SummaryWriter(str(args.output / 'tensorboard'))
            try:
                writer.add_scalar('progress/completed_trials', index + 1, index + 1)
                writer.add_scalar('progress/fraction', (index + 1) / len(jobs), index + 1)
                prefix = f'{trial["model"]}/{row["sequence_condition"]}/{row["case"]["dim"]}d/{trial["mode"]}/{trial["kind"]}'
                for repeat in row['repetitions']:
                    step = trial['block'] * trial['repetitions'] + repeat['repetition']
                    case_prefix = f'{prefix}/{trial["unit_id"]}'
                    for metric in ('seconds', 'milliseconds_per_frame', 'frames_per_second'):
                        if metric in repeat:
                            writer.add_scalar(f'trials/{case_prefix}/{metric}', repeat[metric], step)
                    for scope in ('baseline_gpu_bytes', 'peak_gpu_bytes', 'delta_peak_gpu_bytes'):
                        for allocator, value in repeat.get(scope, {}).items():
                            writer.add_scalar(f'trials/{case_prefix}/{scope}/{allocator}', value, step)
            finally:
                writer.close()
    atomic_json(args.output / 'complete.json', dict(plan_sha256=plan['plan_sha256'], completed=len(jobs)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='CPU metadata selection only; requires a frozen accuracy manifest')
    plan.add_argument('--manifest', type=Path, required=True)
    plan.add_argument('--models', nargs='+', required=True)
    plan.add_argument('--data-root', type=Path, required=True)
    plan.add_argument('--source-root', type=Path, default=Path('/data'))
    plan.add_argument('--host-cache-gib', type=float, default=3.)
    plan.add_argument('--output', type=Path, required=True)
    plan.set_defaults(fn=make_plan)
    for name, fn in [('run', run), ('_worker', worker)]:
        command = sub.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--checkpoints', type=Path, required=True)
        command.add_argument('--data-root', type=Path, required=True)
        command.add_argument('--source-root', type=Path, default=Path('/data'))
        command.add_argument('--bundle', type=Path)
        command.add_argument('--device', default='cuda:1')
        command.add_argument('--output', type=Path, required=True)
        if name == 'run':
            command.add_argument('--pilot-unit', help='Run only one named panel target; results remain a pilot')
        else:
            command.add_argument('--trial', required=True)
        command.set_defaults(fn=fn)
    report = sub.add_parser('summarize')
    report.add_argument('--runs', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    report.add_argument('--bootstrap', type=int, default=2000)
    from .report import summarize
    report.set_defaults(fn=summarize)
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    try:
        args.fn(args)
    except Exception:
        if args.command == '_worker':
            atomic_json(args.output / 'failures' / (args.trial + '.json'),
                        dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
