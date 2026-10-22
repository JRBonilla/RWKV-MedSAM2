"""Durable all-data accuracy campaign: prepare, pilot, then four GPU shards/model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rwkv_medsam2.evaluation.shared.data import atomic_json
from rwkv_medsam2.evaluation.shared.data import digest, file_hash
from rwkv_medsam2.evaluation.accuracy.runner import engine_files, read_frozen
from rwkv_medsam2.evaluation.accuracy.surface_workers import analysis_identity
from rwkv_medsam2.evaluation.shared.models import (
    CONTROLS, checkpoint_identity, environment, initialization_assets, source_snapshot,
)
MODELS = ['corrected_hiera', 'rwkv_medsam2_final_nodistill', 'rwkv_medsam2_final_distill',
          'oxford_medical_sam2', 'uoft_medsam2']


def verified_completion(folder, manifest, model, checkpoint, env, snapshot, shards, shard,
                        frozen=None, assets=None):
    """Reuse a completion marker only when the full planned job identity matches."""
    if not (folder/'complete.json').exists():
        return False
    meta, cases = frozen if frozen is not None else read_frozen(manifest)
    job = json.loads((folder/'job.json').read_text())
    identity = dict(
        profile=meta['profile'], manifest_sha256=file_hash(manifest/'manifest.json'),
        cases_sha256=meta['cases_sha256'], protocol_sha256=meta['protocol_sha256'],
        model=model, checkpoint=checkpoint,
        initialization_assets=initialization_assets(model) if assets is None else assets,
        source_sha256=snapshot['sha256'], environment=env, controls=CONTROLS,
        dynamic_multimask_via_stability=False, dtype='float32', attention_backend='math',
        frame_batch=1, state_policy='all_frame_GPU', seed=42,
        sequence_condition=meta['sequence_condition'], recording_schema=3, save_logits=False,
    )
    expected = [c['unit_id']+':'+mode for c in cases
                if c['eligible'] and int(digest([c['dataset'], c['case_id']]), 16)%shards == shard
                for mode in c['modes']]
    planned = dict(run_id=digest(identity), identity=identity, shard=shard,
                   shards=shards, expected=expected)
    complete = json.loads((folder/'complete.json').read_text())
    if job != planned or any(complete.get(key) != value for key, value in dict(
        run_id=planned['run_id'], completed=len(expected), expected=len(expected),
        surfaces_completed=len(expected), surfaces_requested=True,
    ).items()):
        raise ValueError(f'Completed job identity or coverage changed: {folder}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, help='Reuse an explicitly re-frozen prepared cohort')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse identity-verified completed jobs; resume unfinished shards')
    parser.add_argument('--checkpoints', type=Path, default=Path('/data/ResearchArtifacts/model_collection_20261006/evaluation-checkpoints_CURRENT.json'))
    args = parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cohort=args.cohort.resolve() if args.cohort else out/'cohort'
    (out/'logs').mkdir(exist_ok=True)
    atomic_json(out/'controller.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        checkpoints=str(args.checkpoints),models=MODELS))
    active={};completed=[];stage='initializing'
    frozen_engine = snapshot = env = checkpoints = None
    frozen_cohorts = {}; model_assets = {}
    def status(**extra):
        atomic_json(out/'status.json',dict(stage=stage,controller_pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),completed_jobs=completed,
            active=[dict(name=name,pid=p.pid) for name,(p,h) in active.items()],**extra))
    def start(name,cmd):
        if frozen_engine is not None:
            if engine_files() != frozen_engine or source_snapshot() != snapshot:
                raise ValueError('Campaign source changed; restore the frozen source before resuming')
        log=(out/'logs'/f'{name}.log').open('a')
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        active[name]=(p,log);status();return p
    def reap():
        for name,(p,h) in list(active.items()):
            code=p.poll()
            if code is None:continue
            h.close();del active[name]
            if code:raise RuntimeError(f'{name} failed with exit {code}; see logs/{name}.log')
            completed.append(name);status()
    def wait_all():
        while active:
            reap();time.sleep(2)
    def run_cmd(manifest,output,model,gpu,shard=None):
        cmd=[sys.executable,'-B','-m','rwkv_medsam2.evaluation.accuracy','run',
             '--manifest',str(manifest),'--output',str(output),'--checkpoints',str(args.checkpoints),
             '--model',model,'--device',f'cuda:{gpu}','--surface-workers','1',
             '--input-cache-gib','2','--surface-cache-gib','.5']
        if shard is not None:cmd+=['--shards','4','--shard',str(shard)]
        return cmd
    def run_job(name, manifest, output, model, gpu, shard=None):
        shards = 1 if shard is None else 4
        index = 0 if shard is None else shard
        folder = output/model/f'shard-{index:04d}-of-{shards:04d}'
        if args.resume and verified_completion(folder, manifest, model,
                                               checkpoints[model], env, snapshot, shards, index,
                                               frozen_cohorts[manifest], model_assets[model]):
            completed.append(name);status()
            print(f'Reusing verified completed job: {name}', flush=True)
            return
        start(name, run_cmd(manifest, output, model, gpu, shard))
    try:
        stage='preparing_all_data';status()
        if not (cohort/'preparation-complete.json').exists():
            start('prepare',[sys.executable,'-B',str(ROOT/'scripts/prepare_categorized_accuracy.py'),
                 '--output',str(cohort),'--workers','8'])
            wait_all()
        stage='validating_frozen_sources';status()
        for condition in ('positive', 'full_scan'):
            for manifest in (cohort/condition/'frozen', cohort/'pilot'/condition/'frozen'):
                frozen_cohorts[manifest] = read_frozen(manifest)
        frozen_engine = engine_files()
        snapshot = source_snapshot()
        env = environment()
        analysis_identity()
        for meta, cases in frozen_cohorts.values():
            for package in ('numpy', 'Pillow', 'SimpleITK'):
                if env['packages'][package] != meta['preparation_environment']['packages'][package]:
                    raise ValueError(f'{package} differs from frozen input preparation')
        spec = json.loads(args.checkpoints.read_text())
        checkpoints = {model: checkpoint_identity(model, spec) for model in MODELS}
        model_assets = {model: initialization_assets(model) for model in MODELS}
        stage='pilot';status()
        queue=[(condition,model) for condition in ('positive','full_scan') for model in MODELS]
        slots={}
        while queue or active:
            reap()
            for gpu in range(4):
                if gpu in slots and slots[gpu] not in active:del slots[gpu]
                if gpu in slots or not queue:continue
                condition,model=queue.pop(0);name=f'pilot.{condition}.{model}'
                root=cohort/'pilot'/condition
                run_job(name,root/'frozen',root/'runs',model,gpu);slots[gpu]=name
            if active:time.sleep(2)
        stage='validating_pilot_reports';status()
        for condition in ('positive','full_scan'):
            root=cohort/'pilot'/condition
            report=root/'report'
            if not report.exists():
                start('pilot-report.'+condition,[sys.executable,'-B','-m','rwkv_medsam2.evaluation.accuracy',
                    'summarize','--manifest',str(root/'frozen'),'--runs',str(root/'runs'),
                    '--models',*MODELS,'--output',str(report),'--bootstrap','100'])
                wait_all()
            summary=json.loads((report/'summary.json').read_text())
            if not all(summary['coverage'][m]['full_cohort_complete'] for m in MODELS):
                raise ValueError('Pilot model coverage incomplete')
            categorized=json.loads((report/'categorized-summary.json').read_text())
            if not categorized['views']:raise ValueError('Missing pilot category reports')
        atomic_json(out/'pilot-passed.json',dict(status='passed',models=MODELS,conditions=['positive','full_scan']))
        # Model inference is identical across categories. Each target is predicted
        # once per planned condition, never once per reporting group.
        stage='production';status()
        for model in MODELS:
            for condition in ('positive','full_scan'):
                for gpu in range(4):
                    run_job(f'production.{model}.{condition}.shard{gpu}',
                        cohort/condition/'frozen',out/condition/'runs',model,gpu,gpu)
                wait_all()
        stage='final_reports';status()
        for condition in ('positive','full_scan'):
            start('final-report.'+condition,[sys.executable,'-B','-m','rwkv_medsam2.evaluation.accuracy',
                'summarize','--manifest',str(cohort/condition/'frozen'),
                '--runs',str(out/condition/'runs'),'--models',*MODELS,
                '--output',str(out/condition/'report'),'--bootstrap','2000'])
        wait_all()
        stage='complete';status()
    except BaseException:
        stage='failed';status(traceback=traceback.format_exc())
        raise
    finally:
        for p,h in active.values():
            if p.poll() is None:p.terminate()
        for p,h in active.values():p.wait();h.close()


if __name__=='__main__':main()
