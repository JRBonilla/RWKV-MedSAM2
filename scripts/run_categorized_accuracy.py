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
MODELS = ['corrected_hiera', 'rwkv_medsam2_final_nodistill', 'rwkv_medsam2_final_distill',
          'oxford_medical_sam2', 'uoft_medsam2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, help='Reuse an explicitly re-frozen prepared cohort')
    parser.add_argument('--checkpoints', type=Path, default=Path('/data/ResearchArtifacts/model_collection_20261006/evaluation-checkpoints_CURRENT.json'))
    args = parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cohort=args.cohort.resolve() if args.cohort else out/'cohort'
    (out/'logs').mkdir(exist_ok=True)
    atomic_json(out/'controller.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        checkpoints=str(args.checkpoints),models=MODELS))
    active={};completed=[];stage='initializing'
    def status(**extra):
        atomic_json(out/'status.json',dict(stage=stage,controller_pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),completed_jobs=completed,
            active=[dict(name=name,pid=p.pid) for name,(p,h) in active.items()],**extra))
    def start(name,cmd):
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
    try:
        stage='preparing_all_data';status()
        if not (cohort/'preparation-complete.json').exists():
            start('prepare',[sys.executable,'-B',str(ROOT/'scripts/prepare_categorized_accuracy.py'),
                 '--output',str(cohort),'--workers','8'])
            wait_all()
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
                start(name,run_cmd(root/'frozen',root/'runs',model,gpu));slots[gpu]=name
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
                    start(f'production.{model}.{condition}.shard{gpu}',run_cmd(
                        cohort/condition/'frozen',out/condition/'runs',model,gpu,gpu))
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
