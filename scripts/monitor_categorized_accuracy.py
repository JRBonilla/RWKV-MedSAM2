"""Readable TensorBoard campaign status, independent of model inference."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from torch.utils.tensorboard import SummaryWriter

MODELS = {
    'corrected_hiera': 'Hiera — corrected baseline',
    'rwkv_medsam2_final_nodistill': 'RWKV — non-distilled',
    'rwkv_medsam2_final_distill': 'RWKV — distilled',
    'oxford_medical_sam2': 'Oxford MedSAM2',
    'uoft_medsam2': 'UofT MedSAM2',
}
STAGES = {'preparing_all_data':'Preparing test data', 'pilot':'Checking models on the small pilot',
          'validating_frozen_sources':'Checking frozen evaluation source and checkpoints',
          'validating_pilot_reports':'Checking pilot results and reports',
          'production':'Full accuracy evaluation', 'final_reports':'Building final reports',
          'complete':'Complete', 'failed':'Stopped — needs attention'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def model_progress(root, cohort, model, condition, stage, status):
    pilot = stage in ('pilot', 'validating_pilot_reports')
    runs = (cohort/'pilot'/condition if pilot else root/condition)/'runs'/model
    jobs = list(runs.glob('shard-*/job.json'))
    total = done = 0
    complete = 0
    for job in jobs:
        record=read(job)
        total += len(record['expected'])
        result=read(job.parent/'complete.json')
        progress=read(job.parent/'surfaces/progress.json')
        if result:
            done += result['completed'];complete += 1
        elif progress:
            done += progress['completed']
    if total:
        if complete == len(jobs):return f'Complete — {done:,}/{total:,}'
        return f'{done:,}/{total:,} scored ({100*done/total:.1f}%)'
    prefix=f'pilot.{condition}.{model}' if pilot else f'production.{model}.{condition}.'
    active=[r for r in status['active'] if r['name'].startswith(prefix)]
    if active:
        waiting=False
        for r in active:
            path=Path(f"/proc/{r['pid']}/wchan")
            if path.exists() and 'rpc_' in path.read_text():waiting=True
        return 'Waiting on network storage' if waiting else 'Starting — loading/checking model'
    return 'Queued'


def render(root,cohort,status):
    stage=status['stage'];prep=read(cohort/'preparation-complete.json')
    pilot=read(root/'pilot-passed.json')
    rows=[f'**{STAGES.get(stage,stage)}**', '',
          '| Step | Status |','|---|---|',
          f'| Prepare all 56 datasets | {"Complete" if prep else "Running"} |',
          f'| Five-model pilot | {"Passed" if pilot else "In progress" if stage in ("pilot","validating_pilot_reports") else "Waiting"} |',
          f'| Full evaluation | {"Running" if stage == "production" else "Complete" if stage in ("final_reports","complete") else "Not started"} |','']
    if prep:
        rows.append(f"**Test targets:** {prep['counts']['eligible']:,}. All 56 datasets included.")
    if stage not in ('preparing_all_data',):
        rows += ['', '**Pilot progress**' if stage in ('pilot','validating_pilot_reports') else '**Full evaluation progress**', '',
                 '| Model | 2D + positive-frame sequences | Full stored scans / videos |', '|---|---|---|']
        for model,label in MODELS.items():
            a=model_progress(root,cohort,model,'positive',stage,status)
            b=model_progress(root,cohort,model,'full_scan',stage,status)
            rows.append(f'| {label} | {a} | {b} |')
    rows += ['', 'Step = completed evaluation conditions across the full campaign. Each condition is one target and prompt mode. '
             'Scored counts include completed boundary metrics; GPU predictions can be slightly ahead.', '',
             '**This is the overview.** The table shows all five models together. Scalars shows completion percentages. '
             'Detailed accuracy curves and individual GPU logs are still recorded separately.', '',
             'Main, overlap-category and all-data results use the same predictions. Final CSV/JSON reports '
             'will separate the categories and show summaries with and without PAIP.', '',
             'Last checked: '+datetime.now(ZoneInfo('America/Toronto')).strftime('%b %d, %I:%M:%S %p %Z')]
    if stage=='failed':
        rows += ['', '**Error:** '+status.get('traceback','See controller log').strip().splitlines()[-1]]
    return '\n'.join(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cohort',type=Path)
    args=p.parse_args();root=args.output;cohort=args.cohort or root/'cohort'
    writer=SummaryWriter(str(root/'tensorboard-overview'),flush_secs=15)
    try:
        while True:
            status=read(root/'status.json')
            marker=read(cohort/'preparation-complete.json')
            progress=read(cohort/'preparation-progress.json')
            total_expected = sum(read(cohort/c/'frozen/manifest.json')['counts']['conditions'] for c in ('positive','full_scan')) if marker else 0
            completed_by_model = {}
            for model, label in MODELS.items():
                done = 0
                for condition in ('positive','full_scan'):
                    for shard in (root/condition/'runs'/model).glob('shard-*'):
                        result = read(shard/'complete.json') or read(shard/'surfaces/progress.json')
                        if result:
                            done += result['completed']
                completed_by_model[label] = done
            total_completed = sum(completed_by_model.values())
            step = total_completed
            if marker:
                writer.add_scalar('preparation/fraction',1.,step)
            elif progress:
                writer.add_scalar('preparation/fraction',progress['images']/progress['expected_images'],step)
            writer.add_text('Progress',render(root,cohort,status),step)
            if total_expected:
                for label, done in completed_by_model.items():
                    writer.add_scalar('Completion percent/' + label, 100*done/total_expected, step)
                writer.add_scalar('Completion percent/All five models', 100*total_completed/(len(MODELS)*total_expected), step)
            production_done=sum(n.startswith('production.') for n in status['completed_jobs'])
            writer.add_scalar('campaign/completed_GPU_jobs_out_of_40',production_done,step)
            writer.flush()
            if status['stage'] in ('complete','failed'):break
            time.sleep(15)
    finally:writer.close()

if __name__=='__main__':main()
