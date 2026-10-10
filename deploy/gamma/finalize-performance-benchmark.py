"""Freeze a single broad primary benchmark plus a fixed full-scan stress panel."""
from collections import defaultdict
from pathlib import Path
import copy
import json
import argparse
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rwkv_medsam2.evaluation.shared.data import digest, atomic_json, case_kind
from rwkv_medsam2.evaluation.performance.data import read_plan



def schedule(plan):
    plan = copy.deepcopy(plan)
    jobs = []
    conditions = [(c, mode) for c in plan['cases'] for mode in c['modes']]
    for index, (case, mode) in enumerate(conditions):
        offset = index % len(plan['models'])
        order = plan['models'][offset:] + plan['models'][:offset]
        def add(model, kind, block, repetitions):
            trial = dict(unit_id=case['unit_id'], mode=mode, model=model, kind=kind,
                         block=block, repetitions=repetitions)
            trial['trial_id'] = digest(trial)[:24]
            jobs.append(trial)
        for model in order:
            add(model, 'verify', 0, 1)
        for block in range(2):
            for model in order if block == 0 else list(reversed(order)):
                add(model, 'prediction', block, 3)
        if mode == case['modes'][0]:
            for block in range(2):
                for model in order if block == 0 else list(reversed(order)):
                    add(model, 'encoder', block, 3)
        for model in order:
            add(model, 'memory', 0, 1)
    plan['trials'] = jobs
    plan['warmups'] = 3
    plan['fixed_measurement_schedule'] = dict(timing_blocks=2, timing_repeats_per_block=3,
        warmups_per_worker=3, memory_workers_per_case_model_prompt=1,
        rationale='Spend more of the fixed budget on distinct inputs. All three fresh-process allocated-peak measurements agreed exactly within each model/prompt in the 2D, 3D and video pilots. Retain two opposite-order timing blocks and three warmups.')
    plan['status'] = 'final_fixed_benchmark_small_systems_pilots_passed'
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = digest(plan)
    return plan


def full_scan_stress(parent):
    plan = copy.deepcopy(parent)
    groups = defaultdict(list)
    for case in parent['cases']:
        groups[(case['modality'], case_kind(case))].append(case)
    selected, strata = {}, []
    for group, cases in sorted(groups.items()):
        # One deterministic target per exact image; prefer no dataset or outcome.
        distinct = {}
        for case in sorted(cases, key=lambda c: digest([42, c['unit_id']])):
            distinct.setdefault(case['file_sha256']['image'], case)
        pool = sorted(distinct.values(), key=lambda c: (len(c['frame_ids']), digest([42, c['unit_id']])))
        middle = statistics.median(len(c['frame_ids']) for c in pool)
        short = pool[0]
        long = pool[-1]
        used = {short['file_sha256']['image'], long['file_sha256']['image']}
        candidates = [c for c in pool if c['file_sha256']['image'] not in used]
        medium = min(candidates, key=lambda c: (abs(len(c['frame_ids']) - middle), digest([42, c['unit_id']])))
        for label, case in [('short', short), ('middle', medium), ('long', long)]:
            selected[case['unit_id']] = case
            strata.append(dict(modality=group[0], sequence_kind=group[1], length_position=label,
                               unit_id=case['unit_id'], dataset=case['dataset'], frames=len(case['frame_ids'])))
    plan['cases'] = sorted(selected.values(), key=lambda c: (c['modality'], case_kind(c), len(c['frame_ids'])))
    plan['panel_labels'] = {row['unit_id']:dict(main=False,stress=True,length_bin=row['length_position']) for row in strata}
    plan['coverage'] = []
    plan['fixed_stress_selection'] = strata
    plan['parent_full_scan_panel_sha256'] = parent['plan_sha256']
    plan['included_unresolved_sources'] = sum(c['source_identity_status'] != 'verified' for c in plan['cases'])
    plan['selection'] = 'Fixed short/middle/long full stored inputs per modality/sequence kind from the staged parent panel; stress results only, not all-dataset full-scan averages. Metadata seed 42; no model outcomes used.'
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-plan', type=Path, required=True)
    parser.add_argument('--full-scan-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    primary = read_plan(args.primary_plan)
    primary['parent_large_panel_sha256'] = primary['plan_sha256']
    primary['selection'] += '; final fixed primary panel; no adaptive expansion'
    primary = schedule(primary)
    secondary = schedule(full_scan_stress(read_plan(args.full_scan_plan)))
    for name, plan in [('positive', primary), ('full_scan', secondary)]:
        atomic_json(args.output / (name + '.json'), plan)
        print(f"{name}: {len(plan['cases'])} targets, {len(plan['trials'])} jobs")


if __name__ == '__main__':
    main()
