"""Freeze all audited test targets without editing historical metadata or training caches."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rwkv_medsam2.evaluation.shared.data import digest, file_hash, atomic_json, freeze_case, condition_case, case_kind
from rwkv_medsam2.evaluation.accuracy.cohorts import AuditedCohort, evaluation_tasks


def audited_source(ds, g, cvc, thy):
    raw=g['images'][0]['path']; ident=str(g['identifier'])
    if ds=='CVC-ClinicDB':return 'sequence:'+cvc[Path(raw).stem]
    if ds=='ThyroidUltra':return 'annotation:'+thy[int(re.search(r'image_(\d+)\.png',raw)[1])]['annot_id']
    if ds=='Mosmeddata':return re.sub(r'_\d+$','',Path(raw).stem)
    if ds=='CHASEDB1':return re.search(r'Image_(\d+)[LR]',raw)[1]
    if ds=='FetReg':return re.search(r'Video\d+',raw)[0]
    if ds=='ROSE-2':return ident.split('_')[0]
    if ds=='ISLES-2022':return ident.split('_ses-')[0]
    if ds=='MSD':return Path(raw).name.split('.nii')[0] # remove derived split/modality identifiers
    if ds in ('CHAOS','QUBIQ2021','WBC','WMH'):return g['subdataset_name']+':'+ident
    if ds in ('ACDC_LUNGHP','CoNSeP','PROMISE12','ROSE-1'):
        return None # IDs reused across official subsets; no authoritative patient mapping here
    return ident


def freeze_image(rows):
    import numpy as np
    from PIL import Image
    import SimpleITK as sitk
    from rwkv_medsam2.evaluation.accuracy.inputs import InputCache
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    cache = dict(inputs=InputCache(1024**3), hashes={}, encoder_hashes={})
    result = []
    for row in rows:
        if row['dim'] == 2:
            with Image.open(row['mask']) as im:
                labels = np.unique(np.asarray(im))
        else:
            labels = np.unique(sitk.GetArrayFromImage(sitk.ReadImage(row['mask'])))
        positive = labels[labels > 0]
        if len(positive) != 1 or float(positive[0]) != int(positive[0]):
            raise ValueError(f"Expected exactly one positive component label: {row['mask']}: {positive}")
        row['label'] = int(positive[0])
        frozen = freeze_case(row, Path('/data'),
            'encoder_replacement_positive_primary_fullscan_robustness_v3_draft', freeze_cache=cache)
        result.append((row, frozen))
    return result


def reviewed_inputs(inventory, audit_dir):
    audit = AuditedCohort(inventory, audit_dir)
    cvc = {r['frame_id']: r['sequence_id'] for r in csv.DictReader(open('/data/research/CVC-ClinicDB/metadata.csv'))}
    thy = list(csv.DictReader(open('/data/research/ThyroidUltra/thyroidultrasoundcineclip/extracted_ThyroidUltra/full/metadata_full.csv')))
    glas = {r['name'].strip(): r['patient ID'].strip() for r in csv.DictReader(
        open('/data/research/GlaS/Warwick_QU_Dataset/Grade.csv', newline=''), skipinitialspace=True)}
    known_identity = {'CHASEDB1', 'GlaS', 'Mosmeddata', 'ThyroidUltra', 'CVC-ClinicDB',
                      'FetReg', 'CAMUS', 'PAIP2019', 'Neonatal', 'ISLES-2022', 'ROSE-2'}
    by_ds = defaultdict(list)
    for index, m in audit.inventory.items():
        by_ds[m['dataset']].append((index, m))
    all_rows = []
    for ds, records in sorted(by_ds.items()):
        original = json.loads(Path(f'/data/Preprocessed_v2.1/{ds}/{ds}_groups.json').read_text())
        lookup = {}
        wanted = defaultdict(set)
        for _, metadata in records:
            im, ma = metadata['sequence'][0]
            wanted[ma].add(im)
        for sub in original['subdatasets']:
            for g in sub['test']:
                group_images = set(g.get('proc_images', []))
                for mask in g.get('proc_masks', []):
                    for image_path in wanted.get(mask['path'], set()) & group_images:
                        lookup[(image_path, mask['path'])] = g
        for index, m in records:
            if len(m['sequence']) != 1:
                raise ValueError('Multi-file record requires explicit review')
            image, mask = m['sequence'][0]
            g = lookup[(image, mask)]
            source = ('patient:' + glas[Path(g['images'][0]['path']).stem]
                      if ds == 'GlaS' else audited_source(ds, g, cvc, thy))
            source = source or (m.get('subdataset', '') + ':' + str(m.get('source_identifier') or g['identifier']))
            tasks, provenance = evaluation_tasks(m)
            for task in tasks:
                row = dict(case_id=digest([ds, image])[:24], target_id=digest(mask)[:24],
                    dataset=ds, task=task, source_id=str(source),
                    source_identity_status='verified' if ds in known_identity else 'unresolved',
                    source_note='Local source grouping audit; identity means patient/scan/video/slide as available. '
                                'Unresolved identifiers are not certified patients. External pretraining is unknown.',
                    split='test', provenance_reviewed=True, image=image, mask=mask, dim=m['dim'],
                    modality=m['modality'], label=None, target_scope='filtered_component', target_name=m['class'],
                    preprocessing_note='Existing DRIPP filtered component; stored cropped/resized image and mask. '
                                       'Full-scan robustness means full STORED scan, not uncropped original acquisition.',
                    legacy_index=index, legacy_identifier=m.get('identifier'), spacing_verified_mm=False,
                    stored_preprocessing=g.get('preprocessing_metadata', {}))
                if m['dim'] == 3:
                    if g.get('subdataset_pipeline') != '3D':
                        raise ValueError('Stored 3D record has an unreviewed pipeline')
                    row.update(axis_zyx=0, reverse_slices=False, anatomical_volume=ds != 'CAMUS',
                               geometry_note='DRIPP stored ZYX axis 0, forward stored order; crop and XY resize preserve slice order. '
                                             'Physical millimetre spacing not certified after legacy preprocessing.')
                    if ds == 'CAMUS':
                        if not all('_half_sequence.nii.gz' in im['path'] for im in g['images']):
                            raise ValueError('Unreviewed CAMUS temporal source')
                        row.update(sequence_kind='video', temporal_axis_note='Stored axis 0 is forward CAMUS half-sequence time.')
                all_rows.append(audit.annotate(row))
        print(json.dumps(dict(reviewed_dataset=ds, targets=len(records))), flush=True)
    # Unify reviewed source groups and cross-dataset scan aliases transitively.
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]
    for r in all_rows:
        source = digest([r['dataset'], r['source_id']])
        alias = r.get('global_source_id', source)
        a, b = sorted((root(source), root(alias)))
        parent[b] = a
    for r in all_rows:
        r['global_source_id'] = 'global:' + root(digest([r['dataset'], r['source_id']]))
    return all_rows, audit


def freeze_population(args):
    from rwkv_medsam2.evaluation.accuracy.runner import engine_files, PROFILE, SEED, derive_full_scan
    from rwkv_medsam2.evaluation.shared.models import environment
    from types import SimpleNamespace
    args.output.mkdir(parents=True, exist_ok=False)
    rows, audit = reviewed_inputs(args.inventory, args.audit_dir)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['image']].append(row)
    folder = args.output/'positive/frozen'
    folder.mkdir(parents=True)
    protocol_path = ROOT/'docs/evaluation/shared-protocol.json'
    from rwkv_medsam2.evaluation.shared.data import validate_protocol
    validate_protocol(json.loads(protocol_path.read_text()))
    (folder/'protocol.json').write_bytes(protocol_path.read_bytes())
    counts, seen, coverage, category_counts = Counter(), set(), Counter(), Counter()
    started = time.monotonic()
    pilot = {}
    reviewed_path = args.output/'reviewed-cases.jsonl'
    with reviewed_path.open('w') as reviewed, (folder/'cases.jsonl').open('w') as frozen_file:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for n, batch in enumerate(pool.imap(freeze_image, grouped.values(), chunksize=1)):
                for row, c in batch:
                    if c['unit_id'] in seen:
                        raise ValueError('Duplicate frozen unit')
                    seen.add(c['unit_id'])
                    reviewed.write(json.dumps(row, sort_keys=True)+'\n')
                    frozen_file.write(json.dumps(c, sort_keys=True)+'\n')
                    counts['records'] += 1
                    counts['eligible' if c['eligible'] else 'excluded'] += 1
                    coverage[c['dataset']] += 1
                    category_counts[c['reporting']['training_overlap']] += 1
                    if c['eligible']:
                        counts['conditions'] += len(c['modes'])
                        counts['prediction_frames_all_conditions'] += len(c['frame_ids'])*len(c['modes'])
                        # Fixed smallest-frame representative per category, modality kind,
                        # and repaired task; selection does not use model scores.
                        keys = [('category', c['reporting']['training_overlap']), ('kind', case_kind(c))]
                        if 'repair' in c['task_label_provenance']:
                            keys.append(('repaired', c['dataset']))
                        for key in keys:
                            if key not in pilot or (len(c['frame_ids']), c['unit_id']) < (len(pilot[key]['frame_ids']), pilot[key]['unit_id']):
                                pilot[key] = c
                if n % 100 == 0:
                    atomic_json(args.output/'preparation-progress.json', dict(images=n+1, expected_images=len(grouped),
                        counts=dict(counts), elapsed_seconds=time.monotonic()-started))
                    print(json.dumps(dict(frozen_images=n+1, expected_images=len(grouped), records=counts['records'])), flush=True)
    if len(coverage) != 56 or len(rows) != counts['records']:
        raise ValueError('Full dataset coverage failed')
    meta = dict(schema=3, profile=PROFILE, seed=SEED, sequence_condition='positive_frames',
        support_policy='quarter_positive', prompt_seed=42, scope='primary_all_data_categorized',
        storage_per_model_uncompressed=dict(packed_masks_bytes=counts['prediction_frames_all_conditions']*512*512//8,
            optional_fp32_logits_bytes=counts['prediction_frames_all_conditions']*512*512*4),
        protocol_sha256=file_hash(folder/'protocol.json'), cases_sha256=file_hash(folder/'cases.jsonl'),
        reviewed_cases_sha256=file_hash(reviewed_path), engine_files=engine_files(), preparation_environment=environment(),
        counts=dict(counts), metric_grid='canonical_512x512_per_slice',
        box_coordinates='inclusive min/max pixel indices; SAM point labels 2 and 3',
        support_mask_encoding='numpy.packbits C order, zlib, base64; 512x512', scientific_protocol_certified=False,
        preparation_script_sha256=file_hash(Path(__file__)),
        audit_sha256=audit.audit_id, dataset_coverage=dict(coverage), category_counts=dict(category_counts))
    atomic_json(folder/'manifest.json', meta)
    atomic_json(folder/'preparation-errors.json', [])
    derive_full_scan(SimpleNamespace(manifest=folder, output=args.output/'full_scan/frozen'))
    pilot_cases = sorted({c['unit_id']: c for c in pilot.values()}.values(), key=lambda c: c['unit_id'])
    pf = args.output/'pilot/positive/frozen';pf.mkdir(parents=True)
    (pf/'protocol.json').write_bytes((folder/'protocol.json').read_bytes())
    (pf/'cases.jsonl').write_text(''.join(json.dumps(c, sort_keys=True)+'\n' for c in pilot_cases))
    pc = Counter(records=len(pilot_cases), eligible=len(pilot_cases), excluded=0,
        conditions=sum(len(c['modes']) for c in pilot_cases),
        prediction_frames_all_conditions=sum(len(c['frame_ids'])*len(c['modes']) for c in pilot_cases))
    pm = dict(meta, scope='operational_pilot_only', cases_sha256=file_hash(pf/'cases.jsonl'), counts=dict(pc))
    atomic_json(pf/'manifest.json', pm)
    derive_full_scan(SimpleNamespace(manifest=pf, output=args.output/'pilot/full_scan/frozen'))
    atomic_json(args.output/'preparation-complete.json', dict(counts=dict(counts), datasets=dict(coverage),
        categories=dict(category_counts), pilot_targets=len(pilot_cases), seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--inventory', type=Path, default=ROOT/'artifacts/gamma-test-inventory/legacy_test_metadata.jsonl')
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'reports/leakage_extent_20261007')
    freeze_population(parser.parse_args())
