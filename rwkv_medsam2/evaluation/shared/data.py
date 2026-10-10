"""CPU-only input freezing and metric primitives for the accuracy study."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import zlib

PROFILE = 'independent2d-positive-primary-fullscan-robustness-v3'
RESOLUTION = 512
SEED = 42


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def jsonl(path):
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSON') from exc


def unit_id(case):
    return digest([case[k] for k in ('dataset', 'task', 'case_id', 'target_id')])


def case_kind(case):
    """Reporting category; dim describes the stored array, not time versus depth."""
    return 'video' if case.get('sequence_kind') == 'video' else f"{case['dim']}d"


def validate_case(c):
    required = ('case_id', 'target_id', 'dataset', 'task', 'source_id', 'modality',
                'image', 'mask', 'preprocessing_note', 'source_note')
    for key in required:
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f'Missing nonempty {key}')
    if c.get('split') != 'test' or c.get('provenance_reviewed') is not True:
        raise ValueError('Require reviewed held-out test provenance')
    if c.get('source_identity_status') not in ('verified', 'unresolved'):
        raise ValueError('Declare verified or unresolved source identity')
    if c.get('target_scope') not in ('semantic', 'instance', 'filtered_component'):
        raise ValueError('Declare target meaning; filtered components are not complete semantic labels')
    if type(c.get('label')) is not int or c['label'] <= 0:
        raise ValueError('Explicit positive integer mask label required')
    if c.get('spacing_verified_mm') is True and not c.get('spacing_provenance_note'):
        raise ValueError('Verified physical units require spacing_provenance_note')
    if c.get('dim') not in (2, 3):
        raise ValueError('Expected a 2D image or a stored 3D array')
    if c.get('sequence_kind') not in (None, 'video'):
        raise ValueError('Unknown sequence kind')
    if case_kind(c) == 'video' and c['dim'] != 3:
        raise ValueError('Video requires a stored frame stack')
    if c['dim'] == 3:
        if c.get('axis_zyx') not in (0, 1, 2) or type(c.get('reverse_slices')) is not bool:
            raise ValueError('Explicit slice axis in SimpleITK ZYX array and direction required')
        if case_kind(c) == 'video':
            if (c.get('anatomical_volume') is not False or c['reverse_slices']
                    or not c.get('temporal_axis_note') or not c.get('geometry_note')):
                raise ValueError('Video requires explicit temporal-axis provenance, forward order, and anatomical_volume=false')
        elif not c.get('geometry_note') or c.get('anatomical_volume') is not True:
            raise ValueError('Reviewed anatomical slice order required; declare video explicitly for temporal stacks')


def read_case(c, data_root):
    """Return raw ordered frames and target masks, without foreground trimming."""
    import numpy as np
    from PIL import Image
    paths = [Path(c[k]) if Path(c[k]).is_absolute() else Path(data_root) / c[k]
             for k in ('image', 'mask')]
    if c['dim'] == 2:
        with Image.open(paths[0]) as im:
            if im.mode not in ('L', 'RGB'):
                raise ValueError(f'2D input must be declared byte L/RGB, found {im.mode}')
            image = np.asarray(im).copy()
        with Image.open(paths[1]) as im:
            mask = np.asarray(im).copy()
        if image.dtype != np.uint8 or mask.ndim != 2 or image.shape[:2] != mask.shape:
            raise ValueError('Unexpected 2D dtype, mask channels, or image/mask grid')
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        frames, masks = image[None], mask[None]
        geometry = {'original_hw': list(mask.shape), 'units': 'processed_pixels'}
        limits = None
    else:
        import SimpleITK as sitk
        im, ma = [sitk.ReadImage(str(p)) for p in paths]
        if im.GetDimension() != 3 or ma.GetDimension() != 3:
            raise ValueError('Expected scalar 3D images')
        if im.GetNumberOfComponentsPerPixel() != 1 or ma.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError('Vector volumes need explicitly reviewed preprocessing')
        for accessor in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
            if not np.allclose(getattr(im, accessor)(), getattr(ma, accessor)(), rtol=0, atol=1e-5):
                raise ValueError(f'Image/mask geometry mismatch: {accessor}')
        frames, masks = [np.moveaxis(sitk.GetArrayFromImage(v), c['axis_zyx'], 0) for v in (im, ma)]
        if c['reverse_slices']:
            frames, masks = frames[::-1], masks[::-1]
        if not np.isfinite(frames).all():
            raise ValueError('Nonfinite volume intensities')
        limits = [float(x) for x in np.percentile(frames, [0.5, 99.5])]
        geometry = {name: list(getattr(im, accessor)()) for name, accessor in
                    [('size_xyz', 'GetSize'), ('spacing_xyz', 'GetSpacing'),
                     ('origin_xyz', 'GetOrigin'), ('direction', 'GetDirection')]}
        geometry.update(axis_zyx=c['axis_zyx'], reverse_slices=c['reverse_slices'])
    if not np.isfinite(masks).all() or not np.equal(masks, np.floor(masks)).all():
        raise ValueError('Mask labels must be finite integers')
    return frames, masks == c['label'], {'geometry': geometry, 'intensity_limits': limits,
                                        'frames': len(frames)}


def canonical_frame(raw, limits):
    """Fixed PIL float bilinear resampling, then exactly one ImageNet transform."""
    import numpy as np
    from PIL import Image
    x = raw.astype(np.float32)
    if limits is None:
        x /= np.float32(255)
    else:
        lo, hi = limits
        x = np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1)
        x = np.repeat(x[..., None], 3, axis=-1)
    if x.shape[:2] != (RESOLUTION, RESOLUTION):
        x = np.stack([np.asarray(Image.fromarray(x[..., k]).resize(
            (RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR)) for k in range(3)], axis=-1)
    mean = np.asarray([.485, .456, .406], dtype=np.float32)
    std = np.asarray([.229, .224, .225], dtype=np.float32)
    x = np.ascontiguousarray(((x - mean) / std).transpose(2, 0, 1)[None], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite canonical input')
    return x


def canonical_mask(raw):
    import numpy as np
    from PIL import Image
    if raw.shape != (RESOLUTION, RESOLUTION):
        raw = np.asarray(Image.fromarray(raw.astype(np.uint8)).resize(
            (RESOLUTION, RESOLUTION), Image.Resampling.NEAREST))
    return np.ascontiguousarray(raw, dtype=bool)


def support_indices(positive, protocol_id, case_id, seed=SEED):
    if len(positive) < 2:
        return []
    k = min(len(positive) - 1, max(1, math.ceil(.25 * len(positive))))
    ranked = sorted(positive, key=lambda t: (hashlib.sha256(
        f'{protocol_id}|{case_id}|{seed}|{t}'.encode()).hexdigest(), t))
    return sorted(ranked[:k])


def encode_mask(mask):
    import numpy as np
    return base64.b64encode(zlib.compress(np.packbits(mask.reshape(-1)).tobytes())).decode('ascii')


def decode_mask(payload):
    import numpy as np
    packed = np.frombuffer(zlib.decompress(base64.b64decode(payload)), dtype=np.uint8)
    if len(packed) != RESOLUTION * RESOLUTION // 8:
        raise ValueError('Invalid saved support mask length')
    return np.unpackbits(packed).reshape(RESOLUTION, RESOLUTION).astype(bool)


def freeze_case(c, data_root, protocol_id, support_policy='quarter_positive', prompt_seed=SEED, freeze_cache=None):
    import numpy as np
    validate_case(c)
    if 'reporting' in c:
        from ..accuracy.cohorts import validate_reporting
        validate_reporting(c)
    if support_policy not in ('quarter_positive', 'single_middle_positive'):
        raise ValueError('Unknown support policy')
    c = dict(c, unit_id=unit_id(c), support_policy=support_policy, prompt_seed=prompt_seed)
    paths = {key: Path(c[key]) if Path(c[key]).is_absolute() else Path(data_root) / c[key]
             for key in ('image', 'mask')}
    if freeze_cache is None:
        c['file_sha256'] = {key: file_hash(path) for key, path in paths.items()}
        frames, masks, transform = read_case(c, data_root)
        cached_hashes = None
    else:
        from ..accuracy.inputs import signature
        def cached_hash(path):
            stamp = (str(path), signature(path))
            if stamp not in freeze_cache['hashes']:
                freeze_cache['hashes'][stamp] = file_hash(path)
            return freeze_cache['hashes'][stamp]
        c['file_sha256'] = {key: cached_hash(path) for key, path in paths.items()}
        frames, masks, transform = freeze_cache['inputs'].read_case(c, data_root)
        image_key = digest([c['file_sha256']['image'], c['dim'], transform])
        cached_hashes = freeze_cache['encoder_hashes'].get(image_key)
    positives, frame_hashes, mask_hashes = [], [], []
    for t in range(len(frames)):
        target = canonical_mask(masks[t])
        if target.any():
            positives.append(t)
        frame_hashes.append(cached_hashes[t] if cached_hashes is not None else
                            array_hash(canonical_frame(frames[t], transform['intensity_limits'])))
        mask_hashes.append(array_hash(target))
    if freeze_cache is not None:
        freeze_cache['encoder_hashes'][image_key] = frame_hashes
    if not positives or (c['dim'] == 3 and len(positives) < 2):
        return dict(c, eligible=False, exclusion_reason='no_positive_target' if not positives
                    else 'fewer_than_two_positive_slices', transform=transform)
    if c['dim'] == 2:
        support = [0]
    elif support_policy == 'single_middle_positive':
        midpoint = (positives[0] + positives[-1]) / 2
        support = [min(positives, key=lambda t: (abs(t - midpoint), t))]
    else:
        support = support_indices(positives, protocol_id, c['unit_id'], prompt_seed)
    prompts = {}
    for t in support:
        target = canonical_mask(masks[t])
        yy, xx = np.where(target)
        prompts[str(t)] = {'box_xy': [[int(xx.min()), int(yy.min())], [int(xx.max()), int(yy.max())]],
                           'mask_packbits_zlib_base64': encode_mask(target),
                           'mask_sha256': array_hash(target)}
    frozen = dict(c, eligible=True, transform=transform, positive_frames=len(positives),
                  positive_frame_ids=positives, support_frames=support, prompts=prompts,
                  encoder_input_sha256=frame_hashes, target_sha256=mask_hashes,
                  modes=['box'] if c['dim'] == 2 else ['box', 'mask'])
    frozen['shared_case_sha256'] = digest(frozen)
    return condition_case(frozen, 'positive_frames')


def confusion(prediction, target):
    import numpy as np
    p, g = np.asarray(prediction, bool), np.asarray(target, bool)
    if p.shape != g.shape:
        raise ValueError('Prediction and target shapes differ')
    return {'tp': int(np.count_nonzero(p & g)), 'fp': int(np.count_nonzero(p & ~g)),
            'fn': int(np.count_nonzero(~p & g)), 'tn': int(np.count_nonzero(~p & ~g))}


def overlaps(counts):
    tp, fp, fn = [counts[k] for k in ('tp', 'fp', 'fn')]
    return {'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.}


def add_counts(rows):
    rows = list(rows)
    return {key: sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn', 'tn')}



def condition_case(case, condition):
    """Change inference input membership only; prompts and canonical inputs stay frozen."""
    if condition not in ('positive_frames', 'full_scan'):
        raise ValueError('Unknown sequence condition')
    out = dict(case)
    if not out['eligible']:
        return out
    if out['dim'] == 2:
        out.update(sequence_condition='independent_2d', frame_ids=[0])
        if condition == 'full_scan':
            out.update(eligible=False, exclusion_reason='independent_2d_reported_in_primary_only')
    else:
        ids = list(out['positive_frame_ids']) if condition == 'positive_frames' else list(range(out['transform']['frames']))
        out.update(sequence_condition=condition, frame_ids=ids)
        if not set(out['support_frames']).issubset(ids):
            raise ValueError('Condition dropped a frozen support slice')
    return out


def sequence_plan(case):
    """Stored slice IDs remain auditable; SAM memory uses dense selected-sequence IDs."""
    ids = case['frame_ids']
    if ids != sorted(set(ids)) or not ids:
        raise ValueError('Frame IDs must be unique, ordered and nonempty')
    n = case['transform']['frames']
    if ids[0] < 0 or ids[-1] >= n:
        raise ValueError('Stored slice ID out of bounds')
    condition = case['sequence_condition']
    if (case['dim'] == 2) != (condition == 'independent_2d'):
        raise ValueError('Condition dimension mismatch')
    expected = ([0] if case['dim'] == 2 else case['positive_frame_ids']
                if condition == 'positive_frames' else list(range(n)))
    if condition not in ('independent_2d', 'positive_frames', 'full_scan') or ids != expected:
        raise ValueError('Frame membership differs from the declared condition')
    support = set(case['support_frames'])
    if not support.issubset(ids):
        raise ValueError('Missing frozen support')
    mapping = {stored: index for index, stored in enumerate(ids)}
    order = sorted(support) + [t for t in ids if t not in support]
    return ids, mapping, order

def validate_result(row, case, mode, run_id):
    """Reconstruct overlap scores from complete per-frame counts on resume/merge."""
    if case_kind(row['case']) != case_kind(case):
        raise ValueError('Result sequence kind differs from frozen case')
    if (row['run_id'] != run_id or row['unit_key'] != case['unit_id'] + ':' + mode
            or row['case_sha256'] != digest(case) or row['status'] != 'complete'
            or row['mode'] != mode or row['encoder_inputs_verified'] is not True):
        raise ValueError('Result identity or input verification mismatch')
    metadata_keys = ('unit_id', 'case_id', 'target_id', 'dataset', 'task', 'source_id',
                     'source_identity_status', 'target_scope', 'modality', 'dim')
    expected_metadata = {key: case[key] for key in metadata_keys}
    for key in ('reporting', 'global_source_id', 'task_label_provenance'):
        if key in case:
            expected_metadata[key] = case[key]
    if case_kind(case) == 'video':
        expected_metadata['sequence_kind'] = 'video'
    if row['case'] != expected_metadata:
        raise ValueError('Result hierarchy/target metadata differs from manifest')
    ids, mapping, _ = sequence_plan(case)
    n = len(ids)
    if (row['sequence_condition'] != case['sequence_condition'] or row['frame_ids'] != ids
            or row['stored_frames'] != case['transform']['frames']
            or row['shared_case_sha256'] != case['shared_case_sha256']):
        raise ValueError('Result inference condition or stored slice mapping differs')
    frames = row['per_frame']
    support = set(case['support_frames'])
    if len(frames) != n or [r['frame'] for r in frames] != ids:
        raise ValueError('Result has missing, repeated, or unordered frames')
    if row['frames'] != n or row['support_frames'] != sorted(support):
        raise ValueError('Result frame/support coverage differs')
    for r in frames:
        if r['sequence_index'] != mapping[r['frame']]:
            raise ValueError('Dense sequence index differs from stored slice ID mapping')
        if r['support'] != (r['frame'] in support):
            raise ValueError('Result support flag differs')
        counts = [r[k] for k in ('tp', 'fp', 'fn', 'tn')]
        if any(type(v) is not int or v < 0 for v in counts) or sum(counts) != RESOLUTION ** 2:
            raise ValueError('Invalid pixel confusion counts')
    query = frames if case['dim'] == 2 else [r for r in frames if not r['support']]
    if row['query_frames'] != len(query):
        raise ValueError('Result query coverage differs')
    positive_ids = set(case['positive_frame_ids'])
    positive_queries = [r for r in query if r['frame'] in positive_ids]
    for name, values in [('primary', query), ('all_frame_assisted', frames), ('positive_query_overlap', positive_queries)]:
        counts = add_counts(values)
        if row[name] != {**counts, **overlaps(counts)}:
            raise ValueError(f'{name} scores do not match per-frame confusion counts')


def validate_protocol(protocol):
    """Reject changed knobs that this fixed reference implementation does not implement."""
    expected = {
        'shared_input.resolution_hw': [512, 512],
        'shared_input.channels': 3,
        'shared_input.case_batch_size': 1,
        'shared_input.encoder_frame_batch_size': 1,
        'shared_input.2d_scaling': 'declared byte-valued inputs / 255',
        'shared_input.3d_scaling': 'one volume-wide 0.5th/99.5th percentile mapping, clipped to [0,1]',
        'tasks.independent_2d.primary_prompt_modes': ['box'],
        'tasks.independent_2d.cross_image_memory': False,
        'tasks.positive_3d.role': 'primary',
        'tasks.positive_3d.remove_empty_before_inference': True,
        'tasks.full_3d.role': 'secondary_robustness',
        'tasks.full_3d.prompt_modes': ['box', 'mask'],
        'tasks.full_3d.foreground_trimming': False,
        'tasks.full_3d.reverse_duplicate_slices': False,
        'tasks.full_3d.retain_empty_slices': True,
        'tasks.full_3d.support_count': 'min(P-1, max(1, ceil(0.25*P))) where P >= 2 target-positive slices',
        'tasks.full_3d.primary_seed': 42,
        'inference.dtype': 'float32',
        'inference.memory_dtype': 'float32',
        'inference.autocast': False,
        'inference.tf32': False,
        'inference.torch_compile': False,
        'inference.attention_backend': 'math',
        'inference.cpu_threads': 2,
        'metrics.probability_threshold': .5,
        'metrics.threshold_operator': '>=',
        'metrics.both_empty_overlap': 1,
        'metrics.one_empty_overlap': 0,
        'metrics.primary_smoothing': 0,
    }
    controls = {'use_mask_input_as_output_without_sam': True,
                'dynamic_multimask_via_stability': False,
                'binarize_mask_from_pts_for_mem_enc': False, 'fill_hole_area': 0,
                'non_overlap_masks': False, 'non_overlap_masks_for_mem_enc': False,
                'multimask_output_in_sam': False, 'multimask_output_for_tracking': False}
    expected.update({'inference.controls.' + key: value for key, value in controls.items()})
    for path, value in expected.items():
        actual = protocol
        for key in path.split('.'):
            actual = actual[key]
        if actual != value:
            raise ValueError(f'Unsupported protocol change at {path}: update and validate the implementation first')
