"""Prepared host inputs and uninstrumented shared prediction computation."""
import hashlib
import random

from ..shared.data import (array_hash, canonical_frame, canonical_mask, decode_mask,
                            digest, file_hash, read_case, sequence_plan)
from ..shared.prediction import predict_frame


def seed_case(torch, case, mode):
    import numpy as np
    seed = int(digest([42, case['unit_id'], mode])[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_host(case, mode, max_bytes):
    import numpy as np
    import torch
    ids, mapping, order = sequence_plan(case)
    support = set(case['support_frames'])
    # Includes prepared FP32 images, prompt masks, and returned boolean masks.
    required = len(ids) * 512 * 512 * (3 * 4 + 1) + len(support) * 512 * 512 * 4
    if required > max_bytes:
        raise ValueError(f'Prepared input/output budget exceeded: {required} > {max_bytes}; '
                         'do not trim this case or silently change the timing interval')
    for key in ('image', 'mask'):
        if file_hash(case[key]) != case['file_sha256'][key]:
            raise ValueError(f'Changed {key} file')
    raw, masks, transform = read_case(case, '.')
    if transform != case['transform']:
        raise ValueError('Frozen preprocessing changed')
    images, prompts = {}, {}
    for t in ids:
        frame = canonical_frame(raw[t], transform['intensity_limits'])
        target = canonical_mask(masks[t])
        if array_hash(frame) != case['encoder_input_sha256'][t] or array_hash(target) != case['target_sha256'][t]:
            raise ValueError('Frozen image/target tensor changed')
        images[t] = torch.from_numpy(frame)
        if t in support:
            payload = case['prompts'][str(t)]
            mask = decode_mask(payload['mask_packbits_zlib_base64'])
            if array_hash(mask) != payload['mask_sha256'] or not np.array_equal(mask, target):
                raise ValueError('Support payload changed')
            prompts[t] = ({'point_coords': torch.tensor([payload['box_xy']], dtype=torch.float32),
                           'point_labels': torch.tensor([[2, 3]], dtype=torch.int32)}, None) if mode == 'box' else (
                               None, torch.from_numpy(mask.astype(np.float32))[None, None])
    return dict(images=images, prompts=prompts, ids=ids, mapping=mapping, order=order,
                support=support, prepared_budget_bytes=required)


def predict(model, prepared, device):
    """Host tensors -> host boolean masks, with fresh GPU state on every call.

    No hashing, metric computation, serialization, telemetry or logging here.
    The caller enters inference/precision contexts and synchronizes the interval.
    """
    memory = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
    outputs = {}
    try:
        for t in prepared['order']:
            points, supplied = prepared['prompts'].get(t, (None, None))
            points = {k: v.to(device) for k, v in points.items()} if points is not None else None
            supplied = supplied.to(device) if supplied is not None else None
            out = predict_frame(model, prepared['images'][t].to(device), points, supplied,
                memory, prepared['mapping'][t], t in prepared['support'], len(prepared['ids']))
            outputs[t] = (out['pred_masks_high_res'][0, 0] >= 0).cpu().numpy()
            memory['cond_frame_outputs' if t in prepared['support'] else 'non_cond_frame_outputs'][prepared['mapping'][t]] = out
            del out, points, supplied
        return outputs
    finally:
        memory.clear()


def prediction_hash(outputs, order):
    import numpy as np
    h = hashlib.sha256()
    for t in order:
        h.update(t.to_bytes(8, 'little'))
        h.update(np.packbits(outputs[t]).tobytes())
    return h.hexdigest()


def parameter_counts(model):
    def count(modules):
        params = {id(p): p for m in modules for p in m.parameters()}
        return sum(p.numel() for p in params.values())
    encoder = [model.image_encoder]
    names = ['image_encoder']
    if model.use_high_res_features_in_sam:
        encoder += [model.sam_mask_decoder.conv_s0, model.sam_mask_decoder.conv_s1]
        names += ['sam_mask_decoder.conv_s0', 'sam_mask_decoder.conv_s1']
    trunk = getattr(model.image_encoder, 'trunk', None)
    return dict(total=count([model]), image_encoder_module=count([model.image_encoder]),
                complete_forward_image=count(encoder), complete_encoder_modules=names,
                backbone=count([trunk]) if trunk is not None else None,
                flops=None, macs=None,
                arithmetic_status='not_measured; custom WKV arithmetic accounting still required')
