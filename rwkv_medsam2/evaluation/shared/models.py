"""Load one explicitly pinned model per process; never select latest checkpoints."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys

from .data import digest, file_hash

ROOT = Path(__file__).resolve().parents[3]
CONTROLS = {
    'use_mask_input_as_output_without_sam': True,
    'binarize_mask_from_pts_for_mem_enc': False,
    'fill_hole_area': 0,
    'non_overlap_masks': False,
    'non_overlap_masks_for_mem_enc': False,
    'multimask_output_in_sam': False,
    'multimask_output_for_tracking': False,
}
FORBIDDEN_HIERA = '564e65c142e10c94e527e9d97b669cc22322b4fb6b5ef607ddcff3fb2a515079'
MODELS = ('rwkv_medsam2_final_nodistill', 'rwkv_medsam2_final_distill',
          'oxford_medical_sam2', 'uoft_medsam2', 'corrected_hiera')


def environment():
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'SimpleITK', 'tensorboard',
                 'hydra-core', 'omegaconf', 'timm', 'ninja', 'pandas', 'scipy', 'surface-distance'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'packages': packages}


def source_snapshot(bundle_root=None):
    """Hash source content, including dirty files, rather than relying on git HEAD."""
    files = {}
    roots = [ROOT / 'rwkv_medsam2', ROOT / 'ext', ROOT / 'dripp']
    roots += ([Path(bundle_root) / 'external/oxford', Path(bundle_root) / 'external/uoft']
              if bundle_root else [Path('/data/jrbonill/oxford_medsam2_env/Medical-SAM2'),
                                   Path('/data/jrbonill/medsam2_env/MedSAM2')])
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.cu', '.cpp', '.h', '.cuh'):
                if not any(part in ('.git', '__pycache__', '.venv', 'node_modules') for part in path.parts):
                    files[str(path)] = file_hash(path)
    helper = ROOT / 'notebooks/strict_video_protocol_lib.py'
    files[str(helper)] = file_hash(helper)
    return {'sha256': digest(files), 'files': files}


def checkpoint_identity(name, spec):
    if name not in MODELS:
        raise ValueError(f'Unknown model: {name}')
    item = spec['models'][name]
    if not item.get('checkpoint') or not item.get('sha256'):
        raise ValueError(f'{name}: final checkpoint and SHA256 must be supplied explicitly')
    actual = file_hash(item['checkpoint'])
    if actual != item['sha256']:
        raise ValueError(f'{name}: checkpoint hash mismatch')
    if name == 'corrected_hiera' and actual == FORBIDDEN_HIERA:
        raise ValueError('Historical Hiera checkpoint is excluded from this study')
    return {'path': str(Path(item['checkpoint']).resolve()), 'sha256': actual}


def initialization_assets(name, bundle_root=None):
    paths = []
    if name.startswith('rwkv_') or name == 'corrected_hiera':
        paths = ['/data/TrainingCheckpoints/pretrained/sam2.1_hiera_tiny.pt']
    elif name == 'oxford_medical_sam2':
        paths = ['/data/jrbonill/oxford_medsam2_env/Medical-SAM2/checkpoints/sam2_hiera_tiny.pt']
    if bundle_root:
        paths = [str(Path(bundle_root) / 'checkpoints' / Path(path).name) for path in paths]
    return {path: file_hash(path) for path in paths}


def configure_runtime(device):
    # Set before importing the model libraries or their CUDA extensions.
    os.environ['VRWKV_DISABLE_COMPILE'] = '1'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    os.environ['VRWKV_T_MAX'] = '256'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import random
    import numpy as np
    import torch
    if not device.startswith('cuda:') or not torch.cuda.is_available():
        raise ValueError('Select an available CUDA device explicitly, e.g. cuda:0')
    torch.cuda.set_device(torch.device(device))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def load_model(name, checkpoint, device, bundle_root=None):
    """Reuse construction only, not the legacy benchmark's loaders or metrics."""
    import torch
    sys.path[:0] = [str(ROOT), str(ROOT / 'ext'), str(ROOT / 'dripp')]
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location('accuracy_model_helper',
                                                ROOT / 'notebooks/strict_video_protocol_lib.py')
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    helper.DEVICE = torch.device(device)
    if bundle_root:
        bundle_root = Path(bundle_root).resolve()
        helper.OXFORD_REPO = str(bundle_root / 'external/oxford')
        helper.OXFORD_SITE = ''
        helper.OXFORD_SAM_CKPT = str(bundle_root / 'checkpoints/sam2_hiera_tiny.pt')
        helper.UOFT_MEDSAM2_BASE = str(bundle_root / 'external/uoft')
        helper.UOFT_MEDSAM2_SITE = ''
    def read_config(path):
        cfg = helper.OmegaConf.load(path)
        helper.OmegaConf.resolve(cfg)
        return cfg
    helper.load_config = read_config
    if name == 'corrected_hiera':
        helper.MODEL_REGISTRY[name] = dataclasses.replace(
            helper.MODEL_REGISTRY['sam2_1_final_base'], name=name)
    entry = helper.MODEL_REGISTRY[name]
    entry.checkpoint_path = checkpoint['path']
    entry.checkpoint_dir = ''
    # Hard-disable every automatic checkpoint fallback, including helper defaults.
    helper.pick_checkpoint = lambda _: checkpoint['path']
    if name == 'oxford_medical_sam2':
        helper.OXFORD_MED_PRETRAIN = checkpoint['path']
    if name == 'uoft_medsam2':
        helper.UOFT_MEDSAM2_CKPT = checkpoint['path']
    original_config = helper.load_rwkv_config
    def accuracy_config(item):
        cfg = original_config(item)
        cfg.init.student_ckpt_path = '/data/TrainingCheckpoints/pretrained/sam2.1_hiera_tiny.pt'
        if bundle_root:
            cfg.init.student_ckpt_path = str(bundle_root / 'checkpoints/sam2.1_hiera_tiny.pt')
        cfg.model.use_mask_input_as_output_without_sam = True
        cfg.optimizations.fuse_rwkv_shift_mix = False
        cfg.optimizations.compile_image_encoder = False
        cfg.optimizations.image_feature_chunk_size = 1
        cfg.optimizations.prediction_pin_cpu_features = False
        if item.builder == 'rwkv2_mlp2_optimized':
            cfg.model.image_encoder.trunk.fuse_shift_mix = False
        return cfg
    helper.load_rwkv_config = accuracy_config
    bundle = helper.load_model_bundle(name)
    model = getattr(bundle['predictor'], 'model', bundle['predictor']).float().eval()
    # Strictly reload final weights, including external builders that load loosely.
    state = helper.unwrap_state_dict(torch.load(checkpoint['path'], map_location='cpu', weights_only=False))
    if name == 'oxford_medical_sam2':
        state = {str(k).removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    if file_hash(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Checkpoint changed during loading')
    for key, value in CONTROLS.items():
        if not hasattr(model, key):
            raise ValueError(f'Missing protocol control: {key}')
        setattr(model, key, value)
    if not hasattr(model.sam_mask_decoder, 'dynamic_multimask_via_stability'):
        raise ValueError('Decoder lacks the required stability control')
    model.sam_mask_decoder.dynamic_multimask_via_stability = False
    if model.num_maskmem < 1:
        raise ValueError('Model temporal memory is disabled')
    if any(m.training or hasattr(m, '_orig_mod') for m in model.modules()):
        raise ValueError('Training or compiled model found')
    for value in list(model.parameters()) + list(model.buffers()):
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Nonfinite or non-FP32 model tensor')
    return model
