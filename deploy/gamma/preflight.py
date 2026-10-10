"""Check Gamma's libraries and real CUDA operations; not an accuracy validation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    report = {'python': platform.python_version(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'arch_list': torch.cuda.get_arch_list(),
              'protocol_certified': False, 'gpus': []}
    assert torch.__version__.startswith('2.7.1+cu118'), report
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    assert os.environ.get('TORCH_CUDA_ARCH_LIST') == '6.1', 'Source deploy/gamma/env.sh first'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in ['torchvision', 'numpy', 'scipy', 'pandas', 'cv2', 'SimpleITK',
                 'hydra', 'iopath', 'decord', 'cupy', 'rwkv_medsam2.train_sam2']:
        importlib.import_module(name)
    assert os.environ.get('VRWKV_DISABLE_COMPILE') == '1', 'Source deploy/gamma/env.sh first'
    import vrwkv.vrwkv as vrwkv
    assert vrwkv._COMPILED_SPATIAL_SHIFT_MIX is vrwkv._spatial_shift_mix
    assert vrwkv._COMPILED_CHANNEL_SHIFT_MIX is vrwkv._channel_shift_mix
    report['rwkv_shift_mix_backend'] = 'eager'
    RUN_CUDA = vrwkv.RUN_CUDA
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device), torch.inference_mode():
            props = torch.cuda.get_device_properties(device)
            assert (props.major, props.minor) == (6, 1), f'Expected Pascal sm_61: {props}'
            x = torch.ones((1, 3, 32, 32), device='cuda', dtype=torch.float32)
            conv = torch.nn.Conv2d(3, 4, 3).cuda().eval()
            assert torch.isfinite(conv(x)).all()
            q = torch.randn((1, 2, 8, 32), device='cuda')
            with sdpa_kernel(SDPBackend.MATH):
                assert torch.isfinite(F.scaled_dot_product_attention(q, q, q)).all()
            w = torch.zeros(32, device='cuda')
            k = torch.zeros((1, 8, 32), device='cuda')
            v = torch.ones_like(k)
            out = RUN_CUDA(1, 8, 32, w, w, k, v)
            assert torch.isfinite(out).all()
            assert torch.allclose(out, torch.ones_like(out), atol=1e-5)
            import cupy as cp
            with cp.cuda.Device(device):
                assert float(cp.arange(4, dtype=cp.float32).sum()) == 6.0
            torch.cuda.synchronize()
            report['gpus'].append({'index': device, 'name': props.name,
                                   'memory_bytes': props.total_memory,
                                   'capability': [props.major, props.minor],
                                   'fp32_conv_attention_wkv': 'passed'})
    report['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    report['status'] = 'environment_smoke_passed_not_model_or_protocol_validation'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
