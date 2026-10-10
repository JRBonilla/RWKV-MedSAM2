"""CPU regression checks for RWKV's Pascal import policy; no CUDA build or kernels."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = r'''
import getpass
import os
from unittest.mock import patch
import torch
import torch.utils.cpp_extension

disabled = os.environ['VRWKV_DISABLE_COMPILE'] == '1'
def compile_spy(fn, **kwargs):
    if disabled:
        raise AssertionError('Pascal import called torch.compile')
    assert kwargs == {'fullgraph': True, 'dynamic': False, 'options': {'triton.cudagraphs': False}}
    return fn

with patch('torch.utils.cpp_extension.load') as extension, \
     patch('torch.compile', side_effect=compile_spy) as compile_mock, \
     patch('getpass.getuser', side_effect=KeyError('getpwuid(): uid not found: 31092')):
    import vrwkv.vrwkv as model
    assert extension.call_count == 1  # The real WKV CUDA extension is still required.
    assert compile_mock.call_count == (0 if disabled else 2)
    if disabled:
        assert model._COMPILED_SPATIAL_SHIFT_MIX is model._spatial_shift_mix
        assert model._COMPILED_CHANNEL_SHIFT_MIX is model._channel_shift_mix
'''


class CompilePolicyTest(unittest.TestCase):
    def probe(self, disabled):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', VRWKV_DISABLE_COMPILE=str(disabled),
                   PYTHONPATH=str(ROOT / 'ext'))
        subprocess.run([sys.executable, '-c', PROBE], env=env, cwd=ROOT, check=True)

    def test_pascal_import_never_calls_compile_with_unknown_uid(self):
        self.probe(1)

    def test_default_preserves_existing_compile_configuration(self):
        self.probe(0)


if __name__ == '__main__':
    unittest.main()
