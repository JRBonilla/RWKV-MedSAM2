import builtins
import logging
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import dripp.config as config
import dripp.helpers as helpers
from dripp.preprocessor import Preprocessor


def fake_cupy(device_count=1):
    module = types.ModuleType("cupy")
    module.float32 = np.float32
    module.integer = np.integer
    module.zeros = np.zeros
    module.bincount = np.bincount
    module.nonzero = np.nonzero
    module.issubdtype = np.issubdtype
    module.asnumpy = np.asarray
    module.get_array_module = lambda _array: module
    module.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            getDeviceCount=lambda: device_count,
            getDeviceProperties=lambda _index: {"name": b"Fake CUDA Device"},
        ),
        Device=lambda _index: types.SimpleNamespace(use=lambda: None),
        Stream=types.SimpleNamespace(
            null=types.SimpleNamespace(synchronize=lambda: None)
        ),
    )
    return module


class FakeCupyArray:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype

    def ravel(self):
        return self.values.ravel()


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.original_gpu_enabled = config.GPU_ENABLED
        self.logger = logging.getLogger(f"{__name__}.{self._testMethodName}")
        self.logger.handlers = [logging.NullHandler()]
        self.logger.propagate = False

    def tearDown(self):
        config.GPU_ENABLED = self.original_gpu_enabled

    def _preprocessor(self):
        return Preprocessor(
            target_size=(2, 2), min_mask_size=1,
            dataset_logger=self.logger, dataset_name="backend-test",
        )

    def test_cpu_startup_does_not_import_cupy(self):
        config.GPU_ENABLED = False
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "cupy":
                raise AssertionError("CPU startup attempted to import CuPy")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            preprocessor = self._preprocessor()
        self.assertIs(preprocessor.xp, np)

    def test_gpu_selection_happens_after_runtime_flag_changes(self):
        module = fake_cupy()
        config.GPU_ENABLED = True
        with patch.dict(sys.modules, {"cupy": module}):
            preprocessor = self._preprocessor()
        self.assertIs(preprocessor.xp, module)

    def test_gpu_request_without_cupy_has_clear_error(self):
        config.GPU_ENABLED = True
        original_import = builtins.__import__

        def missing_cupy(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_cupy):
            with self.assertRaisesRegex(RuntimeError, "CuPy is not installed"):
                self._preprocessor()

    def test_gpu_request_without_cuda_has_clear_error(self):
        module = fake_cupy(device_count=0)
        config.GPU_ENABLED = True
        with patch.dict(sys.modules, {"cupy": module}):
            with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
                self._preprocessor()

    def test_helper_uses_backend_from_received_array(self):
        module = fake_cupy()
        array = FakeCupyArray([[0, 2], [2, 0]])
        with patch.dict(sys.modules, {"cupy": module}):
            matched = helpers.match_mask_class(
                "mask.nii.gz", array,
                {"default": {"lesion": "2"}}, "default",
                logger=self.logger,
            )
        self.assertEqual(matched, "lesion")

    def test_numpy_conversion_is_backend_explicit(self):
        config.GPU_ENABLED = False
        preprocessor = self._preprocessor()
        source = np.arange(8).reshape(2, 2, 2)
        self.assertIs(preprocessor._as_numpy(source), source)


if __name__ == "__main__":
    unittest.main()
