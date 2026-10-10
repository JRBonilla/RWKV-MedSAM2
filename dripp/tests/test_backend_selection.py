"""
Test CPU/GPU backend selection and explicit array conversion using a fake CuPy module.
"""

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
    """
    Create a minimal CuPy module stand-in with configurable CUDA availability.

    Args:
        device_count (int): Number of CUDA devices reported by the fake CuPy backend.

    Returns:
        types.ModuleType: Fake array backend exposing the methods used by the tested
            preprocessing paths.
    """
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
    """
    Wrap a NumPy array with the minimal interface needed for CuPy backend tests.
    """

    def __init__(self, values):
        """
        Wrap NumPy values in an array stand-in used to exercise backend selection.

        Args:
            values (np.ndarray | Sequence): Values to expose through the fake backend
                array.

        Returns:
            None.
        """
        self.values = np.asarray(values)
        self.dtype = self.values.dtype

    def ravel(self):
        """
        Flatten the wrapped NumPy values.

        Args:
            None.

        Returns:
            np.ndarray: One-dimensional view of the fixture array.
        """
        return self.values.ravel()


class BackendSelectionTests(unittest.TestCase):
    """
    Check lazy GPU initialization and backend-specific array conversion.
    """

    def setUp(self):
        """
        Save the active GPU setting and prepare a silent test logger.

        Args:
            None.

        Returns:
            None.
        """
        self.original_gpu_enabled = config.GPU_ENABLED
        self.logger = logging.getLogger(f"{__name__}.{self._testMethodName}")
        self.logger.handlers = [logging.NullHandler()]
        self.logger.propagate = False

    def tearDown(self):
        """
        Restore the GPU setting after each backend-selection test.

        Args:
            None.

        Returns:
            None.
        """
        config.GPU_ENABLED = self.original_gpu_enabled

    def _preprocessor(self):
        """
        Construct a small preprocessor using the current runtime backend setting.

        Args:
            None.

        Returns:
            Preprocessor: Test preprocessor configured with a two-pixel resize target.
        """
        return Preprocessor(
            target_size=(2, 2),
            min_mask_size=1,
            dataset_logger=self.logger,
            dataset_name="backend-test",
        )

    def test_cpu_startup_does_not_import_cupy(self):
        """
        Verify that CPU initialization leaves CuPy unimported.

        Args:
            None.

        Returns:
            None.
        """
        config.GPU_ENABLED = False
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            """
            Reject unexpected CuPy imports while delegating other imports normally.

            Args:
                name (str): Module name passed to the import wrapper.
                args (tuple): Additional positional arguments forwarded to the real
                    module importer.
                kwargs (dict): Keyword arguments forwarded to the delegated module
                    import.

            Returns:
                module: Imported module for names other than cupy.
            """
            if name == "cupy":
                raise AssertionError("CPU startup attempted to import CuPy")

            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            preprocessor = self._preprocessor()

        self.assertIs(preprocessor.xp, np)

    def test_gpu_selection_happens_after_runtime_flag_changes(self):
        """
        Verify that backend selection respects runtime changes to the GPU setting.

        Args:
            None.

        Returns:
            None.
        """
        module = fake_cupy()
        config.GPU_ENABLED = True

        with patch.dict(sys.modules, {"cupy": module}):
            preprocessor = self._preprocessor()

        self.assertIs(preprocessor.xp, module)

    def test_gpu_request_without_cupy_has_clear_error(self):
        """
        Verify that a GPU request reports a missing CuPy dependency clearly.

        Args:
            None.

        Returns:
            None.
        """
        config.GPU_ENABLED = True
        original_import = builtins.__import__

        def missing_cupy(name, *args, **kwargs):
            """
            Simulate unavailable CuPy while delegating other module imports.

            Args:
                name (str): Module name passed to the import wrapper.
                args (tuple): Additional positional arguments forwarded to the real
                    module importer.
                kwargs (dict): Keyword arguments forwarded to the delegated module
                    import.

            Returns:
                module: Imported module for names other than cupy; raises ImportError
                    for cupy.
            """
            if name == "cupy":
                raise ImportError("not installed")

            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_cupy):
            with self.assertRaisesRegex(RuntimeError, "CuPy is not installed"):
                self._preprocessor()

    def test_gpu_request_without_cuda_has_clear_error(self):
        """
        Verify that a GPU request reports an unavailable CUDA device clearly.

        Args:
            None.

        Returns:
            None.
        """
        module = fake_cupy(device_count=0)
        config.GPU_ENABLED = True

        with patch.dict(sys.modules, {"cupy": module}):
            with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
                self._preprocessor()

    def test_helper_uses_backend_from_received_array(self):
        """
        Verify that array helpers select the backend from their input arrays.

        Args:
            None.

        Returns:
            None.
        """
        module = fake_cupy()
        array = FakeCupyArray([[0, 2], [2, 0]])

        with patch.dict(sys.modules, {"cupy": module}):
            matched = helpers.match_mask_class(
                "mask.nii.gz",
                array,
                {"default": {"lesion": "2"}},
                "default",
                logger=self.logger,
            )

        self.assertEqual(matched, "lesion")

    def test_numpy_conversion_is_backend_explicit(self):
        """
        Verify explicit host conversion for NumPy and CuPy arrays.

        Args:
            None.

        Returns:
            None.
        """
        config.GPU_ENABLED = False
        preprocessor = self._preprocessor()
        source = np.arange(8).reshape(2, 2, 2)
        self.assertIs(preprocessor._as_numpy(source), source)


if __name__ == "__main__":
    unittest.main()
