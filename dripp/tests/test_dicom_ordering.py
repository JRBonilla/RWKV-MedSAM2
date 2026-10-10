"""
Test DICOM slice ordering from acquisition geometry and instance-number fallbacks.
"""

from types import SimpleNamespace
import unittest

from dripp.preprocessor import _dicom_slice_key_from_dataset


def _dataset(position, orientation, instance):
    """
    Build a DICOM header fixture with position, orientation, and instance number.

    Args:
        position (Sequence[float]): Patient-space coordinates of the synthetic DICOM
            slice.
        orientation (Sequence[float]): Synthetic DICOM direction cosines for the row and
            column axes.
        instance (int): Synthetic DICOM instance number used as a fallback slice order.

    Returns:
        pydicom.dataset.Dataset: Header for slice-ordering tests.
    """
    return SimpleNamespace(
        ImagePositionPatient=position,
        ImageOrientationPatient=orientation,
        InstanceNumber=instance,
    )


class DicomOrderingTests(unittest.TestCase):
    """
    Check geometry-aware and fallback DICOM slice ordering.
    """

    def test_slice_key_follows_negative_acquisition_normal(self):
        """
        Verify slice ordering along a negative acquisition normal.

        Args:
            None.

        Returns:
            None.
        """
        orientation = [1, 0, 0, 0, -1, 0]
        slices = [
            _dataset([0, 0, -208], orientation, 209),
            _dataset([0, 0, 0], orientation, 1),
            _dataset([0, 0, -1], orientation, 2),
        ]

        ordered = sorted(slices, key=_dicom_slice_key_from_dataset)

        self.assertEqual([item.InstanceNumber for item in ordered], [1, 2, 209])

    def test_slice_key_follows_positive_acquisition_normal(self):
        """
        Verify slice ordering along a positive acquisition normal.

        Args:
            None.

        Returns:
            None.
        """
        orientation = [1, 0, 0, 0, 1, 0]
        slices = [
            _dataset([0, 0, 2], orientation, 3),
            _dataset([0, 0, 0], orientation, 1),
            _dataset([0, 0, 1], orientation, 2),
        ]

        ordered = sorted(slices, key=_dicom_slice_key_from_dataset)

        self.assertEqual([item.InstanceNumber for item in ordered], [1, 2, 3])

    def test_slice_key_falls_back_to_instance_number(self):
        """
        Verify instance-number ordering when DICOM geometry is unavailable.

        Args:
            None.

        Returns:
            None.
        """
        slices = [
            SimpleNamespace(InstanceNumber=3),
            SimpleNamespace(InstanceNumber=1),
            SimpleNamespace(InstanceNumber=2),
        ]

        ordered = sorted(slices, key=_dicom_slice_key_from_dataset)

        self.assertEqual([item.InstanceNumber for item in ordered], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
