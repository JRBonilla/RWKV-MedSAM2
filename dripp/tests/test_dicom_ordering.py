from types import SimpleNamespace
import unittest

from dripp.preprocessor import _dicom_slice_key_from_dataset


def _dataset(position, orientation, instance):
    return SimpleNamespace(
        ImagePositionPatient=position,
        ImageOrientationPatient=orientation,
        InstanceNumber=instance,
    )


class DicomOrderingTests(unittest.TestCase):
    def test_slice_key_follows_negative_acquisition_normal(self):
        orientation = [1, 0, 0, 0, -1, 0]
        slices = [
            _dataset([0, 0, -208], orientation, 209),
            _dataset([0, 0, 0], orientation, 1),
            _dataset([0, 0, -1], orientation, 2),
        ]

        ordered = sorted(slices, key=_dicom_slice_key_from_dataset)

        self.assertEqual([item.InstanceNumber for item in ordered], [1, 2, 209])

    def test_slice_key_follows_positive_acquisition_normal(self):
        orientation = [1, 0, 0, 0, 1, 0]
        slices = [
            _dataset([0, 0, 2], orientation, 3),
            _dataset([0, 0, 0], orientation, 1),
            _dataset([0, 0, 1], orientation, 2),
        ]

        ordered = sorted(slices, key=_dicom_slice_key_from_dataset)

        self.assertEqual([item.InstanceNumber for item in ordered], [1, 2, 3])

    def test_slice_key_falls_back_to_instance_number(self):
        slices = [
            SimpleNamespace(InstanceNumber=3),
            SimpleNamespace(InstanceNumber=1),
            SimpleNamespace(InstanceNumber=2),
        ]

        ordered = sorted(slices, key=_dicom_slice_key_from_dataset)

        self.assertEqual([item.InstanceNumber for item in ordered], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
