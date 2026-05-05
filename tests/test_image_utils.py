"""Unit tests for image utility helpers."""

import numpy as np
from pydicom.dataset import Dataset

from cv.image_utils import _apply_window_level, _extract_dicom_metadata_from_bytes, _normalize_array_to_uint8


def test_normalize_array_to_uint8_scales_values() -> None:
    input_array = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    output_array = _normalize_array_to_uint8(input_array)

    assert output_array.dtype == np.uint8
    assert int(output_array.min()) == 0
    assert int(output_array.max()) == 255


def test_normalize_array_to_uint8_constant_array_returns_zeroes() -> None:
    input_array = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    output_array = _normalize_array_to_uint8(input_array)

    assert np.array_equal(output_array, np.zeros((2, 2), dtype=np.uint8))


def test_extract_dicom_metadata_from_bytes_returns_expected_fields() -> None:
    dataset = Dataset()
    dataset.PatientName = "Jane Doe"
    dataset.PatientID = "12345"
    dataset.Modality = "MR"
    dataset.StudyDate = "20260505"
    dataset.StudyTime = "120000"
    dataset.SeriesDescription = "Knee Follow-up"
    dataset.Rows = 256
    dataset.Columns = 256
    dataset.PixelSpacing = [0.5, 0.5]

    # Build a minimal DICOM-like payload path: use pydicom to write proper bytes.
    from io import BytesIO
    from pydicom.filewriter import dcmwrite

    dataset.is_little_endian = True
    dataset.is_implicit_VR = True

    payload = BytesIO()
    dcmwrite(payload, dataset, write_like_original=True)
    metadata = _extract_dicom_metadata_from_bytes(payload.getvalue())

    assert metadata["Patient Name"] == "Jane Doe"
    assert metadata["Patient ID"] == "12345"
    assert metadata["Modality"] == "MR"
    assert metadata["Dimensions"] == "256 x 256"


def test_apply_window_level_bounds_intensity() -> None:
    input_array = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    output_array = _apply_window_level(input_array, center=100.0, width=100.0)

    assert output_array.dtype == np.uint8
    assert int(output_array[0, 0]) == 0
    assert int(output_array[1, 1]) == 255
