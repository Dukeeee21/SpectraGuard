"""Tests para el preprocesamiento: extracción de rostro y espectro DFT."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ml.preprocessing.face_extractor import FaceExtractor, NoFaceDetectedError
from ml.preprocessing.frequency import compute_frequency_spectrum


class TestComputeFrequencySpectrum:
    def test_output_shape_dtype_and_range(self) -> None:
        image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

        spectrum = compute_frequency_spectrum(image)

        assert spectrum.shape == (64, 64)
        assert spectrum.dtype == np.float32
        assert spectrum.min() >= 0.0
        assert spectrum.max() <= 1.0

    def test_high_frequency_pattern_has_more_energy_at_the_edges(self) -> None:
        size = 64
        flat_image = np.full((size, size, 3), 128, dtype=np.uint8)

        checkerboard = (np.indices((size, size)).sum(axis=0) % 2).astype(np.uint8)
        checkerboard_image = np.stack([checkerboard * 255] * 3, axis=-1)

        flat_spectrum = compute_frequency_spectrum(flat_image)
        checkerboard_spectrum = compute_frequency_spectrum(checkerboard_image)

        # Tras el fftshift, la componente de máxima frecuencia (el patrón de
        # checkerboard) cae en la esquina del espectro; una imagen plana no
        # tiene esa frecuencia, así que ahí su energía debería ser ~0.
        border = 4
        edge_energy_flat = flat_spectrum[:border, :].mean()
        edge_energy_checkerboard = checkerboard_spectrum[:border, :].mean()

        assert edge_energy_checkerboard > edge_energy_flat


def _fake_face(bbox: list[float], det_score: float, kps: list[list[float]]) -> SimpleNamespace:
    """Imita el objeto `Face` que devuelve `insightface.app.FaceAnalysis.get()`
    (solo los atributos que usa `FaceExtractor`: bbox, det_score, kps)."""
    return SimpleNamespace(
        bbox=np.array(bbox, dtype=np.float32),
        det_score=det_score,
        kps=np.array(kps, dtype=np.float32),
    )


class TestFaceExtractor:
    def test_raises_when_no_face_is_detected(self) -> None:
        extractor = FaceExtractor(image_size=112, device="cpu")
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(extractor._app, "get", return_value=[]):
            with pytest.raises(NoFaceDetectedError):
                extractor.extract(blank_image)

    def test_align_produces_square_crop_of_requested_size(self) -> None:
        extractor = FaceExtractor(image_size=112, device="cpu")
        source_image = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
        eyes = np.array([[70.0, 80.0], [130.0, 80.0]])

        aligned = extractor._align(source_image, eyes)

        assert aligned.shape == (112, 112, 3)

    def test_extract_picks_the_most_confident_face(self) -> None:
        extractor = FaceExtractor(image_size=64, device="cpu")
        source_image = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)

        low_confidence = _fake_face(
            bbox=[0, 0, 50, 50],
            det_score=0.55,
            kps=[[10, 10], [40, 10], [25, 25], [15, 40], [35, 40]],
        )
        high_confidence = _fake_face(
            bbox=[60, 60, 150, 150],
            det_score=0.97,
            kps=[[80, 80], [130, 80], [105, 105], [90, 130], [120, 130]],
        )

        with patch.object(extractor._app, "get", return_value=[low_confidence, high_confidence]):
            result = extractor.extract(source_image)

        assert result.confidence == pytest.approx(0.97)
        assert result.box == (60.0, 60.0, 150.0, 150.0)
        assert result.face.shape == (64, 64, 3)
