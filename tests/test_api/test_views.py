"""Tests de integración de `POST /api/v1/analyze/`, contra Postgres real.

La extracción de rostro (MTCNN) se mockea: probar el endpoint no requiere
una foto real con rostro, solo verificar que la vista arma correctamente
el request/response, valida la subida y persiste el resultado.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from rest_framework import status
from rest_framework.test import APIClient

from apps.detection.models import AnalysisLog
from apps.detection.services import inference
from ml.models.hybrid_model import HybridDetector
from ml.preprocessing.face_extractor import FaceExtractionResult, NoFaceDetectedError

_ANALYZE_URL = "/api/v1/analyze/"


def _jpeg_upload(name: str = "photo.jpg") -> SimpleUploadedFile:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color=(120, 140, 160)).save(buffer, format="JPEG")
    return SimpleUploadedFile(name, buffer.getvalue(), content_type="image/jpeg")


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def loaded_predictor(tmp_path: Path, settings) -> Iterator[None]:
    """Crea un checkpoint válido (pesos sin entrenar, pero con la forma
    correcta) y hace que el servicio de inferencia lo cargue, como si
    Django acabara de arrancar con un modelo ya entrenado disponible."""
    checkpoint_path = tmp_path / "test_checkpoint.pth"
    torch.save(HybridDetector(pretrained_spatial=False).state_dict(), checkpoint_path)

    settings.ML_MODEL = {"WEIGHTS_PATH": checkpoint_path, "DEVICE": "cpu", "IMAGE_SIZE": 224}
    inference.load_predictor()
    yield
    inference.reset()


@pytest.mark.django_db
class TestImageAnalysisView:
    def test_returns_503_when_model_not_loaded(self, api_client: APIClient) -> None:
        inference.reset()

        response = api_client.post(_ANALYZE_URL, {"image": _jpeg_upload()}, format="multipart")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_returns_422_when_no_face_detected(
        self, api_client: APIClient, loaded_predictor: None
    ) -> None:
        with patch(
            "ml.preprocessing.face_extractor.FaceExtractor.extract",
            side_effect=NoFaceDetectedError("no face"),
        ):
            response = api_client.post(_ANALYZE_URL, {"image": _jpeg_upload()}, format="multipart")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_rejects_non_image_upload(self, api_client: APIClient, loaded_predictor: None) -> None:
        text_upload = SimpleUploadedFile("not_image.txt", b"hola", content_type="text/plain")

        response = api_client.post(_ANALYZE_URL, {"image": text_upload}, format="multipart")

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_happy_path_returns_prediction_and_persists_log(
        self, api_client: APIClient, loaded_predictor: None
    ) -> None:
        fake_face = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        fake_extraction = FaceExtractionResult(
            face=fake_face, confidence=0.97, box=(10.0, 10.0, 200.0, 200.0)
        )

        with patch(
            "ml.preprocessing.face_extractor.FaceExtractor.extract",
            return_value=fake_extraction,
        ):
            response = api_client.post(_ANALYZE_URL, {"image": _jpeg_upload()}, format="multipart")

        assert response.status_code == status.HTTP_200_OK
        body = response.json()
        assert body["label"] in {"real", "ai_generated"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["face_confidence"] == pytest.approx(0.97)

        assert AnalysisLog.objects.count() == 1
        log_entry = AnalysisLog.objects.get()
        assert log_entry.label == body["label"]
