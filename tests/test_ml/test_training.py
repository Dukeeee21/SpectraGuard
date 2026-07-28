"""Smoke test del pipeline de entrenamiento.

No hay un dataset real de rostros reales/generados en el repo, así que la
extracción de rostro se mockea (las imágenes sintéticas no tienen un rostro
que MTCNN pueda detectar). Lo que se valida acá es que todo el circuito
-- escaneo de carpetas, Dataset, DataLoader, loop de entrenamiento,
guardado de checkpoint y reporte de métricas -- corre de punta a punta sin
romperse, no la calidad de un modelo entrenado con datos de juguete.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from ml.preprocessing.face_extractor import FaceExtractionResult
from ml.training.train import train


def _make_dataset_dir(root: Path, real_count: int, ai_count: int) -> None:
    for class_name, count in (("real", real_count), ("ai_generated", ai_count)):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (64, 64), color=(i * 10 % 255, 50, 100)).save(class_dir / f"{i}.jpg")


@pytest.fixture(autouse=True)
def _mock_face_extraction():
    fake_face = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    fake_result = FaceExtractionResult(face=fake_face, confidence=0.9, box=(0.0, 0.0, 64.0, 64.0))
    with patch("ml.preprocessing.face_extractor.FaceExtractor.extract", return_value=fake_result):
        yield


class TestTrainingPipeline:
    def test_train_runs_end_to_end_and_saves_checkpoint(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        _make_dataset_dir(data_root, real_count=6, ai_count=6)
        output_path = tmp_path / "checkpoint.pth"

        best_metrics = train(
            data_root=data_root,
            output_path=output_path,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            val_fraction=0.34,
            image_size=224,
            device="cpu",
        )

        assert output_path.exists()
        assert 0.0 <= best_metrics.accuracy <= 1.0
        assert 0.0 <= best_metrics.f1_score <= 1.0

        report = json.loads(output_path.with_suffix(".metrics.json").read_text())
        assert report["dataset_size"] == 12

    def test_raises_clear_error_when_data_root_is_empty(self, tmp_path: Path) -> None:
        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        with pytest.raises(ValueError, match="No se encontraron imágenes"):
            train(data_root=empty_root, output_path=tmp_path / "out.pth", epochs=1)
