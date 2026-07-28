"""Predictor: envuelve el pipeline completo de inferencia (extracción de
rostro + espectro DFT + HybridDetector) para servir una predicción a partir
de una imagen en memoria. No depende de Django.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.models.hybrid_model import HybridDetector
from ml.preprocessing.face_extractor import FaceExtractor
from ml.preprocessing.frequency import compute_frequency_spectrum

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelNotLoadedError(RuntimeError):
    """El checkpoint del modelo no existe todavía (no se corrió
    `ml/training/train.py`) o no se pudo cargar."""


@dataclass(frozen=True)
class PredictionResult:
    label: str  # "real" | "ai_generated"
    confidence: float  # confianza de la etiqueta predicha, en [0, 1]
    ai_probability: float  # P(generado por IA), en [0, 1]
    face_confidence: float  # confianza de la detección de rostro (MTCNN)
    processing_time_ms: float


class Predictor:
    """Carga el modelo una sola vez y expone `predict()` para reutilizarlo
    en cada request. Pensado para instanciarse una única vez por proceso
    (ver `apps/detection/services/inference.py`, que lo hace en
    `AppConfig.ready()`)."""

    def __init__(self, weights_path: Path, image_size: int = 224, device: str = "cpu") -> None:
        if not weights_path.exists():
            raise ModelNotLoadedError(
                f"No se encontró el checkpoint del modelo en '{weights_path}'. "
                "Hay que entrenarlo con ml/training/train.py y colocarlo ahí "
                "(o ajustar MODEL_WEIGHTS_PATH en .env) antes de poder servir "
                "predicciones reales."
            )

        self.device = torch.device(device)
        self.image_size = image_size

        self._face_extractor = FaceExtractor(image_size=image_size, device=device)
        self._model = HybridDetector(pretrained_spatial=False)
        state_dict = torch.load(weights_path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

        self._face_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image | np.ndarray) -> PredictionResult:
        """Puede lanzar `ml.preprocessing.face_extractor.NoFaceDetectedError`
        si no hay rostro detectable en `image` — se deja propagar para que
        el caller (la vista DRF) decida el código de estado HTTP."""
        start = time.perf_counter()

        extraction = self._face_extractor.extract(image)
        spectrum = compute_frequency_spectrum(extraction.face)

        face_tensor = self._face_transform(extraction.face).unsqueeze(0).to(self.device)
        spectrum_tensor = torch.from_numpy(spectrum).unsqueeze(0).unsqueeze(0).to(self.device)

        logit = self._model(face_tensor, spectrum_tensor)
        ai_probability = torch.sigmoid(logit).item()

        label = "ai_generated" if ai_probability >= 0.5 else "real"
        confidence = ai_probability if label == "ai_generated" else 1.0 - ai_probability

        return PredictionResult(
            label=label,
            confidence=confidence,
            ai_probability=ai_probability,
            face_confidence=extraction.confidence,
            processing_time_ms=(time.perf_counter() - start) * 1000,
        )
