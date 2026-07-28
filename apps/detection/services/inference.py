"""Puente entre Django y `ml.inference.predictor`. Mantiene un único
`Predictor` cargado en memoria por proceso, instanciado desde
`DetectionConfig.ready()` al arrancar Django.
"""

from __future__ import annotations

import logging
from pathlib import Path

from django.conf import settings

from ml.inference.predictor import ModelNotLoadedError, Predictor

logger = logging.getLogger(__name__)

_predictor: Predictor | None = None
_model_version: str | None = None
_load_error: str | None = None


def load_predictor() -> None:
    """Intenta cargar el modelo en memoria. Si el checkpoint todavía no
    existe (no se corrió el entrenamiento), no tira abajo el arranque de
    Django: deja el servicio en estado "no listo" y registra por qué, para
    que la vista pueda responder 503 en vez de fallar al importar.
    """
    global _predictor, _model_version, _load_error

    weights_path: Path = settings.ML_MODEL["WEIGHTS_PATH"]
    try:
        _predictor = Predictor(
            weights_path=weights_path,
            image_size=settings.ML_MODEL["IMAGE_SIZE"],
            device=settings.ML_MODEL["DEVICE"],
        )
        _model_version = weights_path.stem
        logger.info("Modelo de detección cargado: %s", _model_version)
    except ModelNotLoadedError as exc:
        _load_error = str(exc)
        logger.warning("Modelo de detección no disponible: %s", exc)


def get_predictor() -> Predictor | None:
    return _predictor


def get_model_version() -> str:
    return _model_version or "unknown"


def get_load_error() -> str | None:
    return _load_error


def reset() -> None:
    """Solo para tests: vuelve el servicio al estado "no cargado", ya que
    `_predictor` vive a nivel de módulo y sobrevive entre tests."""
    global _predictor, _model_version, _load_error
    _predictor = None
    _model_version = None
    _load_error = None
