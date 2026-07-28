"""Detección, alineación y recorte de rostros.

Este módulo no depende de Django: recibe una imagen en memoria y devuelve un
array de numpy, para poder testearse y reutilizarse igual en un script de
entrenamiento que en el servicio de inferencia de la API (Fase 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

# Posiciones canónicas de los ojos dentro del recorte de salida cuadrado,
# como fracción de `image_size`. Calibradas para dejar margen simétrico
# alrededor del rostro (frente, mentón, orejas) tras alinear: si el recorte
# quedara demasiado ajustado se perdería el borde del rostro, que es donde
# más se concentran los artefactos de mezcla de GANs/difusión.
_CANONICAL_LEFT_EYE = (0.35, 0.35)
_CANONICAL_RIGHT_EYE = (0.65, 0.35)


class NoFaceDetectedError(RuntimeError):
    """La imagen de entrada no contiene ningún rostro detectable por MTCNN."""


@dataclass(frozen=True)
class FaceExtractionResult:
    face: np.ndarray  # HxWx3 uint8 RGB, alineado y recortado a `image_size`
    confidence: float  # probabilidad de detección del rostro elegido, en [0, 1]
    box: tuple[float, float, float, float]  # bounding box original (x1, y1, x2, y2)


class FaceExtractor:
    """Envuelve MTCNN (facenet-pytorch) para detectar, alinear por landmarks
    oculares y recortar el rostro principal de una imagen."""

    def __init__(self, image_size: int = 224, device: str = "cpu") -> None:
        self.image_size = image_size
        # keep_all=True: pedimos todos los rostros detectados (con sus
        # landmarks) para elegir nosotros el de mayor confianza, en vez de
        # depender del criterio "el más grande" que usa MTCNN por defecto.
        self._mtcnn = MTCNN(keep_all=True, device=device)

    def extract(self, image: Image.Image | np.ndarray) -> FaceExtractionResult:
        """Detecta el rostro de mayor confianza en `image` y devuelve su
        recorte alineado. Lanza `NoFaceDetectedError` si no hay ninguno."""
        pil_image = self._to_pil_rgb(image)

        boxes, probs, landmarks = self._mtcnn.detect(pil_image, landmarks=True)
        if boxes is None:
            raise NoFaceDetectedError("No se detectó ningún rostro en la imagen.")

        best_index = int(np.argmax(probs))
        box = boxes[best_index]
        eyes = landmarks[best_index][:2]  # [ojo_1, ojo_2] en coordenadas de imagen

        aligned = self._align(np.array(pil_image), eyes)
        x1, y1, x2, y2 = (float(v) for v in box)
        return FaceExtractionResult(
            face=aligned,
            confidence=float(probs[best_index]),
            box=(x1, y1, x2, y2),
        )

    def _align(self, image_rgb: np.ndarray, eyes: np.ndarray) -> np.ndarray:
        """Aplica una transformación de similitud (rotación + escala +
        traslación) que lleva ambos ojos a su posición canónica, y recorta a
        `image_size`x`image_size`. Alinear por los ojos normaliza la
        inclinación de la cabeza antes de buscar artefactos: sin esto, una
        foto simplemente ladeada podría confundirse con una asimetría
        generada por IA.
        """
        size = self.image_size
        target = np.array(
            [
                [_CANONICAL_LEFT_EYE[0] * size, _CANONICAL_LEFT_EYE[1] * size],
                [_CANONICAL_RIGHT_EYE[0] * size, _CANONICAL_RIGHT_EYE[1] * size],
            ],
            dtype=np.float32,
        )
        source = np.asarray(eyes, dtype=np.float32)

        transform, _ = cv2.estimateAffinePartial2D(source, target)
        return cv2.warpAffine(image_rgb, transform, (size, size), flags=cv2.INTER_LINEAR)

    @staticmethod
    def _to_pil_rgb(image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")
