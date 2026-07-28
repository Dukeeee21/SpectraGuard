"""Detección, alineación y recorte de rostros.

Este módulo no depende de Django: recibe una imagen en memoria y devuelve un
array de numpy, para poder testearse y reutilizarse igual en un script de
entrenamiento que en el servicio de inferencia de la API (Fase 4).

Usa insightface (RetinaFace vía ONNX Runtime) en vez de MTCNN
(facenet-pytorch): da los mismos 5 landmarks faciales que necesitamos para
alinear, pero corre sobre ONNX Runtime en lugar de PyTorch, así la versión
de torch del proyecto queda libre para actualizarse (p. ej. para dar
soporte a GPUs nuevas) sin quedar atada a lo que exija esta librería.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

# Posiciones canónicas de los ojos dentro del recorte de salida cuadrado,
# como fracción de `image_size`. Calibradas para dejar margen simétrico
# alrededor del rostro (frente, mentón, orejas) tras alinear: si el recorte
# quedara demasiado ajustado se perdería el borde del rostro, que es donde
# más se concentran los artefactos de mezcla de GANs/difusión.
_CANONICAL_LEFT_EYE = (0.35, 0.35)
_CANONICAL_RIGHT_EYE = (0.65, 0.35)

# 320x320 balancea calidad de detección y velocidad: en pruebas, 640x640
# (el default recomendado por insightface para fotos de alta resolución)
# degradaba fuerte la detección en imágenes de dataset ya redimensionadas a
# 224x224 -- sobre-escalarlas introduce blur que confunde al detector.
_DEFAULT_DET_SIZE = (320, 320)
_DEFAULT_DET_THRESHOLD = 0.3


class NoFaceDetectedError(RuntimeError):
    """La imagen de entrada no contiene ningún rostro detectable."""


@dataclass(frozen=True)
class FaceExtractionResult:
    face: np.ndarray  # HxWx3 uint8 RGB, alineado y recortado a `image_size`
    confidence: float  # confianza de detección del rostro elegido, en [0, 1]
    box: tuple[float, float, float, float]  # bounding box original (x1, y1, x2, y2)


class FaceExtractor:
    """Detecta, alinea por landmarks oculares y recorta el rostro principal
    de una imagen."""

    def __init__(self, image_size: int = 224, device: str = "cpu") -> None:
        self.image_size = image_size

        is_gpu = device.startswith("cuda")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if is_gpu
            else ["CPUExecutionProvider"]
        )

        # allowed_modules=["detection"]: solo cargamos RetinaFace. Los demás
        # modelos del paquete "buffalo_l" (reconocimiento, edad/género, 106
        # landmarks) no hacen falta para recortar y alinear un rostro.
        self._app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection"], providers=providers
        )
        self._app.prepare(
            ctx_id=0 if is_gpu else -1,
            det_size=_DEFAULT_DET_SIZE,
            det_thresh=_DEFAULT_DET_THRESHOLD,
        )

    def extract(self, image: Image.Image | np.ndarray) -> FaceExtractionResult:
        """Detecta el rostro de mayor confianza en `image` y devuelve su
        recorte alineado. Lanza `NoFaceDetectedError` si no hay ninguno."""
        rgb_image = np.array(self._to_pil_rgb(image))
        bgr_image = rgb_image[:, :, ::-1]  # insightface espera BGR (convención OpenCV)

        faces = self._app.get(bgr_image)
        if not faces:
            raise NoFaceDetectedError("No se detectó ningún rostro en la imagen.")

        best = max(faces, key=lambda f: f.det_score)
        eyes = best.kps[:2]  # [ojo_1, ojo_2] en coordenadas de imagen

        aligned = self._align(rgb_image, eyes)
        x1, y1, x2, y2 = (float(v) for v in best.bbox)
        return FaceExtractionResult(
            face=aligned,
            confidence=float(best.det_score),
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
