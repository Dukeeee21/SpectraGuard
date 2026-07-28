"""Dataset de PyTorch para entrenar el `HybridDetector`.

Espera la estructura `root/real/*.jpg` y `root/ai_generated/*.jpg`. Cada
muestra devuelve (rostro normalizado, espectro DFT, etiqueta), reusando el
mismo preprocesamiento de `ml/preprocessing/` que usa la inferencia — así
entrenamiento e inferencia ven exactamente los mismos datos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ml.preprocessing.face_extractor import FaceExtractor, NoFaceDetectedError
from ml.preprocessing.frequency import compute_frequency_spectrum

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# 1.0 = clase positiva. `Predictor` interpreta sigmoid(logit) como
# P(generado por IA), así que la etiqueta tiene que usar la misma
# convención acá o el modelo entrenado quedaría invertido.
_CLASS_TO_LABEL = {"real": 0.0, "ai_generated": 1.0}

Sample3 = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _ImageSample:
    path: Path
    label: float


class DeepfakeDataset(Dataset):
    """`__getitem__` devuelve `None` (en vez de lanzar) cuando la imagen no
    tiene un rostro detectable, para no tirar abajo todo el entrenamiento
    por una foto rara en el dataset — es normal que una fracción de un
    dataset scrapeado no tenga un rostro claro. Usar `collate_skip_invalid`
    como `collate_fn` del DataLoader para descartar esos casos del batch.
    """

    def __init__(self, root: Path, image_size: int = 224, device: str = "cpu") -> None:
        self.samples = self._scan(Path(root))
        if not self.samples:
            raise ValueError(
                f"No se encontraron imágenes en '{root}'. Se espera la "
                f"estructura {root}/real/*.jpg y {root}/ai_generated/*.jpg."
            )
        self._face_extractor = FaceExtractor(image_size=image_size, device=device)
        self._face_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    @staticmethod
    def _scan(root: Path) -> list[_ImageSample]:
        samples: list[_ImageSample] = []
        for class_name, label in _CLASS_TO_LABEL.items():
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in _IMAGE_EXTENSIONS:
                    samples.append(_ImageSample(path=path, label=label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Optional[Sample3]:
        sample = self.samples[index]
        try:
            image = Image.open(sample.path).convert("RGB")
            extraction = self._face_extractor.extract(image)
        except (NoFaceDetectedError, OSError) as exc:
            logger.warning("Se descarta '%s': %s", sample.path, exc)
            return None

        spectrum = compute_frequency_spectrum(extraction.face)
        face_tensor = self._face_transform(extraction.face)
        spectrum_tensor = torch.from_numpy(spectrum).unsqueeze(0)
        label_tensor = torch.tensor([sample.label], dtype=torch.float32)
        return face_tensor, spectrum_tensor, label_tensor


def collate_skip_invalid(batch: list[Optional[Sample3]]) -> Optional[Sample3]:
    """Descarta del batch las muestras que `DeepfakeDataset.__getitem__`
    marcó como inválidas. Devuelve `None` si el batch entero quedó vacío."""
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    faces, spectra, labels = zip(*valid)
    return torch.stack(faces), torch.stack(spectra), torch.stack(labels)
