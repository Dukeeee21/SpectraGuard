"""Loop de entrenamiento del `HybridDetector`.

Uso:
    python -m ml.training.train --data-root data/ --device cpu

Espera `data/real/*.jpg` y `data/ai_generated/*.jpg` (ver
`ml/training/dataset.py`). Los hiperparámetros salen de
`ml/training/config.yaml`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split

from ml.models.hybrid_model import HybridDetector
from ml.training.dataset import DeepfakeDataset, Sample3, collate_skip_invalid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def _compute_metrics(
    loss_sum: float, num_batches: int, tp: int, tn: int, fp: int, fn: int
) -> EpochMetrics:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EpochMetrics(
        loss=loss_sum / num_batches if num_batches else 0.0,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )


def _run_epoch(
    model: HybridDetector,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    loss_sum, num_batches = 0.0, 0
    tp = tn = fp = fn = 0

    for batch in loader:
        if batch is None:  # todo el batch se descartó (sin rostro detectable)
            continue
        faces, spectra, labels = (t.to(device) for t in batch)

        with torch.set_grad_enabled(is_training):
            logits = model(faces, spectra)
            loss = criterion(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_sum += loss.item()
        num_batches += 1

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        tp += int(((predictions == 1) & (labels == 1)).sum())
        tn += int(((predictions == 0) & (labels == 0)).sum())
        fp += int(((predictions == 1) & (labels == 0)).sum())
        fn += int(((predictions == 0) & (labels == 1)).sum())

    return _compute_metrics(loss_sum, num_batches, tp, tn, fp, fn)


def _make_loaders(
    data_root: Path, image_size: int, device: str, batch_size: int, val_fraction: float, seed: int
) -> tuple[DataLoader, DataLoader, int]:
    dataset = DeepfakeDataset(root=data_root, image_size=image_size, device=device)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader: DataLoader[Sample3] = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_invalid
    )
    val_loader: DataLoader[Sample3] = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_skip_invalid
    )
    return train_loader, val_loader, len(dataset)


def _save_metrics_report(checkpoint_path: Path, metrics: EpochMetrics, dataset_size: int) -> None:
    report_path = checkpoint_path.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps({**asdict(metrics), "dataset_size": dataset_size}, indent=2),
        encoding="utf-8",
    )


def train(
    data_root: Path,
    output_path: Path,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_fraction: float = 0.2,
    image_size: int = 224,
    device: str = "cpu",
    seed: int = 42,
) -> EpochMetrics:
    """Entrena `HybridDetector` y guarda el mejor checkpoint (por F1 de
    validación) en `output_path`, junto a un `.metrics.json` con el reporte.
    Devuelve las métricas de ese mejor checkpoint."""
    torch.manual_seed(seed)
    torch_device = torch.device(device)

    train_loader, val_loader, dataset_size = _make_loaders(
        data_root, image_size, device, batch_size, val_fraction, seed
    )

    model = HybridDetector(pretrained_spatial=True).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_metrics: Optional[EpochMetrics] = None

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, criterion, torch_device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, torch_device, None)
        elapsed = time.perf_counter() - start

        logger.info(
            "epoch %d/%d (%.1fs) - train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            epochs,
            elapsed,
            train_metrics.loss,
            val_metrics.loss,
            val_metrics.accuracy,
            val_metrics.f1_score,
        )

        if val_metrics.f1_score > best_f1:
            best_f1 = val_metrics.f1_score
            best_metrics = val_metrics
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            _save_metrics_report(output_path, best_metrics, dataset_size)
            logger.info("Nuevo mejor checkpoint guardado en '%s' (f1=%.4f)", output_path, best_f1)

    if best_metrics is None:
        raise RuntimeError("El entrenamiento no completó ninguna época con datos válidos.")
    return best_metrics


def _load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el HybridDetector de SpectraGuard.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/weights/hybrid_detector_v1.pth"))
    parser.add_argument("--config", type=Path, default=Path("ml/training/config.yaml"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = _load_config(args.config)
    train(
        data_root=args.data_root,
        output_path=args.output,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        val_fraction=config.get("val_fraction", 0.2),
        image_size=config["image_size"],
        device=args.device,
        seed=config.get("seed", 42),
    )


if __name__ == "__main__":
    main()
