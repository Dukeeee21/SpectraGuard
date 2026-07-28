"""Rama espacial: backbone CNN que busca artefactos visuales, asimetrías y
bordes extraños en el rostro alineado.
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class SpatialBranch(nn.Module):
    """Extrae un vector de características del rostro (RGB) usando un
    EfficientNet como backbone.

    Se parte de pesos preentrenados en ImageNet (transfer learning) en vez
    de entrenar desde cero: el dataset de rostros reales/generados es
    órdenes de magnitud más chico que ImageNet, y las texturas y bordes de
    bajo nivel que EfficientNet ya aprendió generalizan bien a esta tarea.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # sin cabeza de clasificación: devuelve el feature vector
        )
        self.output_dim: int = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, face: torch.Tensor) -> torch.Tensor:
        """face: (B, 3, H, W), normalizado con media/desvío de ImageNet."""
        return self.backbone(face)
