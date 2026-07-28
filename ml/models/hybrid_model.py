"""HybridDetector: fusiona la rama espacial y la rama de frecuencia para
emitir la predicción final (real vs. generado por IA).
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.frequency_branch import FrequencyBranch
from ml.models.spatial_branch import SpatialBranch


class HybridDetector(nn.Module):
    """Concatena las salidas de `SpatialBranch` y `FrequencyBranch` y las
    pasa por una cabeza densa (fully connected) que emite un único logit.

    Ninguna de las dos ramas por sí sola es suficiente: los artefactos
    espaciales (bordes, asimetrías) y la firma de frecuencia (ruido de
    upsampling) son señales independientes que distintas técnicas
    generativas exponen en distinto grado. Concatenar antes de la
    clasificación deja que la cabeza densa aprenda a ponderar ambas según
    lo que cada caso realmente delate.
    """

    def __init__(
        self,
        spatial_backbone: str = "efficientnet_b0",
        pretrained_spatial: bool = True,
        frequency_output_dim: int = 256,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            backbone_name=spatial_backbone, pretrained=pretrained_spatial
        )
        self.frequency_branch = FrequencyBranch(output_dim=frequency_output_dim)

        fused_dim = self.spatial_branch.output_dim + frequency_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, face: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        """face: (B, 3, H, W). spectrum: (B, 1, H, W).

        Devuelve el logit crudo de shape (B, 1) — sin sigmoid — para poder
        entrenar con `BCEWithLogitsLoss` (más estable numéricamente que
        aplicar sigmoid + BCELoss por separado). La probabilidad se calcula
        recién en inferencia (Fase 4), aplicando sigmoid sobre este logit.
        """
        spatial_features = self.spatial_branch(face)
        frequency_features = self.frequency_branch(spectrum)
        fused = torch.cat([spatial_features, frequency_features], dim=1)
        return self.classifier(fused)
