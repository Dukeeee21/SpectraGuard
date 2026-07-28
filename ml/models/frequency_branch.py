"""Rama de frecuencia: CNN ligera sobre el espectro DFT para detectar el
ruido de alta frecuencia característico de la IA generativa.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyBranch(nn.Module):
    """Procesa el espectro de magnitud (1 canal) devuelto por
    `ml.preprocessing.frequency.compute_frequency_spectrum`.

    Es deliberadamente mucho más chica que la rama espacial: no necesita
    reconocer objetos, formas ni texturas, solo la energía relativa de
    cada banda de frecuencia. Un backbone preentrenado en ImageNet no
    aportaría nada útil aquí y solo sumaría parámetros a entrenar sin
    datos de sobra para justificarlos.
    """

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.features = nn.Sequential(
            self._conv_block(1, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        # AdaptiveAvgPool2d(1) en vez de aplanar directamente: así la rama
        # acepta cualquier tamaño de espectro de entrada sin recalcular a
        # mano las dimensiones del Linear de salida.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(256, output_dim)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """spectrum: (B, 1, H, W), normalizado a [0, 1]."""
        x = self.features(spectrum)
        x = self.pool(x).flatten(1)
        return self.projection(x)
