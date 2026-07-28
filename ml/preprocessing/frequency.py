"""Cálculo del espectro de frecuencia (DFT) de una imagen.

Las redes generativas (GANs, difusión) dejan una firma de ruido de alta
frecuencia casi imperceptible al ojo humano pero muy marcada en el dominio
de Fourier (p. ej. patrones de checkerboard de las capas de upsampling).
Esta firma alimenta la rama de frecuencia del modelo híbrido (Fase 3).
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_frequency_spectrum(image_rgb: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Devuelve el espectro de magnitud log-escalado de `image_rgb`
    (HxWx3, uint8), normalizado a [0, 1], como un array HxW float32 de un
    solo canal.

    Se trabaja en escala de grises: el ruido de alta frecuencia que interesa
    detectar aparece igual de marcado en luminancia, así que no hace falta
    calcular la DFT por separado en R, G y B — eso solo triplicaría el costo
    sin aportar señal adicional relevante.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft, axes=(0, 1))  # componente DC al centro

    magnitude = cv2.magnitude(dft_shifted[..., 0], dft_shifted[..., 1])
    log_magnitude = np.log(magnitude + epsilon)

    normalized = np.empty_like(log_magnitude)
    cv2.normalize(log_magnitude, normalized, 0.0, 1.0, cv2.NORM_MINMAX)
    # cv2.normalize puede desbordar por unos pocos ULP de float32 fuera de
    # [0, 1] (p. ej. -1e-8); se recorta para que el contrato del rango de
    # salida sea exacto, ya que la rama de frecuencia (Fase 3) lo asume.
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
