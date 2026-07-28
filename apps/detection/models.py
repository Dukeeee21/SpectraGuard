"""Modelos PostgreSQL: historial de análisis y métricas del modelo.

No se guarda la imagen subida por el usuario: son fotos de rostros, y
persistirlas sin una política de retención/consentimiento explícita es un
riesgo de privacidad innecesario para lo que pide esta fase (auditoría de
predicciones, no un archivo de imágenes).
"""

from __future__ import annotations

from django.db import models


class AnalysisLog(models.Model):
    """Un registro por cada imagen analizada por `POST /api/v1/analyze/`."""

    class Label(models.TextChoices):
        REAL = "real", "Real"
        AI_GENERATED = "ai_generated", "Generada por IA"

    label = models.CharField(max_length=20, choices=Label.choices)
    confidence = models.FloatField(help_text="Confianza de la etiqueta predicha, en [0, 1].")
    ai_probability = models.FloatField(help_text="P(generado por IA), en [0, 1].")
    face_confidence = models.FloatField(help_text="Confianza de detección de rostro (MTCNN).")
    processing_time_ms = models.FloatField()
    model_version = models.CharField(max_length=50)
    client_ip = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["-created_at"])]

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.2%}) @ {self.created_at:%Y-%m-%d %H:%M}"


class ModelMetrics(models.Model):
    """Métricas de evaluación de una versión entrenada del modelo. Se cargan
    a mano (o desde un futuro job) después de cada corrida de
    `ml/training/train.py`, para llevar trazabilidad de qué tan bueno es
    cada checkpoint antes de promoverlo a producción."""

    model_version = models.CharField(max_length=50, unique=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    dataset_size = models.PositiveIntegerField()
    evaluated_at = models.DateTimeField()
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-evaluated_at"]
        verbose_name_plural = "model metrics"

    def __str__(self) -> str:
        return f"{self.model_version} (acc={self.accuracy:.2%})"
