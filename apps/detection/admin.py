"""Registro de los modelos de detección en el admin de Django."""

from __future__ import annotations

from django.contrib import admin
from django.http import HttpRequest

from apps.detection.models import AnalysisLog, ModelMetrics


@admin.register(AnalysisLog)
class AnalysisLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "label",
        "confidence",
        "ai_probability",
        "model_version",
        "created_at",
    )
    list_filter = ("label", "model_version")
    ordering = ("-created_at",)

    def has_add_permission(self, request: HttpRequest) -> bool:
        # Los registros solo los crea el endpoint de análisis, no un humano.
        return False


@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = (
        "model_version",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "evaluated_at",
    )
    ordering = ("-evaluated_at",)
