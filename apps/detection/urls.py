"""Rutas del endpoint POST /api/v1/analyze/"""

from __future__ import annotations

from django.urls import path

from apps.detection.views import ImageAnalysisView

app_name = "detection"

urlpatterns = [
    path("analyze/", ImageAnalysisView.as_view(), name="analyze"),
]
