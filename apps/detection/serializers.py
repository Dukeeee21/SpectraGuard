"""Serializers del endpoint de análisis: validan la imagen de entrada y dan
forma a la respuesta de predicción.
"""

from __future__ import annotations

from typing import Any

from rest_framework import serializers

from apps.detection.models import AnalysisLog

_MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class ImageAnalysisRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()

    def validate_image(self, value: Any) -> Any:
        if value.size > _MAX_UPLOAD_SIZE_BYTES:
            raise serializers.ValidationError(
                f"La imagen supera el tamaño máximo permitido de "
                f"{_MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            )
        if value.content_type not in _ALLOWED_CONTENT_TYPES:
            raise serializers.ValidationError(
                f"Formato no soportado ({value.content_type}). "
                f"Usar uno de: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}."
            )
        return value


class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisLog
        fields = [
            "id",
            "label",
            "confidence",
            "ai_probability",
            "face_confidence",
            "processing_time_ms",
            "model_version",
            "created_at",
        ]
        read_only_fields = fields
