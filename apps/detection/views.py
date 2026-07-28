"""Endpoint `POST /api/v1/analyze/`: recibe una imagen y devuelve la
predicción del modelo híbrido (real vs. generada por IA).
"""

from __future__ import annotations

from PIL import Image
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.detection.models import AnalysisLog
from apps.detection.serializers import AnalysisResultSerializer, ImageAnalysisRequestSerializer
from apps.detection.services import inference
from ml.preprocessing.face_extractor import NoFaceDetectedError


class ImageAnalysisView(APIView):
    """`POST /api/v1/analyze/` — multipart/form-data con un campo `image`."""

    throttle_scope = "image-analysis"

    def post(self, request: Request) -> Response:
        predictor = inference.get_predictor()
        if predictor is None:
            return Response(
                {
                    "detail": "El modelo todavía no está disponible: falta "
                    "entrenar y desplegar un checkpoint. "
                    f"({inference.get_load_error()})"
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        request_serializer = ImageAnalysisRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        uploaded_image = request_serializer.validated_data["image"]

        # ImageField ya validó que es una imagen decodificable; solo hay que
        # rebobinar el stream, que la validación de DRF/Django ya consumió.
        uploaded_image.seek(0)
        pil_image = Image.open(uploaded_image)

        try:
            result = predictor.predict(pil_image)
        except NoFaceDetectedError:
            return Response(
                {"detail": "No se detectó ningún rostro en la imagen."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        log_entry = AnalysisLog.objects.create(
            label=result.label,
            confidence=result.confidence,
            ai_probability=result.ai_probability,
            face_confidence=result.face_confidence,
            processing_time_ms=result.processing_time_ms,
            model_version=inference.get_model_version(),
            client_ip=self._client_ip(request),
        )

        return Response(AnalysisResultSerializer(log_entry).data, status=status.HTTP_200_OK)

    @staticmethod
    def _client_ip(request: Request) -> str | None:
        forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR")
