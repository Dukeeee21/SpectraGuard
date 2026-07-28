"""
URL raíz del proyecto. Las rutas de la API viven en apps/detection/urls.py
(Fase 4) y se montan aquí bajo el prefijo /api/v1/.
"""
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("apps.detection.urls")),
    # Documentación OpenAPI/Swagger autogenerada
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
]
