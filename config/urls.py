"""
URL raíz del proyecto. Las rutas de la API viven en apps/detection/urls.py
(Fase 4) y se montan aquí bajo el prefijo /api/v1/.
"""

from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    # Este backend no tiene UI propia (Fase 4 es solo API): mandar la raíz a
    # la documentación interactiva en vez de dejar un 404 confuso.
    path("", RedirectView.as_view(url="/api/docs/", permanent=False)),
    path("admin/", admin.site.urls),
    path("api/v1/", include("apps.detection.urls")),
    # Documentación OpenAPI/Swagger autogenerada
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
]
