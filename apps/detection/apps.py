from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.detection"
    verbose_name = "AI Image Detection"

    def ready(self) -> None:
        # Fase 4: aquí se cargará el modelo PyTorch preentrenado en memoria
        # una sola vez al arrancar Django (evita recargarlo en cada request).
        pass
