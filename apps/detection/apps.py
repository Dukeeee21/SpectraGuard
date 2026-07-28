from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.detection"
    verbose_name = "AI Image Detection"

    def ready(self) -> None:
        from apps.detection.services import inference

        inference.load_predictor()
