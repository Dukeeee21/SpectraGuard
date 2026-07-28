"""
Configuración base de Django, compartida por development.py y production.py.

Por qué separar settings en base/development/production:
  - Evita `if DEBUG:` dispersos por todo el archivo de settings.
  - Permite que producción tenga valores estrictos (DEBUG=False, hosts
    restringidos) sin riesgo de que un desarrollador los sobreescriba
    accidentalmente en local.
"""

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

SECRET_KEY = env("DJANGO_SECRET_KEY", default="insecure-dev-key")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Terceros
    "rest_framework",
    "corsheaders",
    "drf_spectacular",
    # Apps propias
    "apps.detection",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --- Base de datos ---------------------------------------------------------
# PostgreSQL almacena el historial de análisis (AnalysisLog) y métricas del
# modelo (ModelMetrics) definidos en apps/detection/models.py (Fase 4).
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB", default="spectraguard"),
        "USER": env("POSTGRES_USER", default="spectraguard_user"),
        "PASSWORD": env("POSTGRES_PASSWORD", default=""),
        "HOST": env("POSTGRES_HOST", default="localhost"),
        "PORT": env("POSTGRES_PORT", default="5432"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "es-es"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Django REST Framework --------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.JSONParser",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "image-analysis": "30/minute",
    },
}

SPECTACULAR_SETTINGS = {
    "TITLE": "SpectraGuard API",
    "DESCRIPTION": "Detección de imágenes reales vs. generadas por IA (deepfakes, GANs, difusión).",
    "VERSION": "1.0.0",
}

# --- Configuración específica del modelo ML ---------------------------------
# Centralizada aquí (en vez de hardcodeada en views.py) para que el servicio
# de inferencia (apps/detection/services/inference.py, Fase 4) y los scripts
# de entrenamiento (ml/training/, Fase 3) lean la misma fuente de verdad.
ML_MODEL = {
    "WEIGHTS_PATH": BASE_DIR
    / env("MODEL_WEIGHTS_PATH", default="ml/weights/hybrid_detector_v1.pth"),
    "DEVICE": env("MODEL_DEVICE", default="cpu"),
    "IMAGE_SIZE": env.int("INFERENCE_IMAGE_SIZE", default=224),
}
