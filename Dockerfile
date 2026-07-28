FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias de sistema requeridas por psycopg2 (libpq-dev) y opencv-python-headless
# (libglib2.0-0, libsm6, libxext6, libxrender1) para decodificar/procesar imágenes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # insightface declara "opencv-python" (con GUI) como dependencia y pip
    # lo instala encima del opencv-python-headless de requirements.txt. La
    # imagen slim no tiene libGL.so.1 (no instalado a propósito, no hace
    # falta para uso headless), así que si el paquete con GUI queda instalado
    # "import cv2" falla en runtime. Se fuerza headless después, sin deps.
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --no-deps --force-reinstall opencv-python-headless==4.10.0.84

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
