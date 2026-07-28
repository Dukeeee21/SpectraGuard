#!/bin/sh
# Aplica migraciones y recolecta estáticos antes de arrancar gunicorn.
# Se ejecuta en cada arranque del contenedor "web" (ver docker-compose.yml).
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput

exec "$@"
