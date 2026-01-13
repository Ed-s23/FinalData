#!/bin/bash

echo "PORT usado: $PORT"

export DJANGO_SETTINGS_MODULE=FinalDataset.settings

exec gunicorn FinalDataset.wsgi:application \
  --bind 0.0.0.0:${PORT} \
  --workers 1 \
  --threads 2 \
  --timeout 120
