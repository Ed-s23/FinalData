#!/bin/bash

echo "PORT recibido: $PORT"

exec gunicorn FinalDataset.wsgi:application \
  --bind 0.0.0.0:${PORT} \
  --workers 1 \
  --threads 2 \
  --timeout 120
