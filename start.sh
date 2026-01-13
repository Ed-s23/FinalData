#!/bin/bash

echo "Starting Django app on port $PORT"

gunicorn FinalDataset.wsgi:application \
  --bind 0.0.0.0:$PORT \
  --workers 1 \
  --threads 2 \
  --timeout 120
