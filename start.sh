#!/usr/bin/env bash

python manage.py collectstatic --noinput
python manage.py migrate

gunicorn FinalDataset.wsgi:application \
  --bind 0.0.0.0:$PORT
