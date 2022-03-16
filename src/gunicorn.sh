#!/usr/bin/env bash
# gunicorn wsgi:app -w 2 -b 0.0.0.0:5000
gunicorn -c gunicorn.conf.py wsgi:app  --log-level=debug