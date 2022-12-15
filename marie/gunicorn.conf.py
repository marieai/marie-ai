"""gunicorn WSGI server configuration."""
from multiprocessing import cpu_count
from os import environ


def max_workers():
    return 1  # cpu_count()


bind = '0.0.0.0:' + environ.get('PORT', '5000')
max_requests = 1000
worker_class = 'gevent'
# worker_class = 'sync' # Triggers ::  gunicorn CRITICAL WORKER TIMEOUT
workers = max_workers()

raw_env = ["MARIE_CONFIGURATION=production"]
