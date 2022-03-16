import conf
from celery import Celery


def make_celery():
   celery = Celery(__name__, broker=conf.CELERY_BROKER)
   celery.conf.update(conf.as_dict())
   return celery


celery = make_celery()
