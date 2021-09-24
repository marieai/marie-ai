from api.QueueAPI import QueueAPI, QueueListAPI
import time
from flask import jsonify
from flask_restful import Api, Resource
from tasks import celery
import config

api = Api(prefix=config.API_PREFIX)

api.add_resource(QueueListAPI, '/queues', endpoint = 'queues')
api.add_resource(QueueAPI, '/queues/<int:id>', endpoint = 'queue')
# api.add_resource(QueueAPI, '/queues/<string:queue_id>/uploads')
