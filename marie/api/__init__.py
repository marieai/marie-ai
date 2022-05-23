import marie.conf
from marie.api.BoxAPI import BoxAPI, BoxListAPI
from marie.api.MarkAPI import MarkAPI, MarkListAPI
from marie.api.QueueAPI import QueueAPI, QueueListAPI
from marie.api.SegmenterAPI import SegmenterAPI, SegmenterListAPI
from flask_restful import Api

api = Api(prefix=marie.conf.API_PREFIX)  # AttributeError: module 'config' has no attribute 'API_PREFIX
