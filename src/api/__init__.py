import conf
from api.BoxAPI import BoxAPI, BoxListAPI
from api.MarkAPI import MarkAPI, MarkListAPI
from api.QueueAPI import QueueAPI, QueueListAPI
from api.SegmenterAPI import SegmenterAPI, SegmenterListAPI
from flask_restful import Api


api = Api(prefix=conf.API_PREFIX) # AttributeError: module 'config' has no attribute 'API_PREFIX

# # Queues
# api.add_resource(QueueListAPI, '/queues', endpoint='queues')
# api.add_resource(QueueAPI, '/queues/<int:id>', endpoint='queue')

# # Boxes
# api.add_resource(BoxListAPI, '/boxes', endpoint='boxes')
# api.add_resource(BoxAPI, '/box/<int:id>', endpoint='box')

# # # Marks - Inteligent Mark Recognition
# # api.add_resource(MarkListAPI, '/marks', endpoint='marks')
# # api.add_resource(MarkAPI, '/mark/<int:id>', endpoint='mark')


# # ICR
# api.add_resource(IcrExtractAPI, '/extract/<string:queue_id>',
#                  endpoint='extract')

# # Segmenter
# api.add_resource(SegmenterListAPI, '/segmenters', endpoint='segmenters')
# api.add_resource(SegmenterAPI, '/segmenter/<int:id>', endpoint='segmenter')

# Layout

# api.add_resource(QueueAPI, '/layout/<string:queue_id>/uploads')
