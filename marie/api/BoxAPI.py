from flask_restful import Resource, reqparse

import marie.executor
from marie.boxes import box_processor
from marie.utils.utils import current_milli_time


class BoxListAPI(Resource):
    def __init__(self):
        super(BoxListAPI, self).__init__()
        self.reqparse = reqparse.RequestParser()
        self.box_processor = marie.processors.box_processor
        print("BoxListAPI inited")

    def get(self):
        """Get boxes"""
        print(box_processor)
        return {"boxes": current_milli_time()}, 200

    def post(self):
        pass


class BoxAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        super(BoxAPI, self).__init__()

        print("BoxAPI inited")

    def get(self, uid):
        return {"box_id": id}, 200

    def put(self, uid):
        pass

    def delete(self, uid):
        pass
