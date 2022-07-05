from enum import Enum

import torch
from torch.backends import cudnn

from marie import Executor, requests

import io
import os

import hashlib
import imghdr
import numpy as np

import cv2

from flask_restful import Resource, reqparse, request

from marie.api import extract_payload
from marie.executor import TextExtractionExecutor
from marie.logging.logger import MarieLogger
from marie.utils.utils import FileSystem, ensure_exists, current_milli_time
from marie.utils.base64 import base64StringToBytes, encodeToBase64
from marie.utils.docs import docs_from_file

from datetime import datetime

logger = MarieLogger("")


class NERRouter:
    """
    This is an adapter class to allow working with Flask
    """

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        if app is None:
            raise RuntimeError("Expected app arguments is null")

        self.executor = TextExtractionExecutor()

        prefix = "/api"
        app.add_url_rule(
            rule=f"{prefix}/info",
            endpoint="info",
            view_func=self.executor.info,
            methods=["GET"],
        )
        # app.add_url_rule(rule="/status/<queue_id>", endpoint="status", view_func=self.status, methods=["GET"])
        app.add_url_rule(
            rule=f"/{prefix}/status",
            endpoint="status",
            view_func=self.executor.status,
            methods=["GET"],
        )
        app.add_url_rule(
            rule=f"/{prefix}",
            endpoint="api_index",
            view_func=self.executor.status,
            methods=["GET"],
        )
        app.add_url_rule(
            rule=f"{prefix}/extract/<queue_id>",
            endpoint="extract",
            view_func=self.extract,
            methods=["POST"],
        )

    def extract(self, queue_id: str):

        try:
            payload = request.json
            if payload is None:
                return {"error": "empty payload"}, 200
            args = {"queue_id": queue_id, "payload": payload}

            tmp_file, checksum, file_type = extract_payload(payload, queue_id)
            docs = docs_from_file(tmp_file)

            return self.executor.extract(docs, **args)
        except BaseException as error:
            logger.error("Extract error", error)
            return {"error": error}, 200
