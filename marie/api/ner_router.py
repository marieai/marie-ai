from enum import Enum

import torch
from flask import jsonify
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
from marie.executor import TextExtractionExecutor, NerExtractionExecutor
from marie.logging.logger import MarieLogger
from marie.utils.image_utils import hash_file
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

        self.executor = NerExtractionExecutor()
        prefix = "/api"

        app.add_url_rule(
            rule=f"/{prefix}/ner/info",
            endpoint="ner_info",
            view_func=self.executor.info,
            methods=["GET"],
        )

        app.add_url_rule(
            rule=f"{prefix}/ner/<queue_id>",
            endpoint="ner",
            view_func=self.extract,
            methods=["POST"],
        )

    def extract(self, queue_id: str):

        try:
            payload = request.json
            if payload is None:
                return {"error": "empty payload"}, 200

            tmp_file, checksum, file_type = extract_payload(payload, queue_id)
            img_path = tmp_file
            checksum = hash_file(img_path)
            docs = None  # docs_from_file(img_path)
            args = {"queue_id": queue_id, "checksum": checksum, "img_path": img_path}

            reply = self.executor.extract(docs, **args)
            logger.info("Raw reply")
            logger.info(logger)

            return jsonify(reply), 200
        except BaseException as error:
            logger.error("Extract error", error)
            return {"error": error}, 200
