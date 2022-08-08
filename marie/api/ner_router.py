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
from marie.logging.predefined import default_logger
from marie.utils.image_utils import hash_file
from marie.utils.docs import docs_from_file

logger = default_logger


class NERRouter:
    """
    This is an adapter class to allow working with Flask
    """

    def __init__(self, app, **kwargs):
        if app is None:
            raise RuntimeError("Expected app arguments is null")

        if "_name_or_path" not in kwargs:
            raise RuntimeError("Missing attribute : _name_or_path")
        _name_or_path = kwargs.pop("_name_or_path")

        self.executor = NerExtractionExecutor(_name_or_path)
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
            checksum = hash_file(tmp_file)
            docs = docs_from_file(tmp_file)
            kwargs = {"queue_id": queue_id, "checksum": checksum, "img_path": tmp_file}

            results = self.executor.extract(docs, **kwargs)
            logger.info("Raw reply")
            logger.info(results)

            return jsonify(results), 200
        except BaseException as error:
            logger.error("Extract error", error)
            return {"error": error}, 200
