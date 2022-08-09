import warnings
from typing import Any

from flask import jsonify

from flask_restful import Resource, reqparse, request

from marie.api import extract_payload
from marie.conf.helper import storage_provider_config, executor_config
from marie.executor import NerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage

from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger
from marie.utils.image_utils import hash_file
from marie.utils.docs import docs_from_file
from marie import Document, DocumentArray

logger = default_logger


class NERRouter:
    """
    This is an adapter class to allow working with Flask
    """

    def __init__(self, app, config, **kwargs):
        warnings.warn(
            "NERRouter is deprecated; and will be replaced in version 2.6",
            DeprecationWarning,
        )
        if app is None:
            raise RuntimeError("Expected app arguments is null")

        if config is None:
            raise RuntimeError("Expected config arguments is null")

        _name_or_path = (
            kwargs.pop("_name_or_path")
            if "_name_or_path" in kwargs
            else executor_config(config, NerExtractionExecutor.__name__)[
                "_name_or_path"
            ]
        )

        if _name_or_path is None:
            raise RuntimeError("_name_or_path not provided in config or kwargs")

        prefix = "/api"

        self.executor = NerExtractionExecutor(_name_or_path)

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

        try:
            storage_conf = storage_provider_config("postgresql", config)
            self.storage = PostgreSQLStorage(
                hostname=storage_conf["hostname"],
                port=int(storage_conf["port"]),
                username=storage_conf["username"],
                password=storage_conf["password"],
                database=storage_conf["database"],
                table="ner_indexer",
            )
        except Exception as e:
            logger.warning("Storage config not set")

    def __store(self, ref_id: int, ref_type: str, results: Any) -> None:
        """Store results"""
        try:
            if self.storage is not None:
                dd = DocumentArray([Document(content=results)])
                self.storage.add(dd, {"ref_id": ref_id, "ref_type": ref_type})
        except Exception as e:
            print(e)
            logger.error("Unable to store document")

    def extract(self, queue_id: str):
        """Extract based on the supplied payload"""
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

            ref_id = str(payload["doc_id"]) if "doc_id" in payload else checksum
            ref_type = str(payload["doc_type"]) if "doc_type" in payload else ""

            self.__store(ref_id, ref_type, results)
            return jsonify(results), 200
        except BaseException as error:
            logger.error("Extract error", error)
            return {"error": error}, 200
