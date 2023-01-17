from flask_restful import request

from marie.api import extract_payload
from marie.executor.text import TextExtractionExecutor
from marie.logging.logger import MarieLogger
from marie.utils.docs import docs_from_file

logger = MarieLogger("")


class ICRRouter:
    """
    This is an adapter class to allow working with Flask
    """

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        if app is None:
            raise RuntimeError("Expected app arguments is null")

        self.app = app
        self.executor = TextExtractionExecutor()

        prefix = "/api"
        app.add_url_rule(
            rule=f"{prefix}/info",
            endpoint="info",
            view_func=self.executor.info,
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

            tmp_file, checksum, file_type = extract_payload(payload, queue_id)
            docs = docs_from_file(tmp_file)

            args = {"queue_id": queue_id, "payload": payload}
            return self.executor.extract(docs, **args)
        except BaseException as error:
            logger.error("Extract error", error)
            return {"error": error}, 200
