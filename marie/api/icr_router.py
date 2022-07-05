import json

from flask import jsonify, url_for
from flask_restful import Resource, reqparse, request

from marie.api import extract_payload
from marie.executor import TextExtractionExecutor
from marie.logging.logger import MarieLogger
from marie.utils.docs import docs_from_file
from marie.utils.network import get_ip_address

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

        # app.add_url_rule(rule="/status/<queue_id>", endpoint="status", view_func=self.status, methods=["GET"])
        app.add_url_rule(
            rule=f"/{prefix}",
            endpoint="status",
            view_func=self.status,
            methods=["GET"],
        )

        app.add_url_rule(
            rule=f"{prefix}/extract/<queue_id>",
            endpoint="extract",
            view_func=self.extract,
            methods=["POST"],
        )

    def list_routes(self):
        import urllib

        output = []
        app = self.app
        for rule in app.url_map.iter_rules():

            options = {}
            for arg in rule.arguments:
                options[arg] = "[{0}]".format(arg)

            methods = ",".join(rule.methods)
            url = url_for(rule.endpoint, **options)
            line = "{:50s} {:20s} {}".format(rule.endpoint, methods, url)
            output.append(line)

        for line in sorted(output):
            print(line)

        return output

    def status(self):
        """Get application status"""
        import os

        build = {}
        if os.path.exists(".build"):
            with open(".build", "r") as fp:
                build = json.load(fp)
        host = get_ip_address()
        routes = self.list_routes()

        return (
            jsonify(
                {
                    "name": "marie-icr",
                    "host": host,
                    "component": [
                        {"name": "craft", "version": "1.0.0"},
                        {"name": "craft-benchmark", "version": "1.0.0"},
                    ],
                    "build": build,
                    "routes": routes,
                }
            ),
            200,
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
