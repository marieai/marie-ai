from marie import Executor, requests


class SampleRouter(Executor):
    def __init__(self, app, **kwargs):
        if app is None:
            raise RuntimeError("Expected app arguments is null")
        prefix = "/api"
        app.add_url_rule(rule=f"{prefix}/info", endpoint="info", view_func=self.info, methods=["GET"])
        app.add_url_rule(rule=f"{prefix}/status/<queue_id>", endpoint="status", view_func=self.status, methods=["GET"])

    # @requests('xx')
    def status(self, queue_id):
        print(f"Self : {self}")
        return {"routed": f"reply={queue_id}"}

    def info(self):
        print(f"Self : {self}")
        return {"index": "complete"}
