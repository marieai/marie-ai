from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, Flow, requests
from marie.serve.runtimes.gateway.http.fastapi import FastAPIBaseGateway


class MyGateway(FastAPIBaseGateway):
    @property
    def app(self):
        from fastapi import FastAPI

        app = FastAPI(title="Custom FastAPI Gateway")

        @app.get("/endpoint")
        async def get(text: str):
            result = None
            async for docs in self.streamer.stream_docs(
                docs=DocList[TextDoc]([TextDoc(text=text)]),
                exec_endpoint="/",
                target_executor="executor0",
            ):
                result = docs[0].text
            return {"result": result}

        return app


class FirstExec(Executor):
    @requests
    def func(self, docs, **kwargs):
        for doc in docs:
            doc.text += " First"


class SecondExec(Executor):
    @requests
    def func(self, docs, **kwargs):
        for doc in docs:
            doc.text += " Second"


with Flow(port=12345).config_gateway(uses=MyGateway, protocol="http", port=50975).add(
    uses=FirstExec, name="executor0"
).add(uses=SecondExec, name="executor1") as flow:
    flow.block()

# curl -X GET "http://localhost:50975/endpoint?text=abc"
