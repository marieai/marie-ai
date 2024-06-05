from datetime import datetime

from docarray import DocList
from docarray.documents import TextDoc
from fastapi.encoders import jsonable_encoder

import marie
import marie.helper
from marie import Document, DocumentArray
from marie import Gateway as BaseGateway
from marie.serve.runtimes.servers.composite import CompositeServer


class MarieServerGateway(BaseGateway, CompositeServer):
    """A custom Gateway for Marie server.
    Effectively we are providing a custom implementation of the Gateway class that providers communication between individual executors and the server.

    This utilizes service discovery to find deployed Executors from discovered gateways that could have spawned them(Flow/Deployment).

    """

    def __init__(self, **kwargs):
        """Initialize a new Gateway."""
        super().__init__(**kwargs)

        def _extend_rest_function(app):
            @app.get("/endpoint")
            async def get(text: str):
                print(f"Received request at {datetime.now()}")

                result = None
                async for docs in self.streamer.stream_docs(
                    docs=DocList[TextDoc]([TextDoc(text=text)]),
                    exec_endpoint="/",
                    # exec_endpoint="/endpoint",
                    # target_executor="executor0",
                    return_results=False,
                ):
                    result = docs[0].text
                return {"result": result}

            return app

        marie.helper.extend_rest_interface = _extend_rest_function
