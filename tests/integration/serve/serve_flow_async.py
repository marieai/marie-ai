from marie import Client, Executor, Flow, requests, DocumentArray, Document
import asyncio


from docarray import DocumentArray, Document


class MyExec(Executor):
    @requests(on='/foo')
    def foo(self, docs: DocumentArray, **kwargs):
        docs[0] = 'executed MyExec'  # custom logic goes here


flow = Flow(port=[12345, 12344, 12343], protocol=['grpc', 'http', 'websocket']).add(
    uses=MyExec
)
flow.expose_endpoint('/foo', summary='my endpoint')

with flow:
    flow.block()

# MyExec.serve(port=12345, protocol='http')
