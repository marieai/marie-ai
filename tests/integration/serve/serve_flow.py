from marie import Client, Executor, Flow, requests, DocumentArray, Document


class FooExecutor(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        docs.append(Document(text='foo was called'))


f = Flow(protocol='http', port=12345).add(uses=FooExecutor)
with f:
    f.block()
    # client = Client(port=12345)
    # docs = client.post(on='/')
    # print(docs.texts)
