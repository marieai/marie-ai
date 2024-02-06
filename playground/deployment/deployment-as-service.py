from docarray import BaseDoc, DocList

from marie import Deployment, Executor, requests


class MyExecutor(Executor):
    @requests(on='/bar')
    def foo(self, docs: DocList[BaseDoc], **kwargs) -> DocList[BaseDoc]:
        print(docs)


dep = Deployment(port=12345, name='myexec1', uses=MyExecutor, replicas=1)

with dep:
    dep.block()
