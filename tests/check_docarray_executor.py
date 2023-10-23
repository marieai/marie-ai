import time
from marie import Executor, requests, Flow
from docarray import DocList
from docarray.documents import TextDoc


class FastChangingExecutor(Executor):
    @requests()
    def foo(
        self, docs: DocList[TextDoc], parameters: dict, **kwargs
    ) -> DocList[TextDoc]:
        for doc in docs:
            doc.text = "Hello World"


class SlowChangingExecutor(Executor):
    @requests()
    def foo(
        self, docs: DocList[TextDoc], parameters: dict, **kwargs
    ) -> DocList[TextDoc]:
        time.sleep(2)
        print(f" Received {docs.text}")
        for doc in docs:
            doc.text = "Change the document but will not affect response"


f = (
    Flow()
    .add(name="executor0", uses=FastChangingExecutor)
    .add(
        name="floating_executor",
        uses=SlowChangingExecutor,
        needs=["gateway"],
        floating=True,
    )
)
with f:
    f.post(
        on="/endpoint",
        inputs=DocList[TextDoc]([TextDoc()]),
        return_type=DocList[TextDoc],
    )  # we need to send a first
    start_time = time.time()
    response = f.post(
        on="/endpoint",
        inputs=DocList[TextDoc]([TextDoc(), TextDoc()]),
        return_type=DocList[TextDoc],
    )
    end_time = time.time()
    print(f" Response time took {end_time - start_time}s")
    print(f" {response.text}")
