from marie import Client, Document, DocumentArray, Executor, Flow, requests


class FooExecutor(Executor):
    @requests(on="/foo")
    def foo(self, docs: DocumentArray, **kwargs):
        docs.append(Document(text='foo was called'))


# gateway --protocol http --discovery --discovery-host 127.0.0.1 --discovery-port 8500 --host 192.168.102.65 --port 5555
f = Flow(
    protocol='http',
    port=12345,
    discovery=True,
    discovery_host='127.0.0.1',
    discovery_port=8500,
    discovery_watchdog_interval=60,
).add(uses=FooExecutor)
# f.to_docker_compose_yaml('/tmp/docker-compose.yml')

with f:
    f.block()
    client = Client(port=12345, protocol='http')
    docs = client.post(on='/foo')
    print(docs.texts)
