import pytest
from docarray import DocumentArray

from marie.helper import random_port
from marie.runtime import Client, Flow


@pytest.mark.parametrize('protocol', ['grpc', 'ws', 'http'])
def test_client_host_scheme(protocol):
    port = random_port()
    f = Flow(protocol='websocket' if protocol == 'ws' else protocol, port=port).add()
    with f:
        c = Client(host=f'{protocol}://localhost:{port}')
        c.post('/', inputs=DocumentArray.empty(2))
