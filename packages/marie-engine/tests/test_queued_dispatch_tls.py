"""Real TLS, SNI and numeric pinning using generated test-only certificates."""

import asyncio
import datetime
import json
import socket
import ssl

import pytest


@pytest.mark.parametrize(
    'trust,host,expected',
    [
        (True, 'fixture.invalid', ''),
        (True, 'wrong.invalid', 'connect_error'),
        (False, 'fixture.invalid', 'connect_error'),
    ],
)
async def test_real_tls_origin_validation_and_pinning(
    tmp_path, monkeypatch, trust, host, expected
):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'fixture.invalid')])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName('fixture.invalid')]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / 'synthetic.certificate'
    key_path = tmp_path / 'synthetic.private-material'
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    sni, hosts, pinned = [], [], []
    context.set_servername_callback(lambda connection, name, context: sni.append(name))
    active = set()

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            hosts.append(headers.decode().split('Host: ', 1)[1].split('\r\n')[0])
            size = int(
                headers.decode().split('Content-Length: ', 1)[1].split('\r\n')[0]
            )
            await reader.readexactly(size)
            body = json.dumps(
                {'choices': [{'message': {'content': 'synthetic'}}]}
            ).encode()
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Length: '
                + str(len(body)).encode()
                + b'\r\nConnection: close\r\n\r\n'
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0, ssl=context)
    port = server.sockets[0].getsockname()[1]
    if trust:
        monkeypatch.setenv('SSL_CERT_FILE', str(cert_path))
    else:
        monkeypatch.delenv('SSL_CERT_FILE', raising=False)
        monkeypatch.delenv('SSL_CERT_DIR', raising=False)
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:1')
    loop = asyncio.get_running_loop()

    async def resolve(host, port, **kwargs):
        return socket.getaddrinfo('127.0.0.1', port, **kwargs)

    monkeypatch.setattr(loop, 'getaddrinfo', resolve)
    client = EndpointClient(
        RegisteredEndpoint('tls', f'https://{host}:{port}/v1', allow_loopback=True)
    )
    backend = client.client._transport._pool._network_backend.backend
    original_connect = backend.connect_tcp

    async def connect(host, *args, **kwargs):
        pinned.append(host)
        return await original_connect(host, *args, **kwargs)

    monkeypatch.setattr(backend, 'connect_tcp', connect)
    try:
        outcome = await client.execute(
            CompletionCallParams(model='model', messages=[]), timeout_seconds=2
        )
        assert outcome.category == expected
        assert pinned == ['127.0.0.1']
        assert sni == [host]
        assert hosts == (
            [f'{host}:{port}'] if trust and host == 'fixture.invalid' else []
        )
        assert (outcome.response is not None) == (expected == '')
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*tuple(active), return_exceptions=True)


async def test_redirect_never_forwards_registered_credentials():
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    forwarded = []
    source_auth = []
    active = set()

    async def target(reader, writer):
        forwarded.append(True)
        writer.close()
        await writer.wait_closed()

    target_server = await asyncio.start_server(target, '127.0.0.1', 0)
    target_port = target_server.sockets[0].getsockname()[1]

    async def source(reader, writer):
        active.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            source_auth.append(
                b'Authorization: Bearer synthetic-test-token\r\n' in headers
            )
            writer.write(
                f'HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{target_port}/escape\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'.encode()
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    source_server = await asyncio.start_server(source, '127.0.0.1', 0)
    client = EndpointClient(
        RegisteredEndpoint(
            'redirect',
            f'http://127.0.0.1:{source_server.sockets[0].getsockname()[1]}',
            api_key='synthetic-test-token',
            allow_loopback=True,
        )
    )
    try:
        result = await client.execute(
            CompletionCallParams(model='model', messages=[]), timeout_seconds=1
        )
        assert result.category == 'provider_rejected'
        assert source_auth == [True]
        assert not forwarded
    finally:
        await client.close()
        source_server.close()
        target_server.close()
        await source_server.wait_closed()
        await target_server.wait_closed()
        await asyncio.gather(*tuple(active), return_exceptions=True)
