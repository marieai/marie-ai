import asyncio
import uuid

from fastapi import FastAPI, Request, HTTPException
from marie import Client, DocumentArray
from marie import Document
from marie.logging.predefined import default_logger
from marie_server.rest_extension import (
    parse_response_to_payload,
    handle_request,
)

extract_flow_is_ready = False


def extend_rest_interface_extract(app: FastAPI, client: Client) -> None:
    """
    Extends HTTP Rest endpoint to provide compatibility with existing REST endpoints
    :param client:
    :param app:
    :return:
    """

    @app.post('/api/text/extract-test', tags=['text', 'rest-api'])
    async def text_extract_post_test(request: Request):
        default_logger.info("Executing text_extract_post")
        payload = await request.json()
        print(payload.keys())
        inputs = DocumentArray.empty(6)

        # async inputs for the client
        async def async_inputs():
            uuid_val = uuid.uuid4()
            for _ in range(2):
                yield Document(text=f'Doc_{uuid_val}#{_}')
                await asyncio.sleep(0.2)

        # {'data': ????, 'mode': 'multiline', 'output': 'json', 'doc_id': 'e8974900-0bee-4a9a-9c91-d8fdc909f446', 'doc_type': 'example_ner'

        print(">> ")
        outputs = DocumentArray()
        out_text = []
        async for resp in client.post('/text/extract', async_inputs, request_size=1):
            print('--' * 100)
            print(resp)
            print(resp.texts)
            out_text.append(resp.texts)
            print('++' * 100)
            outputs.append(resp)

        return {"message": f"ZZZ : {len(outputs)}", "out_text": out_text}

    @app.get('/api/extract', tags=['text', 'rest-api'])
    async def text_extract_get(request: Request):
        default_logger.info("Executing text_extract_get")
        return {"message": "reply"}

    async def __process(client: Client, input_docs, parameters):
        payload = {}
        async for resp in client.post(
            '/text/extract',
            input_docs,
            request_size=-1,
            parameters=parameters,
            return_responses=True,
        ):
            print("Parsing response to payload")
            payload = parse_response_to_payload(resp, expect_return_value=False)
            break  # we only need the first response

        return payload

    @app.post('/api/extract', tags=['text', 'rest-api'])
    async def text_extract_post(request: Request):
        """
        Handle API Extract endpoint
        :param request:
        :return:
        """
        global extract_flow_is_ready
        if not extract_flow_is_ready and not await client.is_flow_ready():
            raise HTTPException(status_code=503, detail="Flow is not yet ready")
        extract_flow_is_ready = True
        return await handle_request("extract", request, client, __process)

    @app.get('/api/text/status', tags=['text', 'rest-api'])
    async def text_status():
        default_logger.info("Executing text_status")
        return {"status": "OK"}
