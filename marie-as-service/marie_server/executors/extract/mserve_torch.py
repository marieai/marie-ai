import uuid

import numpy as np
import torch
import asyncio
import marie
import marie.helper

from typing import Dict, Union, Optional, TYPE_CHECKING
from marie import Document, DocumentArray, Executor, requests
from marie.logging.predefined import default_logger
from marie import Client, DocumentArray
from fastapi import FastAPI, Request


from marie.api import extract_payload
from marie.utils.docs import docs_from_file, array_from_docs

# from marie.utils.network import get_ip_address


def extend_rest_interface_extract_mixin(app: FastAPI) -> None:
    c = Client(
        host='192.168.102.65', port=52000, protocol='grpc', request_size=1, asyncio=True
    )

    @app.get('/api/text/extractZZ', tags=['text', 'rest-api'])
    async def text_extract():
        default_logger.info("Executing text_extract")

        return {"message": "ABC"}

    @app.post('/api/text/extract-test', tags=['text', 'rest-api'])
    async def text_extract_post_test(request: Request):
        default_logger.info("Executing text_extract_post")
        payload = await request.json()
        print(payload.keys())
        print(request)
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
        async for resp in c.post('/text/extract', async_inputs, request_size=1):
            print('--' * 100)
            print(resp)
            print(resp.texts)
            out_text.append(resp.texts)
            print('++' * 100)
            outputs.append(resp)

        return {"message": f"ZZZ : {len(outputs)}", "out_text": out_text}

    @app.post('/api/extract', tags=['text', 'rest-api'])
    async def text_extract_post(request: Request):
        default_logger.info("Executing text_extract_post")
        try:
            payload = await request.json()
            print(payload.keys())
            print(request)
            queue_id = "0000-0000-0000-0000"

            tmp_file, checksum, file_type = extract_payload(payload, queue_id)
            input_docs = docs_from_file(tmp_file)
            out_docs = array_from_docs(input_docs)
            payload["data"] = None

            args = {
                "queue_id": queue_id,
                "payload": payload,
            }

            # {'data': ????, 'mode': 'multiline', 'output': 'json', 'doc_id': 'e8974900-0bee-4a9a-9c91-d8fdc909f446', 'doc_type': 'example_ner'

            print(">> ")
            print(args)
            outputs = DocumentArray()
            out_text = []
            async for resp in c.post(
                '/text/extract', input_docs, request_size=-1, parameters=args
            ):
                print('--' * 100)
                print(resp)
                print(resp.texts)
                out_text.append(resp.texts)
                print('++' * 100)
                outputs.append(resp)

            return {"message": f"ZZZ : {len(outputs)}", "out_text": out_text}
        except BaseException as error:
            default_logger.error("Extract error", error)
            return {"error": error}

    @app.get('/api/text/status', tags=['text', 'rest-api'])
    async def text_status():
        default_logger.info("Executing text_status")

        return {"message": "ABC"}
