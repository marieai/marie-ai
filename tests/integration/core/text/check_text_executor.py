import asyncio
import os

from marie import Client
from marie.executor.ner.utils import visualize_icr
from marie.executor.text import TextExtractionExecutor
from marie.renderer import PdfRenderer, TextRenderer
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.docs import array_from_docs, docs_from_file
from marie.utils.json import load_json_file, store_json_object
from marie.utils.utils import ensure_exists
from marie_server.rest_extension import (
    parse_payload_to_docs,
    parse_payload_to_docs_sync,
)


def setup_storage():
    handler = S3StorageHandler(
        config={
            "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
            "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
            "S3_STORAGE_BUCKET_NAME": "marie",
            "S3_ENDPOINT_URLXX": "http://localhost:8000",
            "S3_ENDPOINT_URL": "http://64.62.141.143:8000",
            "S3_ADDRESSING_STYLE": "path",
        }
    )

    # export AWS_ACCESS_KEY_ID=MARIEACCESSKEY; export AWS_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY;  aws s3 ls --endpoint-url http://localhost:8000
    StorageManager.register_handler(handler=handler)
    StorageManager.ensure_connection()


def check_executor():
    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")
    ensure_exists("/tmp/fragments")

    img_path = "/home/gbugaj/tmp/marie-cleaner/169150505/PID_1898_9172_0_169150505.tif"

    docs = docs_from_file(img_path)
    frames = array_from_docs(docs)
    kwa = {"payload": {"output": "json", "mode": "line", "format": "xyxy"}}
    kwa = {"payload": {"output": "json", "mode": "sparse", "format": "xywh"}}
    # kwa = {"payload": {"output": "json", "mode": "line"}}

    # load payload from file
    payload = load_json_file(os.path.join("/tmp/payloads", "extract.json"))
    parameters, docs = parse_payload_to_docs_sync(payload)
    parameters["payload"] = payload  # THIS IS TEMPORARY HERE
    kwa = parameters

    if True:
        executor = TextExtractionExecutor()
        results = executor.extract(docs, parameters=kwa)


async def check_executor_via_client():
    # load payload from file
    payload = load_json_file(os.path.join("/tmp/payloads", "payload.json"))
    parameters, docs = parse_payload_to_docs_sync(payload)
    parameters["payload"] = payload  # THIS IS TEMPORARY HERE

    client = Client(
        host="0.0.0.0", port=52000, protocol="grpc", request_size=1, asyncio=True
    )

    async for resp in client.post(
        '/text/extract',
        docs=docs,
        parameters=parameters,
        request_size=-1,
        return_responses=True,
    ):
        print(resp)

    print("DONE")


if __name__ == "__main__":
    setup_storage()
    # check_executor()

    print("Main")
    loop = asyncio.get_event_loop()
    try:
        # asyncio.ensure_future(main_single())
        asyncio.ensure_future(check_executor_via_client())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
