import asyncio
import base64
import os
import uuid

from docarray import DocList
from docarray.documents import TextDoc

from marie import Client
from marie.executor.text import TextExtractionExecutorMock
from marie.storage import S3StorageHandler, StorageManager
from marie.utils.json import load_json_file, store_json_object
from marie_server.rest_extension import parse_payload_to_docs_sync


def setup_storage():
    handler = S3StorageHandler(
        config={
            "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
            "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
            "S3_STORAGE_BUCKET_NAME": "marie",
            "S3_ENDPOINT_URL": "http://localhost:8000",
            "S3_ADDRESSING_STYLE": "path",
        }
    )

    # export AWS_ACCESS_KEY_ID=MARIEACCESSKEY; export AWS_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY;  aws s3 ls --endpoint-url http://localhost:8000
    StorageManager.register_handler(handler=handler)
    StorageManager.ensure_connection()


def check_executor(img_path: str):
    img_path = os.path.expanduser(img_path)

    # Prepare file for upload
    with open(img_path, "rb") as file:
        encoded_bytes = base64.b64encode(file.read())
        base64_str = encoded_bytes.decode("utf-8")

    # Treat the image as a single word.
    # WORD = "word"
    # Sparse text. Find as much text as possible in no particular order.
    # SPARSE = "sparse"
    # Treat the image as a single text line.
    # LINE = "line"
    # Raw line. Treat the image as a single text line, NO bounding box detection performed.
    # RAW_LINE = "raw_line"
    # Multiline. Treat the image as multiple text lines, NO bounding box detection performed.
    # MULTI_LINE = "multiline"

    # Attributes
    # data[null]=> base 64 encoded image
    # mode['word]=> extraction mode
    # output['json']=> json,text,pdf

    uid = str(uuid.uuid4())

    payload = {
        "queue_id": uid,
        "data": base64_str,
        # "uri": "s3://marie/incoming/ocr-0001.tif",
        "mode": "sparse",
        "output": "json",
        "doc_id": f"exec-{uid}",
        "doc_type": "overlay",
        # "features": [{"type": "LABEL_DETECTION", "maxResults": 1}],
    }

    store_json_object(
        payload,
        os.path.expanduser(os.path.join("~/tmp/payloads", f"extract-{uid}.json")),
    )

    # load payload from file
    if False:
        payload = load_json_file(os.path.join("/tmp/payloads", "extract.json"))
        parameters, docs = parse_payload_to_docs_sync(payload)
        parameters["payload"] = payload  # THIS IS TEMPORARY HERE
        kwa = parameters

    parameters, docs = parse_payload_to_docs_sync(payload)
    parameters["payload"] = payload  # THIS IS TEMPORARY HERE
    kwa = parameters

    if True:
        executor = TextExtractionExecutorMock()
        results = executor.extract(docs, parameters=kwa)
        print(results)


async def check_executor_via_client():
    # load payload from file

    # payload = load_json_file(os.path.join("/tmp/payloads", "payload.json"))
    # parameters, docs = parse_payload_to_docs_sync(payload)
    parameters = {}
    parameters["payload"] = {"payload": "test"}  # THIS IS TEMPORARY HERE
    docs = DocList(TextDoc(text="test"))

    client = Client(
        host="0.0.0.0", port=52000, protocol="grpc", request_size=-1, asyncio=True
    )

    ready = await client.is_flow_ready()
    print(f"Flow is ready: {ready}")

    async for resp in client.post(
        "/document/classify",
        docs=docs,
        parameters=parameters,
        request_size=-1,
        return_responses=True,
        return_exceptions=True,
    ):
        print(resp)

    print("DONE")


if __name__ == "__main__":
    # setup_storage()
    # check_executor("~/tmp/4007/176080625.tif")

    if True:
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
