import asyncio

from docarray import DocList
from docarray.documents import TextDoc

from marie import Client


async def check_executor_via_client():
    parameters = {}
    parameters["payload"] = {"payload": "test"}  # THIS IS TEMPORARY HERE
    docs = DocList(TextDoc(text="test"))

    client = Client(
        host="192.168.1.11", port=52000, protocol="grpc", request_size=-1, asyncio=True
    )

    ready = await client.is_flow_ready()
    print(f"Flow is ready: {ready}")

    async for resp in client.post(
        on="/document/extract",
        docs=docs,
        parameters=parameters,
        request_size=-1,
        return_responses=True,
        return_exceptions=True,
    ):
        print(resp)

    print("DONE")


if __name__ == "__main__":
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
