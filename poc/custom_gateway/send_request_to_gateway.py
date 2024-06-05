import asyncio

from docarray import DocList
from docarray.documents import TextDoc

from marie import Client


async def main():
    """
    This function sends a request to a Marie server gateway.
    """
    parameters = {}
    parameters["payload"] = {"payload": "sample payload"}
    docs = DocList[TextDoc]([TextDoc(text="Sample Text")])

    client = Client(
        host="127.0.0.1", port=52000, protocol="grpc", request_size=-1, asyncio=True
    )

    ready = await client.is_flow_ready()
    print(f"Flow is ready: {ready}")

    async for resp in client.post(
        on="/",
        inputs=docs,
        parameters=parameters,
        request_size=-1,
        return_responses=True,  # return DocList instead of Response
        return_exceptions=True,
    ):
        print(resp)
        # for doc in resp:
        #     print(doc.text)

        print(resp.parameters)
        print(resp.data)
        # await asyncio.sleep(1)

    print("DONE")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
