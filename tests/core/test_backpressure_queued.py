import asyncio
import functools
import sys

from docarray import Document

from marie._core.utils import run_background_task

queue = asyncio.Queue()


async def coro_scheduler(limit: int = 2):
    print(f"scheduler limit: {limit}")
    pending = set()

    while True:
        print(f"pending before : {len(pending)} : {queue.qsize()}")
        while len(pending) < limit:
            item = queue.get()
            pending.add(asyncio.ensure_future(item))

        if not pending:
            continue

        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        while done:
            yield done.pop()
            # priority, item = val.result()
            # print(f"done : {priority}  > {item}")
            # await asyncio.sleep(0.5)


async def do_stuff(i):
    1 / i  # raises ZeroDivisionError for i == 0
    await asyncio.sleep(i)
    return i


def return_args_and_exceptions(func):
    return functools.partial(_return_args_and_exceptions, func)


async def _return_args_and_exceptions(func, *args):
    try:
        return *args, await func(*args)
    except Exception as e:
        return *args, e


async def async_consumer():
    print("async_consumer")
    print(f"async_consumer: {queue.qsize()} : {queue.empty()}")

    idx = 0
    async for item in coro_scheduler(limit=3):
        try:
            priority, item = item.result()
            retval = await item(idx)
            print(f"async_consumer: {queue.qsize()} : {queue.empty()} : item: {priority} -> {item} -> {retval}")

            idx += 1
        except Exception as e:
            raise e


async def async_main(args, limit: int):
    print("async_main")
    print(f"args: {args}")

    async def async_producer():
        print("async_inputs")
        for _ in range(20):
            print(f"async_inputs: {_} : {queue.qsize()} : {queue.empty()}")
            # yield Document(text=f"Doc_#{_}")
            priority = queue.qsize()
            item = (_, Document(text=f"Doc_#{_}"))
            item = (priority, return_args_and_exceptions(do_stuff))
            queue.put_nowait(item)
            # await asyncio.sleep(0.2)

        print("async_inputs: done")

    run_background_task(coroutine=async_consumer())
    run_background_task(coroutine=async_producer())


def main():
    limit = int(sys.argv[1])
    args = [float(n) for n in sys.argv[2:]]
    timeout = sum(args) + 0.5
    print("Main")
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(async_main(args, limit))
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()

    # asyncio.run(asyncio.wait_for(async_main(args, limit), timeout))


if __name__ == '__main__':
    main()

# ref : https://death.andgravity.com/limit-concurrency#aside-backpressure
# ref : https://bugs.python.org/issue30782
# ref : https://stackoverflow.com/questions/61474547/asyncio-limit-concurrency-with-async-generator
