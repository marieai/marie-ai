import math
import sys
import time
import asyncio
import functools
import itertools

from docarray import Document

queue = asyncio.Queue()


async def scheduler(limit: int):
    print(f"scheduler limit: {limit}")
    pending = set()

    while True:
        while len(pending) < limit:
            item = queue.get()
            pending.add(asyncio.ensure_future(item))

        await asyncio.sleep(0.5)
        print(f"pending: {len(pending)}")
        if not pending:
            continue

        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        while done:
            yield done.pop()


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


async def async_main(args, limit: int):
    async def async_inputs():
        for _ in range(2):
            yield Document(text=f"Doc_#{_}")
            await asyncio.sleep(0.2)

    # schedule the async inputs in the queue via background task
    asyncio.create_task(
        queue.put(async_inputs()), name="async_inputs_task"
    )

    start = time.monotonic()
    wrapped = return_args_and_exceptions(do_stuff)
    print(f"{(time.monotonic() - start):.1f}: done")

    scheduler(limit)


def main():
    limit = int(sys.argv[1])
    args = [float(n) for n in sys.argv[2:]]
    timeout = sum(args) + 0.5
    asyncio.run(asyncio.wait_for(async_main(args, limit), timeout))


if __name__ == '__main__':
    main()

# ref : https://death.andgravity.com/limit-concurrency#aside-backpressure
# ref : https://bugs.python.org/issue30782
# ref : https://stackoverflow.com/questions/61474547/asyncio-limit-concurrency-with-async-generator
