import sys
import time
import asyncio
import functools
import itertools


async def map_unordered(func, iterable, *, limit):
    try:
        aws = map(func, iterable)
    except TypeError:
        aws = (func(x) async for x in iterable)

    async for task in limit_concurrency(aws, limit):
        yield await task


async def limit_concurrency(aws, limit):
    try:
        aws = aiter(aws)
        is_async = True
    except TypeError:
        aws = iter(aws)
        is_async = False

    aws_ended = False
    pending = set()

    while pending or not aws_ended:
        while len(pending) < limit and not aws_ended:
            try:
                aw = await anext(aws) if is_async else next(aws)
            except StopAsyncIteration if is_async else StopIteration:
                aws_ended = True
            else:
                pending.add(asyncio.ensure_future(aw))

        if not pending:
            return

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


async def async_main(args, limit):
    start = time.monotonic()
    wrapped = return_args_and_exceptions(do_stuff)
    async for arg, result in map_unordered(wrapped, args, limit=limit):
        print(f"{(time.monotonic() - start):.1f}: {arg} -> {result}")
    print(f"{(time.monotonic() - start):.1f}: done")


def on_iter_end(it, callback):
    for x in it:
        yield x
    callback()


async def as_async_iter(it):
    for x in it:
        yield x


def main():
    limit = int(sys.argv[1])
    args = [float(n) for n in sys.argv[2:]]
    timeout = sum(args) + 0.5
    args = on_iter_end(args, lambda: print("iter end"))
    args = as_async_iter(args)
    asyncio.run(asyncio.wait_for(async_main(args, limit), timeout))


if __name__ == '__main__':
    main()

# ref : https://death.andgravity.com/limit-concurrency#aside-backpressure
# ref : https://bugs.python.org/issue30782
# ref : https://stackoverflow.com/questions/61474547/asyncio-limit-concurrency-with-async-generator
