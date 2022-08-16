import asyncio

from marie.concur import ScheduledExecutorService


async def async_foo():
    print("started : async_foo")
    await asyncio.sleep(3)
    print("complete : async_foo")


async def main():
    asyncio.create_task(async_foo())
    # btw, you can also create tasks inside non-async funcs

    print("Do some actions 1")
    await asyncio.sleep(1)
    print("Do some actions 2")
    await asyncio.sleep(1)
    print("Do some actions 3")


if __name__ == "__main__":
    print("Main")
    scheduler = ScheduledExecutorService.new_scheduled_asyncio_pool()
    print(scheduler)

if __name__ == "__main_XX_":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
