import asyncio
import threading
import time
from threading import current_thread

from marie.concur import ScheduledExecutorService
from marie.concur.ScheduledExecutorService import TimeUnit


async def async_task(name: str):
    print(f"{name} : {current_thread().name} {time.time()}  Start")
    await asyncio.sleep(1)
    print(f"{name} : {current_thread().name} {time.time()} Complete")


async def cancel_callback(task):
    print(f"canceling task : {task}")
    await task.stop()


async def main():
    scheduler = ScheduledExecutorService.new_scheduled_asyncio_pool()

    t1 = scheduler.schedule_at_fixed_rate(
        async_task, 1, TimeUnit.MILLISECONDS, name="T1"
    )

    t2 = scheduler.schedule_at_fixed_rate(
        async_task, 2, TimeUnit.MILLISECONDS, name="T2"
    )

    asyncio.get_event_loop().call_later(3, asyncio.create_task, cancel_callback(t1))


async def main_single():
    scheduler = ScheduledExecutorService.new_scheduled_asyncio_pool()
    t1 = scheduler.schedule_at_fixed_rate(
        async_task, 1, TimeUnit.MILLISECONDS, name="T1"
    )

    # call_later() only supports callbacks (regular functions); you  can’t pass in a coroutine.
    asyncio.get_event_loop().call_later(3, asyncio.create_task, cancel_callback(t1))
    # await t1.task


async def main_delay():
    scheduler = ScheduledExecutorService.new_scheduled_asyncio_pool()
    t1 = scheduler.schedule_with_fixed_delay(
        async_task, 2, 1, TimeUnit.MILLISECONDS, name="T1"
    )

    await t1.start()
    print(t1.task)

    asyncio.get_event_loop().call_later(3, cancel_callback, t1)

    await t1.task


if __name__ == "__main__":
    print("Main")
    loop = asyncio.get_event_loop()
    try:
        # asyncio.ensure_future(main_single())
        asyncio.ensure_future(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()

if __name__ == "__main__XX":
    print("Main")

    try:
        asyncio.run(main_single())
    except asyncio.CancelledError:
        pass

    if False:
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main_single())
        except asyncio.CancelledError:
            pass
