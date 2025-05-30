import asyncio
import queue
import threading


def run_coroutine_in_current_loop(coroutine):
    """
    Runs `coroutine` to completion, even if we're inside a running loop.
    - Outside any loop: uses asyncio.run()
    - Inside a loop: spins up a fresh loop in a background thread,
      runs `coroutine`, shuts down async generators, then drains
      any *other* pending tasks before closing.
    """
    try:
        # If no loop is running here, just run normally.
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result_q = queue.Queue()

    def _thread_target():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        async def _runner():
            result = await coroutine
            # 2) clean up any async generators
            await new_loop.shutdown_asyncgens()

            # 3) drain *other* pending tasks, excluding this one
            current = asyncio.current_task()
            pending = [
                t for t in asyncio.all_tasks() if not t.done() and t is not current
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            return result

        try:
            res = new_loop.run_until_complete(_runner())
            result_q.put((True, res))
        except Exception as exc:
            result_q.put((False, exc))
        finally:
            new_loop.close()

    t = threading.Thread(target=_thread_target)
    t.start()
    t.join()

    ok, payload = result_q.get()
    if not ok:
        raise payload
    return payload
