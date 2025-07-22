import random


def format_duration(milliseconds):
    """Given milliseconds, return human readable duration string such as:
    533ms, 2.1s, 4m52s, 34m12s, 1h4m.
    """
    # under 1 ms
    # ex: 0.83ms
    # ex: 8.3ms
    if milliseconds < 10:
        return f"{round(milliseconds, 2)}ms"

    # between 10 ms and 1000 ms
    # ex: 533ms
    if milliseconds < 1000:
        return f"{int(milliseconds)}ms"

    # between one second and one minute
    # ex: 5.6s
    if milliseconds < 1000 * 60:
        seconds = milliseconds / 1000
        return f"{round(seconds, 2)}s"

    # between one minute and 60 minutes
    # 5m42s
    if milliseconds < 1000 * 60 * 60:
        minutes = int(milliseconds // (1000 * 60))
        seconds = int(milliseconds % (1000 * 60) // (1000))
        return f"{minutes}m{seconds}s"

    # Above one hour
    else:
        hours = int(milliseconds // (1000 * 60 * 60))
        minutes = int(milliseconds % (1000 * 60 * 60) // (1000 * 60))
        return f"{hours}h{minutes}m"


def exponential_backoff(failures: int, initial_backoff: int, max_backoff: int) -> float:
    """
    Calculates an exponential backoff time with a random jitter based on the number
    of failures. The backoff time increases exponentially with the number of
    failures, but is capped at a maximum value. A random jitter is then added to
    avoid synchronized retries in large systems.

    :return: The calculated backoff time in seconds as a float.
    """
    base_delay = min(initial_backoff * (2 ** (failures - 1)), max_backoff)
    jitter = random.uniform(0, base_delay * 0.5)
    backoff_time = base_delay + jitter
    return backoff_time
