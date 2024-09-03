import logging
import os
import time
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

MAX_CHUNK_LINE_LENGTH = 10
MAX_CHUNK_CHAR_LENGTH = 20000

JOB_LOGS_PATH_TEMPLATE = "job-driver-{submission_id}.log"


def file_tail_iterator(path: str) -> Iterator[Optional[List[str]]]:
    """Yield lines from a file as it's written.

    Returns lines in batches of up to 10 lines or 20000 characters,
    whichever comes first. If it's a chunk of 20000 characters, then
    the last line that is yielded could be an incomplete line.
    New line characters are kept in the line string.

    Returns None until the file exists or if no new line has been written.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path)}.")

    while not os.path.exists(path):
        logger.debug(f"Path {path} doesn't exist yet.")
        yield None

    EOF = ""

    with open(path, "r") as f:
        lines = []

        chunk_char_count = 0
        curr_line = None

        while True:
            # We want to flush current chunk in following cases:
            #   - We accumulated 10 lines
            #   - We accumulated at least MAX_CHUNK_CHAR_LENGTH total chars
            #   - We reached EOF
            if (
                len(lines) >= 10
                or chunk_char_count > MAX_CHUNK_CHAR_LENGTH
                or curr_line == EOF
            ):
                # Too many lines, return 10 lines in this chunk, and then
                # continue reading the file.
                yield lines or None

                lines = []
                chunk_char_count = 0

            # Read next line
            curr_line = f.readline()

            # `readline` will return
            #   - '' for EOF
            #   - '\n' for an empty line in the file
            if curr_line != EOF:
                # Add line to current chunk
                lines.append(curr_line)
                chunk_char_count += len(curr_line)
            else:
                # If EOF is reached sleep for 1s before continuing
                time.sleep(1)
