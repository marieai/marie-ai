import errno
import os

from .utils import FileSystem


def mkdir_p(path: str) -> str:
    try:
        os.makedirs(path)
        return path
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return path
        else:
            raise
