import glob
import os
from zipfile import ZipFile

from marie.timer import Timer


@Timer(text="Creating zip in {:.2f} seconds", logger=None)
def merge_zip(src_dir, dst_path, glob_filter="*.*"):
    """Add files from directory to the zipfile without absolute path"""
    from os.path import basename

    with ZipFile(dst_path, "w") as newzip:
        for filename in sorted(glob.glob(os.path.join(src_dir, glob_filter))):
            newzip.write(filename, basename(filename))
