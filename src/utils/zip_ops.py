import glob
import os
from zipfile import ZipFile

from timer import Timer


@Timer(text="Creating zip in {:.2f} seconds")
def merge_zip(src_dir, dst_path):
    """Add files from directory to the zipfile without absolute path"""
    from os.path import basename

    with ZipFile(dst_path, "w") as newzip:
        for filename in sorted(glob.glob(os.path.join(src_dir, "*.*"))):
            newzip.write(filename, basename(filename))
