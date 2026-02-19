import os

from marie.utils.tiff_ops import burst_tiff, merge_tiff

if __name__ == "__main__":

    img_path = "~/tmp/163611436.tif"
    dest_dir = "~/tmp/001-300DPI"

    img_path = os.path.expanduser(img_path)
    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    burst_tiff(img_path, dest_dir)
    # merge_tiff(dest_dir, "/tmp/output.tif")
