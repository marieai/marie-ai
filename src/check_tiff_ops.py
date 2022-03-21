import os

import cv2

from utils.tiff_ops import burst_tiff, merge_tiff

if __name__ == "__main__":

    img_path = "/home/greg/dataset/medprov/PID/150300431/PID_576_7188_0_150300431.tif"
    dest_dir = "/home/greg/dataset/medprov/PID/150300431/burst_test"

    if not os.path.exists(img_path):
        raise Exception(f"File not found : {img_path}")

    burst_tiff(img_path, dest_dir)

    merge_tiff(dest_dir, '/tmp/output.tif')
