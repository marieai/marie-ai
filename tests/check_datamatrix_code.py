import os
import sys

import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode

# sudo apt-get install libdmtx0a


def process_data_matrix(filename):
    filename = os.path.expanduser(filename)
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to load image at {filename}")
        sys.exit(1)
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    unique_values = np.unique(gray)
    if len(unique_values) > 2:
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    pad = 100
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    decoded_objects = decode(gray, corrections=3)

    print(decoded_objects)

    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for obj in decoded_objects:
        print("Decoded Data : ", obj.data.decode('utf-8'))
        rect = obj.rect
        cv2.rectangle(output, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height), (0, 0, 255), 2)
        # cv2.putText(output, obj.data.decode('utf-8'), (rect.left, rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        center_x = rect.left + rect.width // 2
        center_y = rect.top + rect.height // 2

        text_size = cv2.getTextSize(obj.data.decode('utf-8'), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]

        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(output, obj.data.decode('utf-8'), (text_x, rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1)

    cv2.imshow("decoded", output)
    cv2.waitKey(0)


if __name__ == '__main__':
    filename = "~/datasets/private/qr-codes/code.png"
    process_data_matrix(filename)
