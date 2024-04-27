import cv2
from pylibdmtx.pylibdmtx import decode

# sudo apt-get install libdmtx0a
filename = "/home/gbugaj/datasets/private/qr-codes/sample-002.png"
image = cv2.imread(filename)
decoded_objects = decode(image)
print(decoded_objects)