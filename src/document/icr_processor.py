import os
import sys
from abc import ABC

from timer import Timer

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class Object(object):
    pass


class IcrProcessor(ABC):
    def __init__(self, work_dir: str = '/tmp/icr', cuda: bool = False) -> None:
        print("Base ICR processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir

    @Timer(text="ICR in {:.2f} seconds")
    def recognize(self, _id, key, img, boxes, image_fragments, lines):
        """Recognize text from multiple images.
        Args:
            id: Unique Image ID
            key: Unique image key/region for the extraction
            img: A pre-cropped image containing characters
        """
