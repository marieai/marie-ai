import os
import sys
import typing
from abc import ABC

import numpy as np
import cv2
import base64
import json

from timer import Timer

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
from utils.utils import ensure_exists

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class Object(object):
    pass


def encodeimg2b64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode(".png", img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text


class IcrProcessor(ABC):
    def __init__(self, work_dir: str = "/tmp/icr", cuda: bool = False) -> None:
        print("Base ICR processor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir

    def extract_text(self, _id, key, image):
        """Recognize text from a single image.
           Process image via ICR, this is lowlever API, to get more usable results call extract_icr.

        Args:
            _id: Unique Image ID
            key: Unique image key
            image: A pre-cropped image containing characters
        """

        print("ICR processing : {}, {}".format(_id, key))
        results = self.recognize_from_boxes([image], [0, 0, image.shape[1], image.shape[0]])
        if len(results) == 1:
            r = results[0]
            return r["text"], r["confidence"]
        return None, 0

    def recognize_from_boxes(self, image, boxes, **kwargs) -> typing.List[typing.Dict[str, any]]:
        """Recognize text from image using lists of bounding boxes.

        Args:
            image: input images, supplied as numpy arrays with shape
                (H, W, 3).
            boxes: A list of boxes to extract
        """
        raise Exception("Not yet implemented")

    def recognize_from_fragments(self, image_fragments):
        """Recognize text from image fragments.

        Args:
            image_fragments: input images, supplied as numpy arrays with shape
                (H, W, 3).
        """
        raise Exception("Not Implemented")

    @Timer(text="ICR in {:.2f} seconds")
    def recognize(self, _id, key, img, boxes, image_fragments, lines):
        """Recognize text from multiple images.
        Args:
            id: Unique Image ID
            key: Unique image key/region for the extraction
            img: A pre-cropped image containing characters
        """
        print(f"ICR recognize : {_id}, {key}")
        assert len(boxes) == len(image_fragments), "You must provide the same number of box groups as images."

        try:
            shape = img.shape
            overlay_image = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
            debug_dir = ensure_exists(os.path.join("/tmp/icr", _id))
            debug_all_dir = ensure_exists(os.path.join("/tmp/icr", "fields", key))

            meta = {"imageSize": {"width": img.shape[1], "height": img.shape[0]}, "lang": "en"}

            words = []
            max_line_number = 0
            results = self.recognize_from_fragments(image_fragments)

            for i in range(len(boxes)):
                box, fragment, line = boxes[i], image_fragments[i], lines[i]
                # txt, confidence = self.extract_text(id, str(i), fragment)
                extraction = results[i]
                txt = extraction["text"]
                confidence = extraction["confidence"]
                # print('Processing [box, line, txt, conf] : {}, {}, {}, {}'.format(box, line, txt, confidence))
                conf_label = f"{confidence:0.4f}"
                txt_label = txt

                payload = dict()
                payload["id"] = i
                payload["text"] = txt
                payload["confidence"] = round(confidence, 4)
                payload["box"] = box
                payload["line"] = line
                payload["fragment_b64"] = encodeimg2b64(fragment)

                words.append(payload)

                if line > max_line_number:
                    max_line_number = line

                if False:
                    overlay_image = drawTrueTypeTextOnImage(
                        overlay_image, txt_label, (box[0], box[1] + box[3] // 2), 18, (139, 0, 0)
                    )
                    overlay_image = drawTrueTypeTextOnImage(
                        overlay_image, conf_label, (box[0], box[1] + box[3]), 10, (0, 0, 255)
                    )

            if False:
                savepath = os.path.join(debug_dir, f"{key}-icr-result.png")
                imwrite(savepath, overlay_image)

                savepath = os.path.join(debug_all_dir, f"{_id}.png")
                imwrite(savepath, overlay_image)

            line_ids = np.empty((max_line_number), dtype=object)
            words = np.array(words)

            for i in range(0, max_line_number):
                current_lid = i + 1
                word_ids = []
                box_picks = []
                word_picks = []

                for word in words:
                    lid = word["line"]
                    if lid == current_lid:
                        word_ids.append(word["id"])
                        box_picks.append(word["box"])
                        word_picks.append(word)

                box_picks = np.array(box_picks)
                word_picks = np.array(word_picks)

                # print(f'**** {len(box_picks)}')
                # FIXME : This si a bug and need to be fixed, this should never happen
                if len(box_picks) == 0:
                    line_ids[i] = {
                        "line": i + 1,
                        "wordids": word_ids,
                        "text": "",
                        "bbox": bbox,
                        "confidence": round(0, 4),
                    }
                    continue

                x1 = box_picks[:, 0]
                idxs = np.argsort(x1)
                aligned_words = word_picks[idxs]
                _w = []
                _conf = []

                for wd in aligned_words:
                    _w.append(wd["text"])
                    _conf.append(wd["confidence"])

                text = " ".join(_w)

                min_x = box_picks[:, 0].min()
                min_y = box_picks[:, 1].min()
                max_w = box_picks[:, 2].max()
                max_h = box_picks[:, 3].max()
                bbox = [min_x, min_y, max_w, max_h]

                line_ids[i] = {
                    "line": i + 1,
                    "wordids": word_ids,
                    "text": text,
                    "bbox": bbox,
                    "confidence": round(np.average(_conf), 4),
                }

            result = {
                "meta": meta,
                "words": words,
                "lines": line_ids,
            }

            if False:
                with open("/tmp/icr/data.json", "w") as f:
                    json.dump(
                        result,
                        f,
                        sort_keys=True,
                        separators=(",", ": "),
                        ensure_ascii=False,
                        indent=4,
                        cls=NumpyEncoder,
                    )

                print("------ Extraction ------------")
                for line in line_ids:
                    txt = line["text"]
                    print(f" >> {txt}")

        except Exception as ex:
            raise ex

        return result, overlay_image

