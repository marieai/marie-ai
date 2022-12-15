import base64
import json
import os
import sys
import typing
from abc import ABC

import cv2
import numpy as np

from marie.base_handler import BaseHandler
from marie.logging.predefined import default_logger
from marie.numpyencoder import NumpyEncoder

# Add parent to the search path, so we can reference the modules(craft, pix2pix) here without throwing and exception
from marie.utils.draw_truetype import drawTrueTypeTextOnImage
from marie.utils.utils import ensure_exists

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

logger = default_logger


def encodeimg2b64(img: np.ndarray) -> str:
    """encode image to base64"""
    retval, buffer = cv2.imencode(".png", img)
    png_as_text = base64.b64encode(buffer).decode()
    return png_as_text


class IcrProcessor(BaseHandler):
    def __init__(self, work_dir: str = "/tmp/icr", cuda: bool = True) -> None:
        super().__init__()
        self.cuda = cuda
        self.work_dir = work_dir

    def extract_text(self, _id, key, image):
        """Recognize text from a single image.
           Process image via ICR, this is low lever API, to get more usable results call extract_icr.

        Args:
            _id: Unique Image ID
            key: Unique image key
            image: A pre-cropped image containing characters
        """

        logger.debug("ICR processing : {}, {}".format(_id, key))
        results = self.recognize_from_boxes(
            [image], [0, 0, image.shape[1], image.shape[0]]
        )
        if len(results) == 1:
            r = results[0]
            return r["text"], r["confidence"]
        return None, 0

    def recognize_from_boxes(
        self, image, boxes, **kwargs
    ) -> typing.List[typing.Dict[str, any]]:
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

    def recognize(self, _id, key, img, boxes, fragments, lines):
        """Recognize text from multiple images.
        Args:
            _id: Unique Image ID
            key: Unique image key/region for the extraction
            img: Image to run recognition against,
            boxes: Boxes to recognize
            fragments: Image fragments to extract, A pre-cropped image containing characters
            lines: Lines associates with the image fragment / boxes
        """

        logger.debug(f"ICR recognize : {_id}, {key}")
        assert len(boxes) == len(
            fragments
        ), "You must provide the same number of box groups as images."
        assert len(boxes) == len(
            lines
        ), "You must provide the same number of lines as boxes."
        encode_fragments = False

        try:
            shape = img.shape
            overlay_image = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
            debug_dir = ensure_exists(os.path.join("/tmp/icr", _id))
            debug_all_dir = ensure_exists(os.path.join("/tmp/icr", "fields", key))

            meta = {
                "imageSize": {"width": img.shape[1], "height": img.shape[0]},
                "page": 0,
                "lang": "en",
            }

            # fail fast as we have not found any bounding boxes
            if len(boxes) == 0:
                logger.warning("Empty bounding boxes, possibly a blank page")
                return {
                    "meta": meta,
                    "words": [],
                    "lines": [],
                }, overlay_image

            words = []
            results = self.recognize_from_fragments(fragments)
            # reindex based on their X positions LTR reading order
            boxes = np.array(boxes)
            lines = np.array(lines)
            results = np.array(results)
            indices = np.argsort(boxes[:, 0])

            # for i, (box, fragment, line, extraction) in enumerate(zip(boxes, fragments, lines, results)):
            for i, index in enumerate(indices):
                box = boxes[index]
                fragment = fragments[index]
                line = lines[index]
                extraction = results[index]

                txt_label = extraction["text"]
                confidence = extraction["confidence"]
                conf_label = round(confidence, 4)

                payload = {
                    "id": i,
                    "text": txt_label,
                    "confidence": conf_label,
                    "box": box,
                    "line": line,
                }
                if encode_fragments:
                    payload["fragment_b64"] = encodeimg2b64(fragment)

                words.append(payload)

                if False:
                    overlay_image = drawTrueTypeTextOnImage(
                        overlay_image,
                        txt_label,
                        (box[0], box[1] + box[3] // 2),
                        18,
                        (139, 0, 0),
                    )
                    overlay_image = drawTrueTypeTextOnImage(
                        overlay_image,
                        conf_label,
                        (box[0], box[1] + box[3]),
                        10,
                        (0, 0, 255),
                    )

            if False:
                savepath = os.path.join(debug_dir, f"{key}-icr-result.png")
                cv2.imwrite(savepath, overlay_image)

                savepath = os.path.join(debug_all_dir, f"{_id}.png")
                cv2.imwrite(savepath, overlay_image)

            unique_line_ids = sorted(np.unique(lines))
            line_results = np.empty(len(unique_line_ids), dtype=object)
            aligned_words = []
            word_index = 0

            for i, line_numer in enumerate(unique_line_ids):
                word_ids = []
                box_picks = []
                word_picks = []

                _w = []
                _conf = []

                for word in words:
                    if line_numer == word["line"]:
                        word["word_index"] = word_index
                        word_picks.append(word)
                        word_ids.append(word["id"])
                        box_picks.append(word["box"])
                        _w.append(word["text"])
                        _conf.append(word["confidence"])
                        aligned_words.append(word)
                        word_index += 1

                if len(box_picks) == 0:
                    raise Exception("Every word needs to be associated with a box")

                text = " ".join(_w)
                box_picks = np.array(box_picks)

                min_x = box_picks[:, 0].min()
                min_y = box_picks[:, 1].min()
                max_w = box_picks[:, 2].max()
                max_h = box_picks[:, 3].max()
                bbox = [min_x, min_y, max_w, max_h]

                line_results[i] = {
                    "line": i + 1,  # Line index (1.. N), relative to the image
                    "wordids": word_ids,  # Word ID that make this line
                    "text": text,  # Text from merged text line
                    "bbox": bbox,  # Bounding box of the text
                    "confidence": round(np.average(_conf), 4),
                }

                if False:
                    print("-------")
                    print(line_results[i])
            #
            # print("aligned_words")
            # print(aligned_words)
            # for i, word in enumerate(aligned_words):
            #     print(f"  -> {word}")

            result = {
                "meta": meta,
                "words": aligned_words,
                "lines": line_results,
            }

            if len(words) != len(aligned_words):
                raise Exception(
                    f"Aligned words should match original words got: {len(aligned_words)}, {len(words)}"
                )

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
                for line in line_results:
                    txt = line["text"]
                    print(f" >> {txt}")

        except Exception as ex:
            raise ex

        return result, overlay_image
