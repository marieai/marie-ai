import difflib
import glob
import os

import cv2
import torch as torch
from PIL import Image

from marie.boxes import BoxProcessorUlimDit, PSMode
from marie.document import TrOcrProcessor
from marie.utils.ocr_debug import dump_bboxes

use_cuda = torch.cuda.is_available()


def build_ocr_engines():
    # return None, None, None

    box_processor = BoxProcessorUlimDit(
        models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
        cuda=use_cuda,
    )

    trocr_processor = TrOcrProcessor(
        models_dir="/mnt/data/marie-ai/model_zoo/trocr",
        model_name_or_path="/data/models/unilm/trocr/ft_SROIE_LINES_SET41/checkpoint_best.pt",
        cuda=use_cuda,
    )
    # craft_processor = CraftOcrProcessor(cuda=True)
    craft_processor = None

    return box_processor, trocr_processor, craft_processor


def process_image(img_path, box_processor, icr_processor):
    image = Image.open(img_path).convert("RGB")
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (
        boxes,
        fragments,
        lines,
        _,
        lines_bboxes,
    ) = box_processor.extract_bounding_boxes("gradio", "field", image, PSMode.SPARSE)

    result, overlay_image = icr_processor.recognize(
        "gradio ", "00000", image, boxes, fragments, lines, return_overlay=True
    )

    dump_bboxes(image, result, prefix=name, threshold=0.90)


def verify_dir(src_dir: str, output_dir: str, processor) -> None:
    print("Verifying images in directory: ", src_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Get a list of all image directories
    ext = "jpg"
    format = "SROIE"  # SROIE / TEXT # READY dir contains plain text files CONVERTED dir contains SROIE format
    # ext = "png"
    # format = "TEXT"  # SROIE / TEXT # READY dir contains plain text files CONVERTED dir contains SROIE format

    img_dirs = glob.glob(os.path.join(src_dir, f"*.{ext}"), recursive=True)
    print("Total images: ", len(img_dirs))

    correct_predictions = 0
    total_predictions = 0

    for idx, img_path in enumerate(img_dirs):
        try:
            print("Processing: ", img_path)
            src_image = os.path.join(src_dir, img_path)
            img = Image.open(src_image)
            # parses the SROIE label file 0-8 are bounding box coordinates and the rest is the text
            label_file = src_image.replace(f".{ext}", ".txt")

            if not os.path.exists(label_file):
                print(f"Label file not found: {label_file}")
                # try to find the label in current directory but using label.txt
                basedir = os.path.dirname(src_image)
                label_file = os.path.join(basedir, "label.txt")
                if not os.path.exists(label_file):
                    print(f"Label file not found: {label_file}")
                    raise Exception("Label file not found : {label_file}")

            with open(os.path.join(src_dir, label_file), "r") as f:
                labels = f.readlines()

            text = labels[0].strip()
            if format == "SROIE":
                parts = text.split(",", maxsplit=8)  # We split at the first 8 commas
                expected_text = parts[8].strip()  # The 9th part is the text
            else:
                expected_text = text

            expected_text = expected_text.strip().upper()

            fragment = cv2.imread(src_image)
            results = trocr_processor.recognize_from_fragments([fragment])
            predicted_text = results[0]["text"]
            predicted_text = predicted_text.strip().upper()
            confidence = results[0]["confidence"]

            # REMOVE SPACES AND STIP
            expected_text = expected_text.strip()
            predicted_text = predicted_text.strip()
            # expected_text = expected_text.replace(" ", "")
            # predicted_text = predicted_text.replace(" ", "")

            differences = difflib.ndiff(expected_text, predicted_text)
            diff_str = " ".join(
                differences
            )  # join differences with space instead of newline
            diff_str = diff_str.replace("\n", " ")  # ensure no newlines

            # Calculate similarity
            seq = difflib.SequenceMatcher(None, expected_text, predicted_text)
            similarity = seq.ratio()

            print("-----------------------")
            print("Text   : ", expected_text)
            print("Result : ", predicted_text)
            print("Confidence: ", confidence)
            print("Confidence: ", confidence)
            print(f"Similarity: ", similarity)

            total_predictions += 1

            if expected_text == predicted_text:
                correct_predictions += 1

            else:
                with open(os.path.join(output_dir, "errors.txt"), "a") as f:
                    f.write(
                        f"F: {src_image}\nE: {expected_text}\nP: {predicted_text}\nC: {confidence}\nS: {similarity}\n\n"
                    )
                # get the filename without extension
                name = os.path.basename(src_image)
                # create folder
                output_path = os.path.join(output_dir, "errors")
                os.makedirs(output_path, exist_ok=True)
                # write the status to a file
                with open(os.path.join(output_path, f"{name}.txt"), "w") as f:
                    f.write(
                        f"E: {expected_text}\nP: {predicted_text}\nC: {confidence}\nS: {similarity}\n\n"
                    )

        except Exception as e:
            print(e)

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Total Predictions: ", total_predictions)
    print("Correct Predictions: ", correct_predictions)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    box_processor, trocr_processor, craf_processor = build_ocr_engines()

    if True:
        verify_dir(
            "/home/greg/datasets/SROIE_OCR/converted",
            "/home/greg/datasets/SROIE_OCR/validation",
            trocr_processor,
        )

    if False:
        verify_dir(
            "/home/greg/datasets/SROIE_OCR/lines/raw",
            "/home/greg/datasets/SROIE_OCR/lines/validation",
            trocr_processor,
        )

    if False:
        verify_dir(
            "/home/greg/datasets/SROIE_OCR/boxes/validated-True/alpha",
            "/home/greg/datasets/SROIE_OCR/validation",
            trocr_processor,
        )
