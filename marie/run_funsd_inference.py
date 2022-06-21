import glob
import io
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import numpy as np
from transformers.utils import check_min_version

from PIL import Image
from transformers import LayoutLMv2Processor, LayoutLMv2FeatureExtractor, LayoutLMv2ForTokenClassification, \
    LayoutLMv2TokenizerFast

# https://programtalk.com/vs4/python/huggingface/transformers/tests/layoutlmv2/test_processor_layoutlmv2.py/
# https://github.com/huggingface/transformers/blob/d3ae2bd3cf9fc1c3c9c9279a8bae740d1fd74f34/tests/layoutlmv2/test_processor_layoutlmv2.py

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from marie.boxes.box_processor import PSMode
from marie.numpyencoder import NumpyEncoder
from marie.utils.image_utils import viewImage, read_image
from marie.utils.utils import ensure_exists

check_min_version("4.5.0")
logger = logging.getLogger(__name__)

# Calling this from here prevents : "AttributeError: module 'detectron2' has no attribute 'config'"
from detectron2.config import get_cfg

from boxes.craft_box_processor import BoxProcessorCraft
from document.trocr_icr_processor import TrOcrIcrProcessor


def from_json_file(filename):
    with io.open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return "other"
    return label



def obtain_words(src_image):
    image = read_image(src_image)

    work_dir_boxes = ensure_exists("/tmp/boxes")
    work_dir_icr = ensure_exists("/tmp/icr")

    boxp = BoxProcessorCraft(work_dir=work_dir_boxes, models_dir="./model_zoo/craft", cuda=True)
    icrp = TrOcrIcrProcessor(work_dir=work_dir_icr, cuda=True)


    key = "funsd"
    boxes, img_fragments, lines, _ = boxp.extract_bounding_boxes(key, "field", image, PSMode.SPARSE)
    results, overlay_image = icrp.recognize(key, "test", image, boxes, img_fragments, lines)

    print(boxes)
    print(results)
    # change from xywy -> xyxy
    x0 = 0
    y0 = 0

    for word in results["words"]:
        x, y, w, h = word["box"]
        w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
        word["box"] = w_box

    return results

def main_image(src_image):
    # labels = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    labels = ["O", 'B-MEMBER_NAME', 'I-MEMBER_NAME', 'B-MEMBER_NAME_ANSWER', 'I-MEMBER_NAME_ANSWER', 'B-MEMBER_NUMBER', 'I-MEMBER_NUMBER', 'B-MEMBER_NUMBER_ANSWER', 'I-MEMBER_NUMBER_ANSWER', 'B-PAN', 'I-PAN', 'B-PAN_ANSWER', 'I-PAN_ANSWER', 'B-DOS', 'I-DOS', 'B-DOS_ANSWER', 'I-DOS_ANSWER', 'B-PATIENT_NAME', 'I-PATIENT_NAME', 'B-PATIENT_NAME_ANSWER', 'I-PATIENT_NAME_ANSWER']
    logger.info("Labels : {}", labels)

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    # prepare for the model
    # we do not want to use the pytesseract
    # LayoutLMv2FeatureExtractor requires the PyTesseract library but it was not found in your environment. You can install it with pip:
    # processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    # Method:2 Create Layout processor with custom future extractor
    # feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    feature_extractor = LayoutLMv2FeatureExtractor(size = 512, apply_ocr=False)# 224
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Display vocabulary
    # print(tokenizer.get_vocab())

    image = Image.open(src_image).convert("RGB")
    # image.show()

    width, height = image.size

    # Next, let's move everything to the GPU, if it's available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Need to obtain boxes and OCR for the document    
    from pathlib import Path
    home = str(Path.home())
    ocr_output_dir = os.path.join(home, '.marie')

    print(f"output dir : {ocr_output_dir}")
    filename = src_image.split("/")[-1]
    json_path = os.path.join(home, '.marie', f'{filename}.json')

    if not os.path.exists(json_path):
        raise Exception(f"OCR File not found : {json_path}")

    # results = from_json_file("/tmp/ocr-results.json")
    results = from_json_file(json_path)

    print(results)
    # results = obtain_words(image)
    visualize_icr(image, results)

    words = []
    boxes = []

    for i, word in enumerate(results["words"]):
        words.append(word["text"].lower())
        # words.append(word["text"])
        box_norm = normalize_bbox(word["box"], (width, height))
        boxes.append(box_norm)

        # This is to prevent following error
        # The expanded size of the tensor (516) must match the existing size (512) at non-singleton dimension 1.
        # print(len(boxes))
        if len(boxes) == 512:
            print('Clipping MAX boxes at 512')
            break


    assert len(words) == len(boxes)
    print(words)
    print(boxes)
    print(len(words))

    encoded_inputs = processor(image, words, boxes=boxes, padding="max_length", truncation=True, return_tensors="pt")
    expected_keys = ["attention_mask", "bbox", "image", "input_ids", "token_type_ids"]
    actual_keys = sorted(list(encoded_inputs.keys()))

    print("Expected Keys : ", expected_keys)
    print("Actual Keys : ", actual_keys)

    img_tensor = encoded_inputs["image"]
    img = Image.fromarray((img_tensor[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0))
    img.save(f'/tmp/tensors/img_tensor.png')
    
    
    os.exit()
    for key in expected_keys:
        print(f"key: {key}")
        print(encoded_inputs[key])

    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    from pathlib import Path
    home = str(Path.home())
    model_dir = os.path.join(home, './tmp/models/layoutlmv2-finetuned-gb', "checkpoint-6500")
    model_dir = "/home/gbugaj/dev/unilm/layoutlmft/examples/checkpoints"
    print(f"output dir : {model_dir}")

    # load the fine-tuned model from the hub
    # model = LayoutLMv2ForTokenClassification.from_pretrained("/tmp/models/layoutlmv2-finetuned-cord")
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_dir)
    model.to(device)

    # forward pass
    outputs = model(**encoded_inputs)
    print(outputs.logits.shape)

    logits = outputs.logits
    # probability = softmax(outputs.logits, axis=-1)
    # print(outputs.logits)
    # print('Probabilities : ')
    # print(probability)

    # https://discuss.pytorch.org/t/logits-vs-log-softmax/95979
    import torch.nn.functional as F
    print('Probas from logits:\n', F.softmax(logits, dim=-1))
    print('Log-softmax:\n', F.log_softmax(logits, dim=-1))
    print('Difference between logits and log-softmax:\n', logits - F.log_softmax(logits, dim=-1))
    print('Probas from log-softmax:\n', F.softmax(F.log_softmax(logits, dim=-1), dim=-1))

    # Let's create the true predictions, true labels (in terms of label names) as well as the true boxes.

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

    print(true_predictions)
    print(true_boxes)

    def iob_to_label(label):
        label = label[2:]
        if not label:
            return "other"
        return label

    label2color = {"question": "blue", "answer": "green", "header": "orange", "other": "violet"}

    label2color = {"pan": "blue", "pan_answer": "green",
                   "dos": "orange", "dos_answer": "violet",
                   "member": "blue", "member_answer": "green",
                   "member_number": "blue", "member_number_answer": "green",
                   "member_name": "blue", "member_name_answer": "green",
                   "patient_name": "blue", "patient_name_answer": "green",
                   "other": "red"
                   }

    draw = ImageDraw.Draw(image, 'RGBA')
    font = ImageFont.load_default()

    for prediction, box in zip(true_predictions, true_boxes):
        # don't draw other 
        label = prediction[2:]
        if not label:
            continue
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label], width=1,  fill=(0, 255, 0, 50))
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill='red', font=font, width=1)
    
    del draw

    # image.show()
    image.save(f"/tmp/tensors/{filename}")


def visualize_icrXXXX(image, icr_data):
    viz_img = image.copy()
    size = 18
    draw = ImageDraw.Draw(viz_img, "RGBA")
    try:
        font = ImageFont.truetype(os.path.join("./assets/fonts", "FreeMono.ttf"), size)
    except Exception as ex:
        # print(ex)
        font = ImageFont.load_default()

    for i, item in enumerate(icr_data["words"]):
        box = item["box"]
        text = item["text"]
        draw.rectangle(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], outline="#993300", fill=(0, 180, 0, 125), width=1
        )
        draw.text((box[0], box[1]), text=text, fill="blue", font=font, stroke_width=0)

    viz_img.save(f"/tmp/tensors/visualize_icr.png")


def visualize_icr(image, icr_data):
    viz_img = image.copy()

    size = 14
    draw = ImageDraw.Draw(viz_img, "RGBA")
    try:
        font = ImageFont.truetype(os.path.join("./assets/fonts", "FreeSans.ttf"), size)
    except Exception as ex:
        print(ex)
        font = ImageFont.load_default()
    
    # print(icr_data)
    words_all = []
    boxes_all = []
    # sort boxes by the  y-coordinate of the bounding box
    words = icr_data["words"]
    for idx, word in enumerate(words):
        box = np.array(word['box']).astype(np.int32)
        boxes_all.append(box)
    
    words = np.array(words)
    boxes_all = np.array(boxes_all)
    print(boxes_all[:, 0])
    # Sort boxes by x,y then reverse to get the right order    
    # idx = np.lexsort((boxes_all[:, 0], boxes_all[:, 1]))[::-1]
    ind = np.lexsort((boxes_all[:,0], boxes_all[:,1]))    
    sorted_words = words[ind]

    print(boxes_all)
    print(idx)
    print(sorted_words)

    for i, item in enumerate(words):
        box = item["box"]
        text = f'({i}){item["text"]}'
        words_all.append(text)
        
        # get text size
        text_size = font.getsize(text)
        button_size = (text_size[0]+8, text_size[1]+8)
        # create image with correct size and black background
        button_img = Image.new('RGBA', button_size, color=(150,255,150,150))
        # put text on button with 10px margins
        button_draw = ImageDraw.Draw(button_img, "RGBA")
        button_draw.text((4, 4), text=text, font=font, stroke_width=0, fill=(0, 0, 0, 0), width=1)
        # draw.rectangle(box, outline="red", width=1)
        # draw.text((box[0], box[1]), text=text, fill="blue", font=font, stroke_width=0)
        # put button on source image in position (0, 0)
        viz_img.paste(button_img, (box[0], box[1]))

    viz_img.save(f"/tmp/tensors/visualize_icr.png")
    st = " ".join(words_all)
    print(st)

    # viz_img.show()


def hash_file(filename):
   """"This function returns the SHA-1 hash
   of the file passed into it"""
   import hashlib

   # make a hash object
   h = hashlib.sha1()

   # open file for reading in binary mode
   with open(filename,'rb') as file:

       # loop till the end of the file
       chunk = 0
       while chunk != b'':
           # read only 1024 bytes at a time
           chunk = file.read(1024)
           h.update(chunk)

   # return the hex representation of digest
   return h.hexdigest()


def ocr_dir(src_dir, filename_filter="*.png"):
    from pathlib import Path
    home = str(Path.home())
    output_dir = os.path.join(home, '.marie')

    print(f"output dir : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for idx, _path in enumerate(glob.glob(os.path.join(src_dir, filename_filter))):
        filename = _path.split("/")[-1]
        json_path = os.path.join(home, '.marie', f'{filename}.json')
        print(f" : {_path}")
        print(f" : {filename}")
        print(f" : {json_path}")
        image = Image.open(image_path).convert("RGB")
        # image.show()
        results = obtain_words(image)
        print(results)

        with open(json_path, "w") as json_file:
            json.dump(
                results,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder,
            )

        visualize_icr(image, results)
        break


if __name__ == "__main__":

    # a = np.array([(3, 2, 4,6), (6, 2, 4,6), (3, 6, 4,6), (3, 4, 4,6), (16, 4, 4,6), (16, 1, 4,6),(5, 3, 4,6)])
    # ind = np.lexsort((a[:,1],a[:,0]))    

    # print(a[ind])

    # os.exit()
    os.putenv("TOKENIZERS_PARALLELISM", "false")
    image_path = "/home/greg/dataset/assets-private/corr-indexer/dataset/train_dataset/images/152606114_2.png"
    image_path = "/home/gbugaj/dataset/private/corr-indexer-converted/dataset/testing_data/images/152658535_2.png"
    image_path = "/home/greg/dataset/assets-private/corr-indexer-converted/dataset/testing_data/images/152658533_2.png"
    
    fname = "152658775_13.png" # F
    fname = "152658533_2.png"  # F
    fname = "152658535_2.png"  # VG
    fname = "152658536_0.png"  # ERROR
    fname = "152658538_2.png"  # P
    fname = "152658540_0.png"  # F
    fname = "152658541_2.png"  # P
    fname = "152658548_2.png"  # P
    fname = "152658549_2.png"  # G
    fname = "152658552_2.png"  # F
    fname = "152658551_4.png"  # F
    fname = "152658547_2.png"  # F
    fname = "152658671_2.png"  # P
    fname = "152658679_0.png"  # F

    fname = "152658535_2.png"
    
    # image_path = f"/home/greg/dataset/assets-private/corr-indexer-converted/dataset/testing_data/images/{fname}"
    # image_path = f"/home/gbugaj/dataset/private/corr-indexer-converted/dataset/testing_data/images/{fname}"
    image_path = f"/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test/{fname}"


    from pathlib import Path
    home = str(Path.home())
    json_path = os.path.join(home, '.marie', f'{fname}.json')
    # json_path = f"/tmp/ocr-results.json"

    print(f'json_path : {json_path}')
    print(f'image_path : {image_path}')

    # ocr_dir("/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test", fname)
    # main_image(image_path)

# (pytorch) gbugaj@asp-gpu001:~/dev/marie-ai$ CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" python ./marie/run_funsd_inference.py 
# json_path : /home/gbugaj/.marie/152658535_2.png.json
# image_path : /home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test/152658535_2.png
# output dir : /home/gbugaj/.marie
# ['17190', 'humana.', '022497 2/3', 'findings', 'review', 'summary', 'request', '(humana', '25074497', 'id', 'only):', 'use', '1', 'young', 'member', 'marsha', 'name:', 'identification', 'member', 'number:', 'h60258563', '08/1', '1955', 'member', 'date', 'of', 'birth:', '25/', 'h60258563', 'patient', 'number:', 'account', '1/25/2022', 'date(s):', '01/25/2022', 'service', '820220280085030', 'number(s):', 'claim', 'company', 'insurance', 'humana', 'legal', 'entity:', 'supported', 'billed', 'code', 'units', 'rationale', 'of', 'date', 'code', 'service', 'documentation', '90670', 'does', '01/25/2022', '90670', 'not', 'substantiate', 'cpt', '90670', 'for', 'code', '01/25/2022', 'date', 'of', 'service', 'the', 'as', 'medical', 'record', 'include', 'the', 'did', 'not', 'and', 'dosage', 'of', 'preumococcal', 'route', 'completed', 'vaccine.', 'reviews', 'are', 'using', 'the', 'cpt', 'coding', 'guidelines', 'and', 'when', 'applicable', 'cms', 'guidelines', 'for', 'medicare', 'member.', 'documentation', 'does', '01/25/2022', 'g0009', 'g0009', 'not', 'substantlate', 'hcpcs', 'g0009', 'for', 'code', 'date', 'of', '01/25/2022', 'the', 'service', 'as', 'medical', 'record', 'did', 'include', 'the', 'not', 'administration', 'of', 'preumococcal', 'vaccine.', 'completed', 'reviews', 'are', 'hcpcs', 'guidelines', 'using', 'the', 'coding', 'guidelines', 'and', 'when', 'applicable', 'cms', 'for', 'medicare', 'member.', 'gchl92nen']
# [[12, 14, 24, 36], [113, 36, 258, 67], [10, 99, 23, 141], [184, 149, 263, 172], [116, 151, 185, 170], [261, 151, 350, 170], [116, 181, 193, 203], [214, 181, 297, 200], [422, 181, 511, 199], [190, 182, 214, 199], [332, 182, 385, 201], [298, 184, 331, 199], [10, 190, 27, 234], [503, 200, 571, 217], [117, 201, 196, 219], [423, 201, 502, 217], [196, 202, 255, 219], [195, 219, 314, 239], [117, 220, 197, 239], [314, 220, 392, 238], [423, 220, 524, 238], [422, 238, 456, 256], [482, 238, 528, 257], [117, 239, 197, 258], [196, 239, 239, 258], [239, 239, 262, 256], [262, 239, 313, 256], [452, 239, 483, 256], [423, 257, 526, 277], [116, 258, 185, 277], [256, 257, 334, 278], [183, 259, 255, 277], [540, 276, 636, 295], [183, 277, 254, 298], [421, 276, 529, 297], [116, 278, 185, 297], [421, 295, 587, 315], [170, 294, 270, 317], [117, 297, 170, 316], [586, 314, 672, 335], [500, 315, 586, 332], [423, 316, 498, 333], [117, 316, 165, 335], [167, 316, 225, 335], [438, 369, 538, 394], [231, 370, 286, 390], [285, 370, 334, 390], [351, 370, 402, 388], [674, 370, 762, 389], [186, 371, 210, 389], [142, 372, 186, 390], [463, 388, 512, 406], [141, 390, 209, 408], [555, 406, 692, 427], [417, 407, 476, 426], [692, 407, 738, 426], [117, 408, 224, 428], [227, 408, 289, 428], [738, 408, 773, 425], [553, 425, 663, 443], [665, 425, 700, 442], [748, 425, 807, 444], [805, 425, 836, 442], [701, 426, 747, 443], [685, 442, 791, 462], [554, 443, 597, 460], [597, 443, 620, 460], [619, 443, 683, 461], [813, 443, 847, 460], [791, 445, 813, 460], [555, 460, 627, 480], [625, 460, 686, 480], [752, 460, 818, 480], [818, 460, 853, 479], [686, 461, 717, 479], [719, 462, 752, 478], [604, 478, 642, 497], [642, 478, 709, 500], [707, 479, 731, 497], [731, 477, 860, 499], [554, 480, 605, 497], [733, 494, 832, 519], [554, 496, 627, 514], [629, 496, 701, 514], [701, 498, 734, 514], [554, 513, 607, 535], [604, 514, 637, 532], [638, 514, 674, 532], [674, 513, 738, 535], [735, 514, 829, 535], [827, 514, 864, 532], [554, 532, 607, 551], [605, 530, 700, 555], [699, 532, 743, 551], [741, 530, 835, 553], [834, 532, 863, 550], [555, 549, 640, 570], [641, 550, 721, 568], [555, 568, 692, 588], [692, 569, 738, 588], [115, 568, 225, 592], [418, 570, 478, 587], [228, 571, 290, 588], [739, 571, 773, 587], [553, 587, 665, 606], [665, 587, 725, 605], [773, 587, 832, 605], [831, 587, 861, 604], [725, 587, 771, 604], [553, 605, 597, 622], [597, 605, 620, 622], [685, 605, 791, 624], [814, 605, 847, 622], [619, 605, 683, 622], [791, 607, 813, 622], [555, 622, 627, 641], [626, 622, 685, 641], [686, 622, 717, 641], [752, 622, 820, 641], [818, 622, 853, 641], [718, 624, 752, 641], [553, 641, 684, 659], [683, 641, 706, 658], [706, 641, 834, 661], [554, 658, 628, 676], [734, 657, 832, 679], [629, 658, 701, 675], [701, 660, 734, 676], [638, 674, 700, 694], [760, 673, 854, 697], [555, 675, 606, 696], [604, 675, 637, 694], [697, 674, 762, 697], [778, 691, 873, 715], [554, 694, 591, 711], [592, 694, 644, 711], [644, 694, 736, 715], [737, 694, 781, 711], [553, 711, 584, 729], [585, 711, 669, 729], [670, 710, 751, 730], [117, 911, 203, 926]]


    # message = hash_file(image_path)
    # print(message)
    ocr_dir("/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test/images", filename_filter="152658538_2.png")
    # 

    if False:
        from pathlib import Path
        home = str(Path.home())
        marie_home = os.path.join(home, '.marie')
        for idx, _path in enumerate(glob.glob(os.path.join(marie_home, "*"))):
            filename = _path.split("/")[-1].replace(".json", "")
            print(filename)
            # json_path = os.path.join(home, '.marie', f'{filename}.json')
            image_path = f"/home/gbugaj/dataset/private/corr-indexer/testdeck-raw-01/images/corr-indexing/test/{filename}"
            print(image_path)
            try:
                main_image(image_path)
            except Exception as e:
                print(e)
                raise e

        
    if False:
        image = Image.open(image_path).convert("RGB")
        results = from_json_file(json_path)
        visualize_icr(image, results)

    if False:
        image = Image.open(image_path).convert("RGB")
        image.show()
        results = obtain_words(image)
        x0 = 0
        y0 = 0
        
        # convert coordinates from x,y,w,h -> x0,y0,x1,y1
        for word in results["words"]:
            x, y, w, h = word["box"]
            w_box = [x0 + x, y0 + y, x0 + x + w, y0 + y + h]
            # word["box"] = w_box
        print(results)

        json_path = os.path.join("/tmp/ocr-results.json")
        with open(json_path, "w") as json_file:
            json.dump(
                results,
                json_file,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=4,
                cls=NumpyEncoder,
            )

        visualize_icr(image, results)

# happy gra day love from izabella