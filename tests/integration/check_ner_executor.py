import glob
import os
import time

import PIL.ImageQt
import cv2.cv2
import numpy as np
import transformers
from PIL import Image, ImageDraw, ImageFont

from marie.executor import NerExtractionExecutor
from marie.utils.docs import load_image, docs_from_file, array_from_docs
from marie.utils.image_utils import hash_file, hash_bytes
from marie.utils.json import store_json_object
from marie.utils.utils import ensure_exists

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList
import torch

def load_image(image_path):
    image = read_image(image_path, format="RGB")
    h = image.shape[0]
    w = image.shape[1]

    # img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224, interp=Image.LANCZOS)])
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224, interp=None)])
    t1 = img_trans.apply_image(image).copy()
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)


def process_file(executor: NerExtractionExecutor, img_path: str):
    filename = img_path.split("/")[-1].replace(".png", "")
    checksum = hash_file(img_path)
    docs = None
    kwa = {"checksum": checksum, "img_path": img_path}
    results = executor.extract(docs, **kwa)

    print(results)
    store_json_object(results, f"/tmp/tensors/json/{filename}.json")
    return results


def process_dir(executor: NerExtractionExecutor, image_dir: str):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.tif"))):
        try:
            process_file(executor, img_path)
        except Exception as e:
            print(e)
            # raise e


def check_layoutlmv3(img_path):
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3FeatureExtractor,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3TokenizerFast,
    )

    loaded, frames = load_image(img_path, img_format="pil")
    image = frames[0]

    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        "microsoft/layoutlmv3-base", add_visual_labels=False
    )
    processor = LayoutLMv3Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # not batched
    words = ["hello", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    input_processor = processor(image, words, boxes=boxes, return_tensors="pt")
    input_feat_extract = feature_extractor(image, return_tensors="pt")

    img_tensor = input_feat_extract["pixel_values"]
    p1 = input_feat_extract["pixel_values"].sum()
    p2 = input_processor["pixel_values"].sum()

    print(input_feat_extract["pixel_values"].shape)
    tensor = (img_tensor[0].cpu()).numpy().astype(np.uint8).transpose(1, 2, 0)
    print(tensor.shape)
    img = Image.fromarray(tensor)
    actual_keys = list(input_processor.keys())
    print(actual_keys)
    img.save(f"/home/greg/tmp/tensorAAA.png")
    print(f" input val = {p1} : {p2}")


if __name__ == "__main__":

    # pip install git+https://github.com/huggingface/transformers
    # 4.18.0  -> 4.21.0.dev0 : We should pin it to this version
    print(transformers.__version__)
    # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_162_6505_0_156695212.png"
    # ensure_exists("/tmp/tensors/json")
    # # check_layoutlmv3(img_path)

    p1 = load_image("/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637188.tif")

    executor = NerExtractionExecutor()
    # process_dir(executor, "/home/greg/dataset/assets-private/corr-indexer/validation/")
    # process_dir(executor, "/home/gbugaj/tmp/medrx")

    if True:
        img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_162_6505_0_156695212.png"
        img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation/PID_631_7267_0_156693952.png"

        # img_path = f"/home/greg/dataset/assets-private/corr-indexer/validation_multipage/merged.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1515_8370_0_157159253.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1925_9291_0_157186552.tif"
        # img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
<<<<<<< HEAD
        # img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
        # img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637299.tif"
        # img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637257.tif"
        # img_path = f"/home/gbugaj/tmp/address-001.png"
        # img_path = f"/home/gbugaj/tmp/paid-001.png"

=======
        img_path = f"/home/gbugaj/tmp/PID_1925_9289_0_157186264.tif"
        img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637299.tif"
        img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637194.tif"
        # img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637188.tif"
        # img_path = f"/home/gbugaj/tmp/medrx/PID_1864_9100_0_157637257.tif"
        # img_path = f"/home/gbugaj/tmp/address-001.png"
        # img_path = f"/home/gbugaj/tmp/paid-001.png"
        # img_path = "/home/gbugaj/tmp/PID_1925_9289_0_157186264.png"
>>>>>>> 153268dbe76bdce02f522bb06c61b4bcddb690dd

        docs = docs_from_file(img_path)
        frames = array_from_docs(docs)

        time_nanosec = time.time_ns()
        src = []
        for i, frame in enumerate(frames):
            src = np.append(src, np.ravel(frame))
        checksum = hash_bytes(src)
        print(checksum)
        time_nanosec = (time.time_ns() - time_nanosec) / 1000000000
        print(time_nanosec)

        process_file(executor, img_path)


#
# {'ADDRESS': {'precision': 0.9109961190168175, 'recall': 0.9389333333333333, 'f1': 0.9247537754432041, 'number': 7500}, 'ANSWER': {'precision': 0.8036775631500743, 'recall': 0.8785786802030456, 'f1': 0.8394606654379667, 'number': 19700}, 'BILLED_AMT': {'precision': 0.665943600867679, 'recall': 0.7675, 'f1': 0.7131242740998839, 'number': 400}, 'BILLED_AMT_ANSWER': {'precision': 0.8453389830508474, 'recall': 0.798, 'f1': 0.8209876543209877, 'number': 500}, 'BIRTHDATE': {'precision': 0.9987893462469734, 'recall': 1.0, 'f1': 0.9993943064809208, 'number': 1650}, 'BIRTHDATE_ANSWER': {'precision': 0.9909310761789601, 'recall': 0.9933333333333333, 'f1': 0.9921307506053268, 'number': 1650}, 'CHECK_AMT_ANSWER': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 0}, 'CLAIM_NUMBER': {'precision': 0.8957654723127035, 'recall': 1.0, 'f1': 0.9450171821305842, 'number': 550}, 'CLAIM_NUMBER_ANSWER': {'precision': 0.9068219633943427, 'recall': 0.9083333333333333, 'f1': 0.9075770191507077, 'number': 600}, 'DOCUMENT_CONTROL': {'precision': 0.6760497302369224, 'recall': 0.7895890410958905, 'f1': 0.728421584733982, 'number': 3650}, 'DOS': {'precision': 0.9516240497581202, 'recall': 0.9029508196721312, 'f1': 0.9266487213997309, 'number': 3050}, 'DOS_ANSWER': {'precision': 0.9894217207334274, 'recall': 0.9051612903225806, 'f1': 0.9454177897574124, 'number': 3100}, 'GREETING': {'precision': 0.8545636910732196, 'recall': 0.8783505154639175, 'f1': 0.8662938485002541, 'number': 4850}, 'HEADER': {'precision': 0.8373873873873874, 'recall': 0.8498285714285714, 'f1': 0.843562110039705, 'number': 8750}, 'LETTER_DATE': {'precision': 0.9394863971523011, 'recall': 0.9986486486486487, 'f1': 0.9681645486702477, 'number': 3700}, 'MEMBER_NAME': {'precision': 0.9551820728291317, 'recall': 0.9742857142857143, 'f1': 0.9646393210749648, 'number': 2100}, 'MEMBER_NAME_ANSWER': {'precision': 0.9397031539888683, 'recall': 0.9647619047619047, 'f1': 0.9520676691729324, 'number': 2100}, 'MEMBER_NUMBER': {'precision': 0.9540523292916401, 'recall': 0.980327868852459, 'f1': 0.9670116429495472, 'number': 3050}, 'MEMBER_NUMBER_ANSWER': {'precision': 0.9583066067992303, 'recall': 0.979672131147541, 'f1': 0.9688715953307393, 'number': 3050}, 'PAID_AMT_ANSWER': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 0}, 'PAN': {'precision': 0.8931552587646077, 'recall': 0.8392156862745098, 'f1': 0.8653457339264051, 'number': 2550}, 'PAN_ANSWER': {'precision': 0.9123994921709692, 'recall': 0.8135849056603773, 'f1': 0.860163574705765, 'number': 2650}, 'PARAGRAPH': {'precision': 0.7499283286235, 'recall': 0.8496983758700696, 'f1': 0.7967019818565493, 'number': 21550}, 'PATIENT_NAME': {'precision': 0.9174311926605505, 'recall': 1.0, 'f1': 0.9569377990430622, 'number': 2100}, 'PATIENT_NAME_ANSWER': {'precision': 0.9028595458368377, 'recall': 0.9334782608695652, 'f1': 0.9179136383069688, 'number': 2300}, 'PHONE': {'precision': 0.6933760683760684, 'recall': 0.9984615384615385, 'f1': 0.8184110970996217, 'number': 650}, 'QUESTION': {'precision': 0.8317463811987451, 'recall': 0.9022089552238806, 'f1': 0.8655459778344168, 'number': 16750}, 'URL': {'precision': 0.9643916913946587, 'recall': 1.0, 'f1': 0.9818731117824773, 'number': 1950}, 'overall_precision': 0.842638008060233, 'overall_recall': 0.8957077625570776, 'overall_f1': 0.8683628051479761, 'overall_accuracy': 0.9306041529745965}
