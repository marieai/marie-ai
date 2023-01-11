# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception
import os
import sys


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import copy

import numpy as np
from PIL import Image

import cv2

# import craft functions
from craft_text_detector import (
    Craft,
    empty_cuda_cache,
    export_detected_regions,
    export_extra_results,
    get_prediction,
    load_craftnet_model,
    load_refinenet_model,
    read_image,
)


def crop_poly_low(img, poly):
    """
    find region using the poly points
    create mask using the poly points
    do mask op to crop
    add white bg
    """
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    pts = poly
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2


print("Eval")

# set image path and export folder directory
image = "figures/padded_snippet-HCFA24.jpg"  # can be filepath, PIL image or numpy array
image = (
    "figures/PID_10_5_0_3108.original.tif"  # can be filepath, PIL image or numpy array
)
image = (
    "figures/PID_10_5_0_3110.original.tif"  # can be filepath, PIL image or numpy array
)
image = (
    "figures/PID_10_5_0_3111.original.tif"  # can be filepath, PIL image or numpy array
)
image = (
    "figures/PID_10_5_0_3112.original.tif"  # can be filepath, PIL image or numpy array
)
image = (
    "figures//PID_10_5_0_3108.original.tif"  # can be filepath, PIL image or numpy array
)

image = "/tmp/hicfa/PID_10_5_0_3101.original.tif"
output_dir = "outputs/"

# create a craft instance
# craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# read image
image = read_image(image)

# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

# perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=True,
    long_size=2550  # 1280
    # long_size=1280
)

# export detected text regions

image_paths = copy.deepcopy(image)

exported_file_paths = export_detected_regions(
    image=image_paths,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True,
)

# export heatmap, detection points, box visualization

image_results = copy.deepcopy(image)
export_extra_results(
    image=image_results,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir,
)


def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)


def paste_fragment(overlay, fragment, pos=(0, 0)):
    # You may need to convert the color.
    fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    fragment_pil = Image.fromarray(fragment)
    overlay.paste(fragment_pil, pos)


# output text only blocks
# deepcopy image so that original is not altered
image = copy.deepcopy(image)
regions = prediction_result["boxes"]

# convert imaget to BGR color
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
file_path = os.path.join(output_dir, "image_cv.png")
cv2.imwrite(file_path, image)

pil_image = Image.new("RGB", (image.shape[1], image.shape[0]), color=(255, 255, 255, 0))

for i, region in enumerate(regions):
    region = np.array(region).astype(np.int32).reshape((-1))
    region = region.reshape(-1, 2)
    poly = region.reshape((-1, 1, 2))

    rect = cv2.boundingRect(poly)
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    if h < 15:
        continue

    rect = np.array(rect, dtype=np.int32)
    snippet = crop_poly_low(image, poly)

    x = rect[0]
    y = rect[1]

    # export corpped region
    file_path = os.path.join(output_dir, "crops", "%s.jpg" % (i))
    cv2.imwrite(file_path, snippet)

    paste_fragment(pil_image, snippet, (x, y))


savepath = os.path.join(output_dir, "%s.jpg" % ("txt_overlay"))
pil_image.save(savepath, format="JPEG", subsampling=0, quality=100)

# unload models from gpu
empty_cuda_cache()

# pil_padded = Image.new('RGB', (shape[1] + pad, shape[0] + pad), color=(255,255,255,0))
# paste_fragment(pil_padded, snippet, (pad//2, pad//2))

# savepath = os.path.join(debug_dir, "%s-%s.jpg" % ('padded_snippet' , key))
# pil_padded.save(savepath, format='JPEG', subsampling=0, quality=100)

# cv_snip = np.array(pil_padded)
# snippet = cv2.cvtColor(cv_snip, cv2.COLOR_RGB2BGR)# convert RGB to BGR
