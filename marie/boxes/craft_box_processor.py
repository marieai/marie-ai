from __future__ import print_function

import copy
import math
import os
import time
import warnings

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw
from torch.autograd import Variable

import marie.models.craft.craft_utils
import marie.models.craft.file_utils
import marie.models.craft.imgproc
from marie.constants import __model_path__
from marie.boxes.box_processor import (
    BoxProcessor,
    PSMode,
    copyStateDict,
    create_dirs,
    estimate_character_width,
)
from marie.lang import Object
from marie.models import craft
from marie.models.craft.craft import CRAFT
from marie.timer import Timer
from marie.utils.image_utils import imwrite, paste_fragment, viewImage
from marie.utils.utils import ensure_exists

from .line_processor import find_line_number, line_merge

# FIXME : Rework package import
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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


def get_prediction(
    craft_net,
    image,
    text_threshold,
    link_threshold,
    low_text,
    cuda,
    poly,
    canvas_size,
    mag_ratio,
    refine_net=None,
    line_refine_net=None,
):
    net = craft_net
    show_time = True
    t0 = time.time()

    ensure_exists("/tmp/fragments")
    # mag_ratio = 1.5
    # resize
    img_resized, target_ratio, size_heatmap = craft.imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    cv2.imwrite("/tmp/fragments/image.png", img_resized)
    # preprocessing
    x = craft.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft.craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    # render_img = score_link
    ret_score_text = craft.imgproc.cvt2HeatmapImg(render_img)
    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    # DO LINE DETECTION
    # refine link
    lines_bboxes = []
    if line_refine_net is not None:
        with torch.no_grad():
            y_refiner = line_refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        linkmap = score_link
        textmap = score_text
        ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

        text_score_comb = np.clip(text_score + link_score, 0, 1) * 255
        text_score_comb = link_score * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_img = cv2.morphologyEx(
            text_score_comb, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        if True:
            cv2.imwrite("/tmp/fragments/lines-morph-craft.png", line_img)
            cv2.imwrite(
                os.path.join("/tmp/fragments/", "h-linkmap-craft.png"), linkmap * 255
            )
            cv2.imwrite(
                os.path.join("/tmp/fragments/", "h-textmap-craft.png"), textmap * 255
            )
            cv2.imwrite(
                os.path.join("/tmp/fragments/", "h-text_score_comb-craft.png"),
                text_score_comb * 255,
            )

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            line_img.astype(np.uint8), connectivity=4
        )

        h, w = line_img.shape
        overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
        line_bboxes = []

        for k in range(1, nLabels):
            size = stats[k, cv2.CC_STAT_AREA]
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            # size filtering
            # if h < 2:
            #     continue
            box = x, y, w, h
            line_bboxes.append(box)
            color = list(np.random.random(size=3) * 256)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)

        cv2.imwrite("/tmp/fragments/img_line-craft.png", overlay)
        lines_bboxes = line_merge(overlay, line_bboxes)

        # coordinate adjustment
        ratio_net = 2
        for k in range(len(lines_bboxes)):
            lines_bboxes[k] = [
                int(lines_bboxes[k][0] * ratio_w * ratio_net),
                int(lines_bboxes[k][1] * ratio_h * ratio_net),
                int(lines_bboxes[k][2] * ratio_w * ratio_net),
                int(lines_bboxes[k][3] * ratio_h * ratio_net),
            ]

        # overlay = np.ones((_h, _w, 3), dtype=np.uint8) * 255
        img_line = copy.deepcopy(image)
        viz_img = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)
        viz_img = Image.fromarray(viz_img)
        draw = ImageDraw.Draw(viz_img, "RGBA")

        for box in lines_bboxes:
            x, y, w, h = box
            color = list(np.random.random(size=3) * 256)
            # cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
            draw.rectangle(
                [x, y, x + w, y + h],
                outline="#993300",
                fill=(
                    int(np.random.random() * 256),
                    int(np.random.random() * 256),
                    int(np.random.random() * 256),
                    125,
                ),
                width=1,
            )

        viz_img.save(
            os.path.join("/tmp/fragments", f"overlay_refiner-final-craft.png"),
            format="PNG",
        )
    # estimate_character_width(render_img, boxes)
    return boxes, polys, ret_score_text, lines_bboxes


class BoxProcessorCraft(BoxProcessor):
    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = os.path.join(__model_path__, "craft"),
        cuda: bool = True,
    ):
        super().__init__(work_dir, models_dir, cuda)
        print("Box processor [craft, cuda={}]".format(cuda))
        self.craft_net, self.refine_net = self.__load(models_dir)

    def unload(self):
        """Unload the model from GPU/CPU"""
        del self.craft_net
        del self.refine_net

    def __load(self, models_dir: str):
        # load models
        args = Object()
        args.trained_model = os.path.join(models_dir, "craft_mlt_25k.pth")
        args.refiner_model = os.path.join(models_dir, "craft_refiner_CTW1500.pth")

        cuda = self.cuda
        refine = True
        # load net
        net = CRAFT()  # initialize

        print("Loading weights from checkpoint (" + args.trained_model + ")")
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        else:
            net.load_state_dict(
                copyStateDict(torch.load(args.trained_model, map_location="cpu"))
            )

        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            # this option is good if the input size does not change each time
            # cudnn.benchmark = True

        net.eval()

        # LinkRefiner
        refine_net = None
        if refine:
            from craft.refinenet import RefineNet

            refine_net = RefineNet()
            print(
                "Loading weights of refiner from checkpoint ("
                + args.refiner_model
                + ")"
            )
            if cuda:
                refine_net.load_state_dict(
                    copyStateDict(torch.load(args.refiner_model))
                )
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(
                    copyStateDict(torch.load(args.refiner_model, map_location="cpu"))
                )

            refine_net.eval()
            args.poly = True

        t = time.time()

        return net, refine_net

    def psm_word(self, image):
        """
        Treat the image as a single word.
        """
        w = image.shape[1]
        bboxes, polys, score_text, lines = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.6,
            link_threshold=0.8,
            low_text=0.3,
            cuda=self.cuda,
            poly=False,
            # canvas_size=1280,#w + w // 2,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1,
            line_refine_net=None,
        )

        return bboxes, polys, score_text, lines

    def psm_sparse(self, image):
        """
        Find as much text as possible (default).
        """
        w = image.shape[1]
        bboxes, polys, score_text, lines = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=None,  # self.refine_net,
            text_threshold=0.7,
            link_threshold=0.45,
            low_text=0.3,
            cuda=self.cuda,
            poly=False,
            # canvas_size=1280,#w + w // 2,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1.0,
            line_refine_net=self.refine_net,
        )

        return bboxes, polys, score_text, lines

    def psm_line(self, image):
        """
        Treat the image as a single text line.
        """
        w = image.shape[1]
        bboxes, polys, score_text, lines = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=None,  # self.refine_net,
            text_threshold=0.4,
            link_threshold=0.2,
            low_text=0.3,
            cuda=self.cuda,
            poly=False,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1,
            line_refine_net=None,
        )

        return bboxes, polys, score_text, lines

    def psm_raw_line(self, image):
        """
        Treat the image as a single text line.
        """
        w = image.shape[1]
        bboxes, polys, score_text, lines = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.4,
            link_threshold=0.2,
            low_text=0.5,
            cuda=self.cuda,
            poly=False,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1,
            line_refine_net=None,
        )

        return bboxes, polys, score_text, lines

    def psm_multiline(self, image):
        """
        Treat the image as a single word.
        """
        w = image.shape[1]
        bboxes, polys, score_text, lines = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.6,
            link_threshold=0.3,
            low_text=0.3,
            cuda=self.cuda,
            poly=False,
            # canvas_size=1280,#w + w // 2,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1,
            line_refine_net=self.refine_net,
        )

        return bboxes, polys, score_text, lines

    # @Timer(text="BoundingBoxes in {:.2f} seconds")
    def extract_bounding_boxes(self, _id, key, img, psm=PSMode.SPARSE):
        # print("Extracting bounding boxes : mode={} key={}, id={}".format(psm, key, _id))

        if img is None:
            raise Exception("Input image can't be empty")
        try:
            crops_dir, debug_dir, lines_dir, mask_dir = create_dirs(
                self.work_dir, _id, key
            )
            image = copy.deepcopy(img)
            w = image.shape[1]  # 1280
            lines_bboxes = []

            # Page Segmentation Model
            if psm == PSMode.SPARSE:
                bboxes, polys, score_text, lines_bboxes = self.psm_sparse(image)
            # elif psm == PSMode.WORD:
            #     bboxes, polys, score_text, lines_bboxes = self.psm_word(image_norm)
            elif psm == PSMode.LINE:
                bboxes, polys, score_text, lines_bboxes = self.psm_line(image)
            elif psm == PSMode.MULTI_LINE:
                bboxes, polys, score_text, lines_bboxes = self.psm_multiline(image)
            elif psm == PSMode.RAW_LINE or psm == PSMode.WORD:
                # NOTE: Semantics have changed and now both RAW_LINE and WORD are same
                # this needs to be handled better, there is no need to have the segmentation for RAW_LINES
                # as we treat the whole line as BBOX
                # bboxes, polys, score_text = self.psm_raw_line(image_norm)
                rect_from_poly = []
                fragments = []
                rect_line_numbers = []
                prediction_result = dict()

                # x, y, w, h = box
                w = image.shape[1]
                h = image.shape[0]
                rect_from_poly.append([0, 0, w, h])
                fragments.append(image)
                rect_line_numbers.append(0)

                return (
                    rect_from_poly,
                    fragments,
                    rect_line_numbers,
                    prediction_result,
                    lines_bboxes,
                )
            else:
                raise Exception(f"PSM mode not supported : {psm}")

            prediction_result = dict()
            prediction_result["bboxes"] = bboxes
            prediction_result["polys"] = polys
            prediction_result["heatmap"] = score_text

            # deepcopy image so that original is not altered
            image = copy.deepcopy(image)
            pil_image = Image.new(
                "RGB", (image.shape[1], image.shape[0]), color=(255, 255, 255, 0)
            )

            rect_from_poly = []
            rect_line_numbers = []
            fragments = []
            ms = int(time.time() * 1000)

            max_h = image.shape[0]
            max_w = image.shape[1]

            for idx, region in enumerate(bboxes):
                region = np.array(region).astype(np.int32).reshape((-1))
                region = region.reshape(-1, 2)
                poly = region.reshape((-1, 1, 2))
                box = cv2.boundingRect(poly)
                box = np.array(box).astype(np.int32)

                if True and len(poly) == 4:
                    hexp = 4
                    vexp = 4
                    box = [
                        max(0, box[0] - hexp // 2),
                        max(0, box[1] - vexp // 2),
                        min(max_w, box[2] + hexp),
                        min(max_h, box[3] + vexp),
                    ]
                    poly_exp = [
                        [[box[0], box[1]]],
                        [[box[0] + box[2], box[1]]],
                        [[box[0] + box[2], box[1] + box[3]]],
                        [[box[0], box[1] + box[3]]],
                    ]
                    poly = np.array(poly_exp)

                x, y, w, h = box
                snippet = crop_poly_low(image, poly)
                line_number = find_line_number(lines_bboxes, box)

                fragments.append(snippet)
                rect_from_poly.append(box)
                rect_line_numbers.append(line_number)

                # export cropped region
                # FIXME : Add debug flags
                if True:
                    file_path = os.path.join(crops_dir, "%s_%s.jpg" % (ms, idx))
                    cv2.imwrite(file_path, snippet)

                paste_fragment(pil_image, snippet, (x, y))
                # break
            # FIXME : Add debug flags
            if True:
                savepath = os.path.join(debug_dir, "%s.png" % "txt_overlay")
                pil_image.save(savepath, format="PNG", subsampling=0, quality=100)

            debug_dir = "/tmp/fragments"
            savepath = os.path.join(debug_dir, "txt_overlay.jpg")
            pil_image.save(savepath, format="JPEG", subsampling=0, quality=100)
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            stacked = np.hstack((cv_img, img))
            save_path = os.path.join(debug_dir, "stacked.png")
            imwrite(save_path, stacked)

            # we can't return np.array here as t the 'fragments' will throw an error
            # ValueError: could not broadcast input array from shape (42,77,3) into shape (42,)
            return (
                rect_from_poly,
                fragments,
                rect_line_numbers,
                prediction_result,
                lines_bboxes,
            )
        except Exception as ex:
            raise ex
