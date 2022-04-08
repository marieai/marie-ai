# -*- coding: utf-8 -*-
# Add parent to the search path so we can reference the modules(craft, pix2pix) here without throwing and exception
from __future__ import print_function

import os
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from timer import Timer
from utils.image_utils import paste_fragment
from line_processor import line_refiner
from line_processor import find_line_index

import copy
import cv2
import numpy as np

from models import craft
from models.craft.craft import CRAFT

import models.craft.craft_utils
import models.craft.file_utils
import models.craft.imgproc

from PIL import Image
from boxes.box_processor import PSMode, estimate_character_width, BoxProcessor, copyStateDict, Object
from utils.utils import ensure_exists

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
):
    net = craft_net
    show_time = True
    t0 = time.time()

    image[image >= 1] = [200]
    # resize
    img_resized, target_ratio, size_heatmap = craft.imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    # image[image == 255] = 80

    cv2.imwrite("/tmp/fragments/img_resized.png", img_resized)
    # preprocessing
    x = craft.imgproc.normalizeMeanVariance(img_resized)
    cv2.imwrite("/tmp/fragments/norm.png", x)

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
    # render_img = np.hstack((render_img, score_link))
    # render_img = score_link
    ret_score_text = craft.imgproc.cvt2HeatmapImg(render_img)
    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    cv2.imwrite("/tmp/fragments/render_img.png", render_img)
    cv2.imwrite("/tmp/fragments/ret_score_text.png", ret_score_text)

    estimate_character_width(render_img, boxes)
    return boxes, polys, ret_score_text


class BoxProcessorCraft(BoxProcessor):
    def __init__(
        self,
        work_dir: str = "/tmp/boxes",
        models_dir: str = "./model_zoo/craft",
        cuda: bool = True,
    ):
        super().__init__(work_dir, models_dir, cuda)
        print("Box processor [craft, cuda={}]".format(cuda))
        self.craft_net, self.refine_net = self.__load(models_dir)

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
            net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location="cpu")))

        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        # LinkRefiner
        refine_net = None
        if refine:
            from craft.refinenet import RefineNet

            refine_net = RefineNet()
            print("Loading weights of refiner from checkpoint (" + args.refiner_model + ")")
            if cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location="cpu")))

            refine_net.eval()
            args.poly = True

        t = time.time()

        return net, refine_net

    def psm_word(self, image):
        """
        Treat the image as a single word.
        """
        w = image.shape[1]
        bboxes, polys, score_text = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.6,
            link_threshold=0.4,
            low_text=0.3,
            cuda=self.cuda,
            poly=False,
            # canvas_size=1280,#w + w // 2,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1,
        )

        return bboxes, polys, score_text

    def psm_sparse(self, image):
        """
        Find as much text as possible (default).
        """
        w = image.shape[1]
        bboxes, polys, score_text = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=None,  # self.refine_net,
            text_threshold=0.6,
            link_threshold=0.4,
            low_text=0.3,
            cuda=self.cuda,
            poly=False,
            # canvas_size=1280,#w + w // 2,
            canvas_size=w,
            # canvas_size=w + w // 2,
            mag_ratio=1,
        )

        return bboxes, polys, score_text

    def psm_line(self, image):
        """
        Treat the image as a single text line.
        """
        w = image.shape[1]
        bboxes, polys, score_text = get_prediction(
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
        )

        return bboxes, polys, score_text

    @Timer(text="BoundingBoxes in {:.2f} seconds")
    def extract_bounding_boxes(self, _id, key, img, psm=PSMode.SPARSE):
        print("Extracting bounding boxes : mode={} key={}, id={}".format(psm, key, _id))

        try:
            debug_dir = ensure_exists(os.path.join(self.work_dir, _id, "bounding_boxes", key, "debug"))
            crops_dir = ensure_exists(os.path.join(self.work_dir, _id, "bounding_boxes", key, "crop"))
            lines_dir = ensure_exists(os.path.join(self.work_dir, _id, "bounding_boxes", key, "lines"))
            mask_dir = ensure_exists(os.path.join(self.work_dir, _id, "bounding_boxes", key, "mask"))

            image = copy.deepcopy(img)
            w = image.shape[1]  # 1280
            # Inverting the image makes box detection substantially better in many case, I think that this make sense
            # to make this a configurable option
            # Make this a configuration
            image_norm = image
            # image_norm = 255 - image
            cv2.imwrite(os.path.join("/tmp/icr/fields/", key, "NORM_%s.png" % _id), image_norm)
            # TODO : Externalize as config

            # Page Segmentation Model
            if psm == PSMode.SPARSE:
                bboxes, polys, score_text = self.psm_sparse(image_norm)
            elif psm == PSMode.WORD:
                bboxes, polys, score_text = self.psm_word(image_norm)
            elif psm == PSMode.LINE:
                bboxes, polys, score_text = self.psm_line(image_norm)
            else:
                raise Exception(f"PSM mode not supported : {psm}")

            prediction_result = dict()
            prediction_result["bboxes"] = bboxes
            prediction_result["polys"] = polys
            prediction_result["heatmap"] = score_text
            regions = bboxes

            result_folder = "./result/"
            if not os.path.isdir(result_folder):
                os.mkdir(result_folder)

            # save score text
            filename = _id
            mask_file = result_folder + "/res_" + filename + "_mask.jpg"
            cv2.imwrite(mask_file, score_text)

            # deepcopy image so that original is not altered
            image = copy.deepcopy(image)
            pil_image = Image.new("RGB", (image.shape[1], image.shape[0]), color=(0, 255, 0, 0))

            rect_from_poly = []
            rect_line_numbers = []
            fragments = []
            ms = int(time.time() * 1000)

            max_h = image.shape[0]
            max_w = image.shape[1]

            for idx, region in enumerate(regions):
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

                # apply connected component analysis to determine if the image is touching the borders and if so expand them
                if False:
                    gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
                    (
                        nLabels,
                        labels,
                        stats,
                        centroids,
                    ) = cv2.connectedComponentsWithStats(gray.astype(np.uint8), connectivity=4)
                    # initialize an output mask to store all characters
                    mask = np.zeros(gray.shape, dtype="uint8")
                    print(f"nLabels : {nLabels}")
                    for i in range(1, nLabels):
                        # extract the connected component statistics for the current label
                        x = stats[i, cv2.CC_STAT_LEFT]
                        y = stats[i, cv2.CC_STAT_TOP]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]
                        area = stats[i, cv2.CC_STAT_AREA]
                        componentMask = (labels == i).astype("uint8") * 255
                        mask = cv2.bitwise_or(mask, componentMask)
                        print(f"component x/y : {x}, {y}, {w}, {h}")

                    # export cropped region
                    file_path = os.path.join(mask_dir, "%s_%s.jpg" % (ms, idx))
                    cv2.imwrite(file_path, mask)

                # FIXME  : This is really slow
                # lines = line_refiner(image, bboxes, _id, lines_dir)
                # line_number = find_line_index(lines, box)
                line_number = 0

                fragments.append(snippet)
                rect_from_poly.append(box)
                rect_line_numbers.append(line_number)

                # export cropped region
                # FIXME : Add debug flags
                if True:
                    file_path = os.path.join(crops_dir, "%s_%s.jpg" % (ms, idx))
                    cv2.imwrite(file_path, snippet)

                # After normalization image is in 0-1 range
                # snippet = (snippet * 255).astype(np.uint8)
                paste_fragment(pil_image, snippet, (x, y))

                # break
            savepath = os.path.join(debug_dir, "%s.jpg" % ("txt_overlay"))
            pil_image.save(savepath, format="JPEG", subsampling=0, quality=100)

            # we can't return np.array here as t the 'fragments' will throw an error
            # ValueError: could not broadcast input array from shape (42,77,3) into shape (42,)
            return rect_from_poly, fragments, rect_line_numbers, prediction_result
        except Exception as ident:
            raise ident


if __name__ == "__main__":
    t = time.time()
    """ For testing images in a folder """
    test_folder = "../examples/set-001/test"
    image_list, _, _ = craft.file_utils.get_files(test_folder)
    result_folder = "./result/"

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    boxer = BoxProcessorCraft(work_dir="/tmp/boxes", models_dir="../model_zoo/craft")
    # load data
    for k, image_path in enumerate(image_list):
        print(
            "Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path),
            end="\r",
        )
        image = craft.imgproc.loadImage(image_path)
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        (
            rect_from_poly,
            fragments,
            rect_line_numbers,
            prediction_result,
        ) = boxer.extract_bounding_boxes(filename, "key", image)
        heatmap = prediction_result["heatmap"]
        polys = prediction_result["polys"]

        # save score text
        mask_file = result_folder + "/res_" + filename + "_mask.jpg"
        cv2.imwrite(mask_file, heatmap)
        craft.file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
        print("elapsed time : {}s".format(time.time() - t))

    print("Total time : {}s".format(time.time() - t))
