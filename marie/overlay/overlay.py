import os
import sys
import time
from shutil import copyfile

import cv2
import numpy as np

from marie.base_handler import BaseHandler
from marie.models.pix2pix.data import create_dataset
from marie.models.pix2pix.models import create_model
from marie.models.pix2pix.options.test_options import TestOptions
from marie.models.pix2pix.util.util import tensor2im
from marie.timer import Timer
from marie.utils.image_utils import imwrite, read_image, viewImage
from marie.utils.utils import ensure_exists

# Add parent to the search path, so we can reference the module here without throwing and exception
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

debug_visualization_enabled = False


class OverlayProcessor(BaseHandler):
    def __init__(self, work_dir, cuda: bool = False):
        print("OverlayProcessor [cuda={}]".format(cuda))
        self.cuda = cuda
        self.work_dir = work_dir
        self.opt, self.model = self.__setup(cuda)
        self.initialized = False

    @staticmethod
    def __setup(cuda):
        """
        Model setup
        """
        gpu_id = "0" if cuda else "-1"

        args = [
            "--dataroot",
            "./data",
            "--name",
            "template_mask_global",
            "--model",
            "test",
            "--netG",
            "global",
            "--direction",
            "AtoB",
            "--model",
            "test",
            "--dataset_mode",
            "single",
            "--gpu_id",
            gpu_id,
            "--norm",
            "instance",
            "--preprocess",
            "none",
            "--checkpoints_dir",
            "./model_zoo/overlay",
        ]

        opt = TestOptions().parse(args)
        # hard-code parameters for test
        opt.eval = True
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.no_dropout = True
        opt.display_id = -1

        model = create_model(opt)
        model.setup(opt)
        model.eval()

        print("Model setup complete")
        return opt, model

    @Timer(text="__extract_segmentation_mask in {:.4f} seconds")
    def __extract_segmentation_mask(self, img, dataroot_dir, work_dir, debug_dir):
        """
        Extract overlay segmentation mask for the image
        """
        model = self.model
        opt = self.opt
        opt.dataroot = dataroot_dir
        image_dir = work_dir
        name = "overlay"

        # create a dataset given opt.dataset_mode and other options
        dataset = create_dataset(opt)
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            # Debug
            if False:
                for label, im_data in visuals.items():
                    image_numpy = tensor2im(im_data)
                    print(f"Tensor debug[{label}]: {image_numpy.shape}")
                    # Tensor is in RGB format OpenCV requires BGR
                    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
                    image_name = "%s_%s.png" % (name, label)
                    save_path = os.path.join(debug_dir, image_name)
                    imwrite(save_path, image_numpy)
                    viewImage(image_numpy, "Tensor Image")

            label = "fake"
            fake_im_data = visuals["fake"]
            image_numpy = tensor2im(fake_im_data)
            # Tensor is in RGB format OpenCV requires BGR
            image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(image_dir, "%s_%s.png" % (name, label))

            # testing only
            if debug_visualization_enabled:
                imwrite(save_path, image_numpy)
            # viewImage(image_numpy, 'Prediction image')

            # TODO : Figure out why after the forward pass it is possible
            # to have different sizes(transforms have not been applied).
            # This is a work around for now

            if img.shape != image_numpy.shape:
                print(f"WARNING(FIXME): overlay shapes do not match(adjusting): {img.shape} != {image_numpy.shape}")
                return image_numpy[: img.shape[0], : img.shape[1], :]

            return image_numpy

    @staticmethod
    def blend_to_text(real_img, fake_img):
        """
        Blend real and fake(generated) images together to generate extracted text mask
        """
        real = read_image(real_img)
        fake = read_image(fake_img)

        print(f"Image shapes (real, fake) : {real.shape} != {fake.shape}")
        # Sizes of input arguments do not match
        # this happens sometimes after a forward pass, the FAKE image is larger than the input
        if real.shape != fake.shape:
            raise Exception(f"Sizes of input arguments do not match(real, fake) : {real.shape} != {fake.shape}")

        blended_img = cv2.bitwise_or(real, fake)
        blended_img[blended_img >= 120] = [255]

        return blended_img

    @Timer(text="Segmented in {:.2f} seconds")
    def segment(self, document_id: str, img_path: str):
        """
        Form overlay segmentation
        """
        print(f"Creating overlay for : {img_path}")
        if not os.path.exists(img_path):
            raise Exception("File not found : {}".format(img_path))

        name = document_id
        work_dir = os.path.join(self.work_dir, name, "work")
        debug_dir = os.path.join(self.work_dir, name, "debug")
        dataroot_dir = os.path.join(self.work_dir, name, "dataroot_overlay")

        ensure_exists(self.work_dir)
        ensure_exists(work_dir)
        ensure_exists(debug_dir)
        ensure_exists(dataroot_dir)

        dst_file_name = os.path.join(dataroot_dir, f"overlay_{name}.png")
        if False and not os.path.exists(dst_file_name):
            copyfile(img_path, dst_file_name)

        copyfile(img_path, dst_file_name)
        real_img = cv2.imread(dst_file_name)
        # viewImage(img, 'Source Image')

        fake_mask = self.__extract_segmentation_mask(real_img, dataroot_dir, work_dir, debug_dir)

        # Unable to segment return empty mask
        if np.array(fake_mask).size == 0:
            print(f"Unable to segment image :{img_path}")
            return None, None

        # Causes by forward pass, incrementing size of the ouput layer
        if real_img.shape != fake_mask.shape:
            print(
                "WARNING(FIXME/ADJUSTING): Sizes of input arguments do not match(real,"
                f" fake) : {real_img.shape} != {fake_mask.shape}"
            )
            # tmp_img = np.ones((fake.shape[0], fake.shape[1], 3), dtype = np.uint8) * 255
            h = min(real_img.shape[0], fake_mask.shape[0])
            w = min(real_img.shape[1], fake_mask.shape[1])
            # # make a blank image
            # img_r = np.ones((h, w), dtype = np.uint8) * 255
            # img_f = np.ones((h, w), dtype = np.uint8) * 255
            real_img = real_img[:h, :w, :]
            fake_mask = fake_mask[:h, :w, :]

            print(f"Image shapes after(real, fake) : {real_img.shape} : {fake_mask.shape}")

        blended = self.blend_to_text(real_img, fake_mask)
        # viewImage(segmask, 'segmask')
        tm = time.time_ns()
        if debug_visualization_enabled:
            imwrite(os.path.join(debug_dir, "overlay_{}.png".format(tm)), fake_mask)

        # real, fake, blended
        return real_img, fake_mask, blended
