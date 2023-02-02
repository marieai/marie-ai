import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import cv2


class MemoryDataset(Dataset):
    def __init__(self, images, opt):
        self.opt = opt
        self.images = images
        self.nSamples = len(images)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        image = self.images[index]
        label = f"img-{index}"
        if type(image) == str:
            try:
                if self.opt.rgb:
                    img = Image.open(image).convert("RGB")  # for color image
                else:
                    img = Image.open(image).convert("L")
            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new("L", (self.opt.imgW, self.opt.imgH))

        elif type(image) == np.ndarray:
            # Convert color to grayscale
            # After normalization image is in 0-1 range  so scale it up to 0-255
            if False:
                image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
                image = image.astype("float32") / 255
                image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image)

            image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)

        if self.opt.rgb:
            img = img.convert("RGB")  # for color image
        else:
            img = img.convert("L")

        return img, label
