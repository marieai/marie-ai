import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, label, img, opt):
        self.opt = opt
        self.label = label
        self.img = img
        # Fixme: setting batch size to 1 will cause "TypeError: forward() missing 2 required positional arguments: 'input' and 'text'"
        self.nSamples = 2

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        image = self.img
        label = self.label
        if type(image) == str:
            try:
                if self.opt.rgb:
                    img = Image.open(image).convert('RGB')  # for color image
                else:
                    img = Image.open(image).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        elif type(image) == np.ndarray:
            img = Image.fromarray(image)

        if self.opt.rgb:
            img = img.convert('RGB')  # for color image
        else:
            img = img.convert('L')

        return img, label
