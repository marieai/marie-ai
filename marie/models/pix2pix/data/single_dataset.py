from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        )
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        def __check_image_size(img, size_divisible):
            """Check if the image size is a multiple of the specified value"""
            w, h = img.size
            if w % size_divisible != 0 or h % size_divisible != 0:
                raise ValueError(
                    "The image size needs to be a multiple of {}. Got {}.".format(
                        size_divisible, img.size
                    )
                )

        def frame_to_divisible(img, size_divisible):
            """Resize the image to a size that is a multiple of the specified value"""
            w, h = img.size
            new_h = int(round(h / size_divisible) * size_divisible)
            new_w = int(round(w / size_divisible) * size_divisible)

            new_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            new_img.paste(img, (0, 0))
            return new_img

        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert("RGB")

        # This is required when working with the 256x256 images from the CycleGAN paper
        # images have to be divisible by 32
        A_img = frame_to_divisible(A_img, 32)
        __check_image_size(A_img, 32)

        A = self.transform(A_img)

        if False:
            # https://github.com/pytorch/vision/releases/tag/v0.8.0
            import torchvision
            import torchvision.transforms.functional as TF

            # tensor_image is a CxHxW uint8 Tensor

            # from torchvision.io import read_image
            # tensor_image = torchvision.io.read_image(A_path)

            tensor_image = TF.to_tensor(A_img)
            A = self.transform(tensor_image.cuda())

        return {"A": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
