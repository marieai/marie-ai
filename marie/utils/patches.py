import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size

from .resize_image import resize_image


def plot_patches(img_arr, org_img_size, stride=None, size=None):
    """
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image
    Args:
        img_arr (numpy.ndarray): [description]
        org_img_size (tuple): [description]
        stride ([type], optional): [description]. Defaults to None.
        size ([type], optional): [description]. Defaults to None.
    Raises:
        ValueError: [description]
    """

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1


def plot_patches_2(
    img_arr, org_img_size, size_h=None, stride_h=None, size_w=None, stride_w=None
):
    """
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image
    Args:
        img_arr (numpy.ndarray): [description]
        org_img_size (tuple): [description]
        stride ([type], optional): [description]. Defaults to None.
        size ([type], optional): [description]. Defaults to None.
    Raises:
        ValueError: [description]
    """

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    # img_arr.shape >  (4, 256, 256, 3)  [array_zie, H, W, C]
    if size_h is None:
        size_h = img_arr.shape[1]

    if size_w is None:
        size_w = img_arr.shape[2]

    if stride_h is None:
        stride_h = size_h

    if stride_w is None:
        stride_w = size_w

    i_max = (org_img_size[0] // stride_h) + 1 - (size_h // stride_h)
    j_max = (org_img_size[1] // stride_w) + 1 - (size_w // stride_w)

    fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1


def get_patches(img_arr, size=256, stride=256):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.

    Args:
        img_arr (numpy.ndarray): [description]
        size (int, optional): [description]. Defaults to 256.
        stride (int, optional): [description]. Defaults to 256.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        numpy.ndarray: [description]
    """
    # check size and stride
    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping

        for i in range(i_max):
            for j in range(i_max):
                print(i * stride, i * stride + size)
                print(j * stride, j * stride + size)
                patches_list.append(
                    img_arr[
                        i * stride : i * stride + size, j * stride : j * stride + size
                    ]
                )

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 3 or 4")

    return np.stack(patches_list)


def get_patches_2(
    img_arr, size_h=None, stride_h=None, size_w=None, stride_w=None, pad=False
):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.

    Args:
        img_arr (numpy.ndarray): [description]
        size (int, optional): [description]. Defaults to 256.
        stride (int, optional): [description]. Defaults to 256.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        numpy.ndarray: [description]
    """
    # check size and stride

    if pad == True:
        h = img_arr.shape[0]
        w = img_arr.shape[1]
        if w % stride_w != 0:
            adj_w = stride_w * (w // stride_w + 1)
        if h % stride_h != 0:
            adj_h = stride_h * (h // stride_h + 1)

        if adj_w != w or adj_h != h:
            img_arr = resize_image(img_arr, (adj_w, adj_h), color=(255, 255, 255))

    if size_w % stride_w != 0:
        raise ValueError("size % stride must be equal 0")

    if size_h % stride_h != 0:
        raise ValueError("size % stride must be equal 0")

    patches_list = []
    overlapping_h = 0
    overlapping_w = 0

    if stride_h != size_h:
        overlapping_h = (size_h // stride_h) - 1

    if stride_w != size_w:
        overlapping_w = (size_w // stride_w) - 1

    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride_h - overlapping_h
        j_max = img_arr.shape[1] // stride_w - overlapping_w

        for i in range(i_max):
            for j in range(j_max):
                # print(i*stride_h, i*stride_h+size_h)
                # print(j*stride_w, j*stride_w+size_w)
                # img[starty:starty+cropy,startx:startx+cropx]
                snip = img_arr[
                    i * stride_h : i * stride_h + size_h,
                    j * stride_w : j * stride_w + size_w,
                ]

                # this is not ana empyt border
                if np.count_nonzero(snip != 255) > 0:
                    patches_list.append(snip)

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                            i * stride : i * stride + size,
                            j * stride : j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 3 or 4")

    return np.stack(patches_list)


def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    """[summary]

    Args:
        img_arr (numpy.ndarray): [description]
        org_img_size (tuple): [description]
        stride ([type], optional): [description]. Defaults to None.
        size ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        numpy.ndarray: [description]
    """
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (i_max**2)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    print("averaging_value = %s" % (averaging_value))

    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size,
                        layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1
        # TODO add averaging for masks - right now it's just overwritting

        #         for layer in range(nm_layers):
        #             # average some more because overlapping 4 patches
        #             img_bg[stride:i_max*stride, stride:i_max*stride, layer] //= averaging_value
        #             # corners:
        #             img_bg[0:stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value
        #             img_bg[0:stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value

        images_list.append(img_bg)

    return np.stack(images_list)


def reconstruct_from_patches_2(
    img_arr, org_img_size, size_h=None, stride_h=None, size_w=None, stride_w=None
):
    """[summary]

    Args:
        img_arr (numpy.ndarray): [description]
        org_img_size (tuple): [description]
        stride ([type], optional): [description]. Defaults to None.
        size ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        numpy.ndarray: [description]
    """
    # (60, 128, 256, 3)
    # (60, 128, 256)

    print(img_arr.shape)
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size_h is None:
        size_h = img_arr.shape[1]

    if stride_h is None:
        stride_h = size_h

    if size_w is None:
        size_w = img_arr.shape[2]

    if stride_w is None:
        stride_w = size_w

    print(org_img_size)
    print("size_h : %d" % (size_h))
    print("stride_h : %d" % (stride_h))
    print("size_w : %d" % (size_w))
    print("stride_w : %d" % (stride_w))
    print("nm_layers : %d" % (img_arr.shape[3]))

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride_h) + 1 - (size_h // stride_h)
    j_max = (org_img_size[1] // stride_w) + 1 - (size_w // stride_w)

    # FIXME :
    total_nm_images = img_arr.shape[0] // (i_max**2)
    total_nm_images = 1

    nm_images = img_arr.shape[0]

    averaging_value = 4
    print("averaging_value = %s" % (averaging_value))

    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                        i * stride_h : i * stride_h + size_h,
                        j * stride_w : j * stride_w + size_w,
                        layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1

        # TODO add averaging for masks - right now it's just overwritting

        #         for layer in range(nm_layers):
        #             # average some more because overlapping 4 patches
        #             img_bg[stride:i_max*stride, stride:i_max*stride, layer] //= averaging_value
        #             # corners:
        #             img_bg[0:stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, 0:stride, layer] *= averaging_value
        #             img_bg[i_max*stride:i_max*stride+stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value
        #             img_bg[0:stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value

        images_list.append(img_bg)

    return np.stack(images_list)
