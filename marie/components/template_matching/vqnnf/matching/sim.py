import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sahi.slicing import slice_image
from scipy.ndimage import _ni_support, binary_erosion
from scipy.ndimage.morphology import distance_transform_edt, generate_binary_structure
from sewar import ergas, msssim, rase, rmse, rmse_sw, sam, scc, uqi, vifp
from skimage import metrics

idx = 0


# ref https://datascience.stackexchange.com/questions/48642/how-to-measure-the-similarity-between-two-images


def preprocessing_accuracy(label_true, label_pred, n_class=2):
    #
    if n_class == 2:
        output_zeros = np.zeros_like(label_pred)
        output_ones = np.ones_like(label_pred)
        label_pred = np.where((label_pred > 0.5), output_ones, output_zeros)
    #
    label_pred = np.asarray(label_pred, dtype="int8")
    label_true = np.asarray(label_true, dtype="int8")
    mask = (label_true >= 0) & (label_true < n_class) & (label_true != 8)
    label_true = label_true[mask].astype(int)
    label_pred = label_pred[mask].astype(int)
    return label_true, label_pred


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result, reference = preprocessing_accuracy(reference, result)
    # reference = reference.cpu().detach().numpy()
    # result = result.cpu().detach().numpy()
    #
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError(
            "The first supplied array does not contain any binary object."
        )
    if 0 == np.count_nonzero(reference):
        raise RuntimeError(
            "The second supplied array does not contain any binary object."
        )
        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )
    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds


def hd95(result, reference, voxelspacing=None, connectivity=2):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    #
    hd95_mean = np.nanmean(hd95)
    return hd95_mean


def slicer(image, patch_size_h, patch_size_w) -> list:
    slice_image_result = slice_image(
        image=image,
        output_file_name="patch_",  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir="/tmp/dim/patches",  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=patch_size_h,
        slice_width=patch_size_w,
        overlap_height_ratio=0,
        overlap_width_ratio=0,
        auto_slice_resolution=True,
    )

    image_list = []
    for idx, slice_result in enumerate(slice_image_result):
        patch = slice_result["image"]
        starting_pixel = slice_result["starting_pixel"]
        image_list.append(patch)
        # shift_amount_list.append(starting_pixel)
    return image_list


def extract_hog_features(gray, channels, device):
    # Apply HOG to the input image
    from skimage.feature import hog

    fd, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2),
        visualize=True,
    )

    return fd, hog_image


def similarity_score_color(template, query, metric) -> float:
    stacked = np.hstack((template, query))
    score = 0

    # find the max dimension of the two images and resize the other image to match it
    # we add 16 pixels to the max dimension to ensure that the image does not touch the border
    max_h = int(max(template.shape[0], query.shape[0]))
    max_w = int(max(template.shape[1], query.shape[1]))

    p1 = cv2.resize(template, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
    p2 = cv2.resize(query, (max_w, max_h), interpolation=cv2.INTER_CUBIC)

    s0 = uqi(p1, p2)  # value between 0 and 1
    # s0 = scc(p1, p2)  # value between 0 and 1

    score = s0
    cv2.imwrite(
        f"/tmp/dim/stacked_color_{idx}_{round(score, 4)}.png",
        stacked,
    )

    return 0


def similarity_score(template, query, metric) -> float:
    image1_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    image1_gray = crop_to_content(image1_gray, content_aware=True)
    image2_gray = crop_to_content(image2_gray, content_aware=True)

    # find the max dimension of the two images and resize the other image to match it
    # we add 16 pixels to the max dimension to ensure that the image does not touch the border
    max_h = int(max(image1_gray.shape[0], image2_gray.shape[0])) + 0
    max_w = int(max(image1_gray.shape[1], image2_gray.shape[1])) + 0

    image1_gray = cv2.resize(image1_gray, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
    image2_gray = cv2.resize(image2_gray, (max_w, max_h), interpolation=cv2.INTER_CUBIC)

    original_image1_gray = image1_gray.copy()
    original_image2_gray = image2_gray.copy()

    if False:
        image1_gray, coord = resize_image(
            image1_gray,
            desired_size=(max_h, max_w),
            color=(255, 255, 255),
            keep_max_size=False,
        )

        image2_gray, coord = resize_image(
            image2_gray,
            desired_size=(max_h, max_w),
            color=(255, 255, 255),
            keep_max_size=False,
        )

    # ensure that the shapes are the same for
    if image1_gray.shape != image2_gray.shape:
        raise ValueError(
            f"Template and prediction snippet have different shapes: {image1_gray.shape} vs {image2_gray.shape}"
        )

    global idx
    idx += 1

    fd_1, hog_image_1 = extract_hog_features(image1_gray, 1, "cpu")
    fd_2, hog_image_2 = extract_hog_features(image2_gray, 1, "cpu")

    image1_gray = hog_image_1
    image2_gray = hog_image_2

    cv2.imwrite(f"/tmp/dim/{idx}_hog_1.png", hog_image_1)
    cv2.imwrite(f"/tmp/dim/{idx}_hog_2.png", hog_image_2)

    # # 0 = Background, 1 = Object
    # bin_image1 = np.where(hog_image_1 > 0, 0, 1)
    # bin_image2 = np.where(hog_image_2 > 0, 0, 1)
    # hd95_score = hd95(bin_image1, bin_image2, connectivity=1, voxelspacing=1)

    # save for debugging

    patch_size_h = image1_gray.shape[0] // 2
    patch_size_w = image1_gray.shape[1] // 2

    # patch_size_h = 16
    # patch_size_w = 16

    # ensure that the patch size is at least 7x7
    if patch_size_h < 7:
        patch_size_h = 7
    if patch_size_w < 7:
        patch_size_w = 7

    # ensure that the patch is not larger than the image
    patch_size_h = min(patch_size_h, image1_gray.shape[0])
    patch_size_w = min(patch_size_w, image1_gray.shape[1])

    patches_t = slicer(image1_gray, patch_size_h, patch_size_w)
    patches_m = slicer(image2_gray, patch_size_h, patch_size_w)

    ix = 1
    s0_total = 0
    s1_total = 0
    s2_total = 0

    slices_len = len(patches_t)
    for p1, p2 in zip(patches_t, patches_m):
        # s0 = rmse(p1, p2)  # value between 0 and 255
        # s0, _ = rmse_sw(p1, p2, ws=16)  # value between 0 and 255
        # s0 = 1 - s0 / 255  # normalize rmse to 1

        s0 = uqi(p1, p2, ws=12)  # value between 0 and 1
        s1 = scc(p1, p2)  # value between 0 and 1
        s2 = metrics.structural_similarity(p1, p2, full=True, data_range=1)[0]

        # # 0 = Background, 1 = Object
        # bin_image1 = np.where(p1 > 0, 0, 1)
        # bin_image2 = np.where(p2 > 0, 0, 1)
        # hd95_score = hd95(bin_image1, bin_image2, connectivity=1, voxelspacing=1)
        # hd95_total += hd95_score

        s0_total += max(s0, 0)
        s1_total += max(s1, 0)
        s2_total += max(s2, 0)
        ix += 1

    s0_total_norm = s0_total / slices_len
    s1_total_norm = s1_total / slices_len
    s2_total_norm = s2_total / slices_len
    score = (s0_total_norm + s1_total_norm + s2_total_norm) / 3
    score = (max(s0_total_norm, s1_total_norm) + s2_total_norm) / 2

    if True:
        print(f"score {idx} : ", score)
        print(f"s0_total {idx} :", s0_total)
        print(f"s1_total {idx} :", s1_total)
        print(f"s0_total_norm {idx} ", s0_total_norm)
        print(f"s1_total_norm {idx} ", s1_total_norm)
        print(f"s2_total_norm {idx} ", s2_total_norm)

    if True:
        stacked = np.hstack(
            (image1_gray, image2_gray, original_image1_gray, original_image2_gray)
        )
        cv2.imwrite(
            f"/tmp/dim/stacked_{idx}_{round(s0_total_norm, 3)}_{round(s1_total_norm, 3)}_sim_{round(s2_total_norm, 3)}_s_{round(score, 3)}.png",
            stacked,
        )

    return score


def crop_to_content(frame: np.ndarray, content_aware=True) -> np.ndarray:
    """
    Crop given image to content
    No content is defined as first non background(white) pixel.

    @param frame: the image frame to process
    @param content_aware: if enabled we will apply more aggressive crop method
    @return: new cropped frame
    """

    start = time.time()
    # conversion required, or we will get 'Failure to use adaptiveThreshold: CV_8UC1 in function adaptiveThreshold'
    # frame = np.random.choice([0, 255], size=(32, 32), p=[0.01, 0.99]).astype("uint8")
    cv2.imwrite("/tmp/fragments/frame-src.png", frame)

    # Transform source image to gray if it is not already
    # check if the image is already in grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if content_aware:
        # apply division normalization to preprocess the image
        blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        # divide
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        op_frame = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        op_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    indices = np.array(np.where(op_frame == [0]))
    img_w = op_frame.shape[1]
    img_h = op_frame.shape[0]
    min_x_pad = 0  # 16  # img_w // 16
    min_y_pad = 0  # img_h // 4

    if len(indices[0]) == 0 or len(indices[1]) == 0:
        print("No content found")
        return frame

    # indices are in y,X format
    if content_aware:
        x = max(0, indices[1].min() - min_x_pad)
        y = 0  # indices[0].min()
        h = img_h  # indices[0].max() - y
        w = min(img_w, indices[1].max() - x + min_x_pad)
    else:
        x = indices[1].min()
        y = indices[0].min()
        h = indices[0].max() - y
        w = indices[1].max() - x

    cropped = frame[y : y + h + 1, x : x + w + 1].copy()
    # cv2.imwrite("/tmp/fragments/cropped.png", cropped)

    dt = time.time() - start
    return cropped


def resize_image(
    image, desired_size, color=(255, 255, 255), keep_max_size=False
) -> tuple:
    """Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    """

    # check if the image is already of the desired size
    if image.shape[0] == desired_size[0] and image.shape[1] == desired_size[1]:
        return image, (0, 0, image.shape[1], image.shape[0])

    size = image.shape[:2]
    if keep_max_size:
        # size = (max(size[0], desired_size[0]), max(size[1], desired_size[1]))
        h = size[0]
        w = size[1]
        dh = desired_size[0]
        dw = desired_size[1]

        if w > dw and h < dh:
            delta_h = max(0, desired_size[0] - size[0])
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left = 40
            right = 40
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
            )
            size = image.shape[:2]
            # cv2.imwrite("/tmp/marie/box_framed_keep_max_size.png", image)
            return image, (left, top, size[1], size[0])

    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(
            image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC
        )
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    # convert top, bottom, left, right to x, y, w, h
    x, y, w, h = left, top, size[1], size[0]

    # cv2.imwrite("/tmp/dim/box_framed.png", image)
    return image, (x, y, w, h)


def viz_patches(patches, filename):
    plt.figure(figsize=(9, 9))
    square_x = patches.shape[1]
    square_y = patches.shape[0]

    ix = 1
    for i in range(square_y):
        for j in range(square_x):
            # specify subplot and turn of axis
            ax = plt.subplot(square_y, square_x, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot
            plt.imshow(patches[i, j, :, :], cmap="gray")
            ix += 1
    # show the figure
    # plt.show()
    plt.savefig(filename)
    plt.close()
