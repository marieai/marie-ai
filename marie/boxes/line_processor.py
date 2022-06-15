from __future__ import print_function

import copy
import os
from time import time
from typing import List, Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

from marie.utils.overlap import find_overlap


def find_line_index(lines, box):
    """Get line index for specific box"""

    line_number = -1
    # TODO : Figure out better way to handle vertical lines
    # prune vertical lines
    x, y, w, h = box
    rat = h / w
    _, line_indexes = find_overlap(box, lines)
    if len(line_indexes) == 1:
        line_number = line_indexes[0] + 1

    print(f"{box}  :: {rat}  : {line_number}")
    if line_number == -1:
        msg = f"Invalid line number : -1, this looks like a bug/vertical line : {box}"
        print(msg)
        # raise Exception(msg)

    return line_number


def line_refinerXXXX(image, bboxes, _id, lines_dir) -> List[Any]:
    """Line refiner creates lines out of set of bounding box regions
    Ref : https://www.answerminer.com/blog/binning-guide-ideal-histogram
    """
    img_h = image.shape[0]
    img_w = image.shape[1]
    all_box_lines = []

    _bboxes = np.array(bboxes)
    h1 = _bboxes[:, 3]
    idxs = np.argsort(h1)
    h1 = h1[idxs]
    print(idxs)
    print(h1)

    h1 = runningMeanFast(h1, 5)
    print(h1)

    # for h in h1:

    hmin = np.min(h1)
    hmax = np.max(h1)
    hmean = np.mean(h1)
    q75, q25 = np.percentile(h1, [75, 25])
    iqr = q75 - q25

    print(h1)
    std = int(np.std(h1))
    n = len(h1)
    print(f" std = {std}, hmin = {hmin}, hmax ={hmax}, hmean = {hmean}, iqr = {iqr}, q25 = {q25}, q75 = {q75}")

    m1 = np.sqrt(n)
    m2 = np.log2(n) + 1
    m3 = 2 * np.cbrt(n)
    m4 = (hmax - hmin) / (3.5 * (std / np.cbrt(n)))
    m5 = (hmax - hmin) / (2 * (iqr / np.cbrt(n)))

    print(f" std = {std}")
    print(f" m1 = {m1}")
    print(f" m2 = {m2}")
    print(f" m3 = {m3}")
    print(f" m4 = {m4}")
    print(f" m5 = {m5}")

    bins = [int(hmean)]
    start = hmean
    while True:
        bin_end = int(start + std)
        start = bin_end
        bins.append(bin_end)
        if bin_end > hmax:
            break

    print(bins)


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode="valid")[(N - 1) :]


def line_refiner(image, bboxes, _id, lines_dir) -> List[Any]:
    """Line refiner creates lines out of set of bounding box regions
    Ref : https://www.answerminer.com/blog/binning-guide-ideal-histogram
    """

    from sklearn import metrics
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import silhouette_score
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    img_h = image.shape[0]
    img_w = image.shape[1]
    all_box_lines = []

    print(bboxes)
    _bboxes = np.array(bboxes)
    h1 = _bboxes[:, 3]
    idxs = np.argsort(h1)
    h1 = h1[idxs]

    print(idxs)
    print(h1)
    name = "kmeans"
    data = h1.reshape(-1, 1)
    print(data)

    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    Sum_of_squared_distances = []
    K = range(1, 12)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)

    print(data_transformed)
    print("Sum_of_squared_distances")
    print(Sum_of_squared_distances)

    plt.plot(K, Sum_of_squared_distances, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal k")
    plt.show()

    for n_cluster in range(2, 16):
        # kmeans = KMeans(n_clusters=n_cluster).fit(data_transformed)
        kmeans = KMeans(init="k-means++", n_clusters=n_cluster, random_state=0)
        kmeans = kmeans.fit(data_transformed)
        label = kmeans.labels_
        sil_coeff = silhouette_score(data_transformed, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

    best_k = 1
    best_d = 0
    for i in range(1, len(Sum_of_squared_distances) - 1):
        a = Sum_of_squared_distances[i]
        b = Sum_of_squared_distances[i + 1]
        d = a - b
        if d > best_d:
            best_d = d
            best_k = i + 2

    best_k = 3
    print(f"Best K = {best_k}")
    # Fit K-means with Scikit
    kmeans = KMeans(init="k-means++", n_clusters=best_k, n_init=10, random_state=0)
    kmeans.fit(data)

    # Predict the cluster for all the samples
    P = kmeans.predict(data)
    print("Predictions")
    print(P)

    return []

    running_mean = runningMeanFast(h1, 4)
    print(running_mean)

    std = int(np.std(h1))
    print(f" std = {std}")

    group = 0
    dy = 0
    groups = np.zeros(len(h1)).astype(int)
    group_map = {}

    for i in range(0, len(h1)):

        if i == 0:
            a = h1[i]
            b = h1[i]
        elif i == len(h1) - 1:
            a = h1[i]
            b = h1[i - 1]
        else:
            a = h1[i - 1]
            b = h1[i]

        d = abs(b - a)
        dy += d
        if d > (std):
            group += 1
            break

        if group not in group_map:
            group_map[group] = []

        group_map[group].append(i)
        groups[i] = group

        print(f"{i} [{h1[i]}] > a = {a}, b = {b}, d = {d}, dy= {dy}, std = {std}  group = {group}")

    print(groups)
    print(group_map)

    for k in group_map.keys():
        print(f"gid = {k}")
        idx = group_map[k]
        group_bboxes = _bboxes[idx]

        if True:
            from PIL import Image, ImageDraw

            img_line = copy.deepcopy(image)
            viz_img = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)
            viz_img = Image.fromarray(viz_img)
            draw = ImageDraw.Draw(viz_img, "RGBA")

            for box in group_bboxes:
                x, y, w, h = box
                color = list(np.random.random(size=3) * 256)
                cv2.rectangle(img_line, (x, y), (x + w, y + h), color, 2)
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

            viz_img.save(os.path.join(lines_dir, "%s-group.png" % k), format="PNG", subsampling=0, quality=100)

    return []

    # bins = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 1.0])
    # def get_bin(value: float) -> int:
    #     return np.digitize([value], np.sort(bins))[0]

    for idx, box in enumerate(bboxes):
        x, y, w, h = box
        box_line = [0, y, img_w, h]
        print(h)
        # box_line = [x, y, w, h]
        box_line = np.array(box_line).astype(np.int32)
        all_box_lines.append(box_line)
        # print(f' >  {idx} : {box} : {box_line}')
    # print(f'all_box_lines : {len(all_box_lines)}')

    all_box_lines = np.array(all_box_lines)
    if len(all_box_lines) == 0:
        return []

    y1 = all_box_lines[:, 1]
    # sort boxes by the  y-coordinate of the bounding box
    idxs = np.argsort(y1)
    lines = []
    iter_idx = 0
    size = len(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        idx = idxs[last]
        box_line = all_box_lines[idx]
        overlaps, indexes = find_overlap(box_line, all_box_lines)
        overlaps = np.array(overlaps)
        avg_h = int(np.average(overlaps[:, 3]))
        avg_y = int(np.average(overlaps[:, 1]))

        print(f"{last}, {idx}, {indexes} : {box_line}  ->  {overlaps}")

        min_x = overlaps[:, 0].min()
        min_y = overlaps[:, 1].min()
        max_w = overlaps[:, 2].max()
        max_h = overlaps[:, 3].max()
        max_y = 0

        for overlap in overlaps:
            x, y, w, h = overlap
            dh = y + h
            if dh > max_y:
                max_y = dh

        max_h = max_y - min_y
        max_h = avg_h
        min_y = avg_y

        box = [min_x, min_y, max_w, max_h]
        lines.append(box)

        # there is a bug when there is a box index greater than candidate index
        # last/idx : 8   ,  2  >  [0 1 4 3 6 5 7 8 2] len = 9  /  [0 1 2 3 4 5 6 7 8 9] len = 10
        # Ex : 'index 9 is out of bounds for axis 0 with size 9'
        #  numpy.delete(arr, obj, axis=None)[source]¶
        indexes = indexes[indexes < idxs.size]
        print(indexes)
        idxs = np.delete(idxs, indexes, axis=0)
        iter_idx = iter_idx + 1
        # prevent inf loop
        if iter_idx > size:
            print("ERROR:Infinite loop detected")
            raise Exception("ERROR:Infinite loop detected")

    # reverse to get the right order
    lines = np.array(lines)[::-1]

    if True:
        from PIL import Image, ImageDraw

        img_line = copy.deepcopy(image)
        viz_img = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)
        viz_img = Image.fromarray(viz_img)
        draw = ImageDraw.Draw(viz_img, "RGBA")

        for line in lines:
            x, y, w, h = line
            color = list(np.random.random(size=3) * 256)
            cv2.rectangle(img_line, (x, y), (x + w, y + h), color, 2)
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

        cv2.imwrite(os.path.join(lines_dir, "%s-lineXX.png" % _id), img_line)
        viz_img.save(os.path.join(lines_dir, "%s-linePIL.png" % _id), format="PNG", subsampling=0, quality=100)

    # refine lines as there could be lines that overlap
    print(f"***** Line candidates size {len(lines)}")

    # sort boxes by the y-coordinate of the bounding box
    y1 = lines[:, 1]
    idxs = np.argsort(y1)
    refine_lines = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        idx = idxs[last]

        box_line = lines[idx]
        overlaps, indexes = find_overlap(box_line, lines)
        overlaps = np.array(overlaps)

        min_x = overlaps[:, 0].min()
        min_y = overlaps[:, 1].min()
        max_w = overlaps[:, 2].max()
        max_h = overlaps[:, 3].max()

        box = [min_x, min_y, max_w, max_h]
        refine_lines.append(box)

        # there is a bug when there is a box index greater than candidate index
        # last/idx : 8   ,  2  >  [0 1 4 3 6 5 7 8 2] len = 9  /  [0 1 2 3 4 5 6 7 8 9] len = 10
        # Ex : 'index 9 is out of bounds for axis 0 with size 9'
        #  numpy.delete(arr, obj, axis=None)[source]¶
        indexes = indexes[indexes < idxs.size]
        idxs = np.delete(idxs, indexes, axis=0)

    print(f"Final line size : {len(refine_lines)}")
    lines = np.array(refine_lines)[::-1]  # Reverse

    if True:
        img_line = copy.deepcopy(image)

        for line in lines:
            x, y, w, h = line
            color = list(np.random.random(size=3) * 256)
            cv2.rectangle(img_line, (x, y), (x + w, y + h), color, 1)

        for idx, box in enumerate(bboxes):
            color = (255, 0, 0)
            cv2.rectangle(img_line, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 1)

        cv2.imwrite(os.path.join(lines_dir, "%s-line.png" % _id), img_line)

    line_size = len(lines)
    print(f"Estimated line count : {line_size}")

    return lines
