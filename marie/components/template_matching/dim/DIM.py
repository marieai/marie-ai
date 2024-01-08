import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1


def imcrop_odd(I, box, indxtrans=False):
    b, c, w, h = I.shape
    if indxtrans == True:
        box = (round(box[0]) - 1, round(box[1]) - 1, odd(box[2]), odd(box[3]))
    else:
        box = (round(box[0]), round(box[1]), odd(box[2]), odd(box[3]))
    boxInbounds = (
        min(h - odd(box[2]), max(0, round(box[0]))),
        min(w - odd(box[3]), max(0, round(box[1]))),
        odd(box[2]),
        odd(box[3]),
    )
    box = boxInbounds
    Ipatch = I[:, :, box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
    return Ipatch, box


def ellipse(alen, blen):
    x, y = np.mgrid[-alen : alen + 1, -blen : blen + 1]
    s = np.zeros_like(x)
    tem = (x**2 / alen**2) + (y**2 / blen**2)
    s[tem <= 1] = 1
    return s


def extract_additionaltemplates(image, template, numadditionaltemplates, keypoints):
    template = torch.flip(template, [2, 3])
    similarity = conv2_same(image, template).squeeze().cpu().numpy()

    n, c, x, y = image.shape
    n, c, h, w = template.shape

    ind = np.argsort(similarity, axis=None)[::-1]
    keypointsCandidates_row = np.array(
        (np.unravel_index(ind, similarity.shape))
    ).transpose(1, 0)
    df = pd.DataFrame(keypointsCandidates_row, columns=list('AB'))
    keypointsCandidates = df[
        (df['A'] + 1 > np.ceil(h / 2))
        & (df['A'] < x - np.ceil(h / 2))
        & (df['B'] + 1 > np.ceil(w / 2))
        & (df['B'] < y - np.ceil(w / 2))
    ].values.astype('float32')
    keypointCandidatesAccepted = []
    addtemplates = []
    numAccepted = 0
    for i in range(len(keypointsCandidates)):
        if numadditionaltemplates == numAccepted:
            break
        if any(
            np.logical_and(
                (abs(keypointsCandidates[i][0] - keypoints[:, 0]) < h),
                (abs(keypointsCandidates[i][1] - keypoints[:, 1]) < w),
            )
        ):
            skip = 1
        else:
            keypointCandidatesAccepted.append(keypointsCandidates[i, :])
            keypoints = np.vstack((keypoints, keypointsCandidates[i, :]))
            numAccepted += 1
            addtemplate, box = imcrop_odd(
                image,
                (
                    int(keypointsCandidates[i, 1] - (w - 1) / 2),
                    int(keypointsCandidates[i, 0] - (h - 1) / 2),
                    w,
                    h,
                ),
            )
            addtemplates.append(addtemplate)
    if len(addtemplates):
        addtemplates = torch.cat(addtemplates, 0)
    return addtemplates


def preprocess(image):
    '''The preprocess described in original DIM paper'''
    cuda = torch.cuda.current_device()
    n, c, h, w = image.size()
    X = torch.zeros(n, 2 * c, h, w, device=cuda)
    for i in range(len(image)):
        for j in range(len(image[i])):
            X[i][2 * (j - 1) + 2] = torch.clamp(image[i][j], min=0)
            X[i][2 * (j - 1) + 3] = torch.clamp(image[i][j], max=0).abs()
    '''    
    Alternatively, you can use the fellowing code which run faster  
    imageon=torch.clamp(image,min=0)
    imageoff=torch.clamp(image,max=0).abs()  
    out=torch.cat((imageon,imageoff),1)
    return out
    '''
    return X


def conv2_same(Input, weight, num=1):
    padding_rows = weight.size(2) - 1
    padding_cols = weight.size(3) - 1
    rows_odd = padding_rows % 2 != 0
    cols_odd = padding_cols % 2 != 0
    if rows_odd or cols_odd:
        Input = F.pad(Input, [0, int(cols_odd), 0, int(rows_odd)])
    weight = torch.flip(weight, [2, 3])
    return F.conv2d(
        Input, weight, padding=(padding_rows // 2, padding_cols // 2), groups=num
    )


def DIM_matching(X, w, iterations):
    cuda = torch.cuda.current_device()
    v = torch.zeros_like(w)
    Y = torch.zeros(X.shape[0], len(w), X.shape[2], X.shape[3], device=cuda)
    tem1 = w.clone()

    for i in range(len(w)):
        v[i] = torch.max(
            torch.tensor(0, dtype=torch.float32, device=cuda),
            w[i]
            / torch.max(
                torch.tensor(1e-6, dtype=torch.float32, device=cuda), torch.max(w[i])
            ),
        )
        tem1[i] = w[i] / torch.max(
            torch.tensor(1e-6, dtype=torch.float32, device=cuda), torch.sum(w[i])
        )
    w = torch.flip(tem1, [2, 3])
    sumV = torch.sum(torch.sum(torch.sum(v, 0), 1), 1)
    epsilon2 = 1e-2
    epsilon1 = torch.tensor(epsilon2, dtype=torch.float32, device=cuda) / torch.max(
        sumV
    )
    for count in range(iterations):
        R = torch.zeros_like(X)
        if not torch.sum(Y) == 0:
            R = conv2_same(Y, v.permute(1, 0, 2, 3))
            R = torch.clamp(R, min=0)
        E = X / torch.max(torch.tensor(epsilon2, dtype=torch.float32, device=cuda), R)
        Input = torch.zeros_like(E)
        Input = conv2_same(E, w)
        tem2 = Y.clone()
        for i in range(len(Input)):
            for j in range(len(Input[i])):
                tem2[i][j] = Input[i][j] * torch.max(epsilon1, Y[i][j])
        Y = torch.clamp(tem2, min=0)
    Y = Y[:, 0, :, :].squeeze(0).cpu().numpy()
    return Y
