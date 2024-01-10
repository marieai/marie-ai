from __future__ import division, print_function

import copy

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from .DIM import (
    DIM_matching,
    conv2_same,
    ellipse,
    extract_additionaltemplates,
    imcrop_odd,
    preprocess,
)

matplotlib.use("Agg")


# https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/


def extract_hog_features(gray, channels, device) -> torch.Tensor:
    # Apply HOG to the input image
    from skimage.feature import hog

    fd, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2),
        visualize=True,
    )

    feature_arr = np.array(hog_image)
    return to_feature_map(feature_arr, channels, device)


def extract_lbp_features(gray, channels, device) -> torch.Tensor:
    from skimage.feature import local_binary_pattern

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp = lbp.astype("uint8")
    feature_arr = np.array(lbp)
    return to_feature_map(feature_arr, channels, device)


def to_feature_map(feature_arr, channels, device) -> torch.Tensor:
    """create new tensor of N x C x H x W"""
    features = torch.zeros(
        (1, channels, feature_arr.shape[0], feature_arr.shape[1])
    ).to(device)
    # copy the feature to all channels
    features[:, :, :, :] = torch.from_numpy(feature_arr).float().to(device)
    return features


class Featex:
    def __init__(self, model, use_cuda, layer1, layer2, layer3):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.feature3 = None
        self.U1 = None
        self.U2 = None
        self.U3 = None
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:36]
        # self.model = self.model[:19]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[layer1].register_forward_hook(self.save_feature1)
        self.model[layer1 + 1] = torch.nn.ReLU(inplace=False)
        self.model[layer2].register_forward_hook(self.save_feature2)
        self.model[layer2 + 1] = torch.nn.ReLU(inplace=False)
        self.model[layer3].register_forward_hook(self.save_feature3)
        self.model[layer3 + 1] = torch.nn.ReLU(inplace=False)

    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
        self.visualize_feature(self.feature1, "feature1")

    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        self.visualize_feature(self.feature2, "feature2")

    def save_feature3(self, module, input, output):
        self.feature3 = output.detach()
        self.visualize_feature(self.feature3, "feature3")

    def visualize_feature(self, feature, name):

        if True:
            return

        import matplotlib.pyplot as plt

        feature = feature.cpu().numpy()

        # feature = (feature - feature.min()) / (feature.max() - feature.min())
        # feature = (feature * 255).astype(np.uint8)

        layer_viz = feature[0, :, :, :]
        plt.figure(figsize=(30, 30))

        for i, filter in enumerate(layer_viz):
            if i == 16:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(4, 4, i + 1)
            plt.imshow(filter, cmap="jet")
            plt.axis("on")

        print(f"Saving layer {name} feature maps...")
        plt.savefig(f"/tmp/dim/layer_{name}.png")
        plt.close()

    def __call__(self, input, mode="normal"):
        channel = 64
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)

        if channel < self.feature1.shape[1]:
            reducefeature1, self.U1 = runpca(self.feature1, channel, self.U1)
        else:
            reducefeature1 = self.feature1
        if channel < self.feature2.shape[1]:
            reducefeature2, self.U2 = runpca(self.feature2, channel, self.U2)
        else:
            reducefeature2 = self.feature2
        if channel < self.feature3.shape[1]:
            reducefeature3, self.U3 = runpca(self.feature3, channel, self.U3)
        else:
            reducefeature3 = self.feature3

        h = self.feature1.size()[3]
        w = self.feature1.size()[2]

        if mode == "big":
            # resize feature1 to the same size of feature2
            w = self.feature3.size()[2]
            h = self.feature3.size()[3]

            reducefeature1 = F.interpolate(
                reducefeature1,
                size=(self.feature3.size()[2], self.feature3.size()[3]),
                mode="bilinear",
                align_corners=True,
            )
            reducefeature2 = F.interpolate(
                reducefeature2,
                size=(self.feature3.size()[2], self.feature3.size()[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            reducefeature2 = F.interpolate(
                reducefeature2,
                size=(self.feature1.size()[2], self.feature1.size()[3]),
                mode="bilinear",
                align_corners=True,
            )
            reducefeature3 = F.interpolate(
                reducefeature3,
                size=(self.feature1.size()[2], self.feature1.size()[3]),
                mode="bilinear",
                align_corners=True,
            )

        # Apply HOG and LBP  to the input image
        # convert from tensor to numpy
        src_input = input.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # blur the image
        # src_input = cv2.GaussianBlur(src_input, (5, 5), 3)
        gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (h, w))

        hog_feature = extract_hog_features(gray, channel, input.device)
        # lbp_feature = extract_lbp_features(gray, channel, input.device)

        # return torch.cat(
        #     (reducefeature1, reducefeature2, reducefeature3, hog_feature), dim=1
        # )
        return torch.cat((reducefeature1, reducefeature2, reducefeature3), dim=1)
        # return torch.cat((hog_feature, hog_feature), dim=1)


def runpca(x, components, U):
    whb = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    shape = whb.shape
    raw = whb.reshape((shape[0] * shape[1], shape[2]))
    X_norm, mu, sigma = featureNormalize(raw)
    if U is None:
        Sigma = np.dot(np.transpose(X_norm), X_norm) / raw.shape[0]
        U, S, V = np.linalg.svd(Sigma)
    Z = projectData(X_norm, U, components)
    val = (
        torch.tensor(Z.reshape((shape[0], shape[1], components)))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .cuda(),
        U,
    )
    return val


def featureNormalize(X):
    n = X.shape[1]

    sigma = np.zeros((1, n))
    mu = np.zeros((1, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(n):
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]
    return X, mu, sigma


def projectData(X_norm, U, K):
    Z = np.zeros((X_norm.shape[0], K))

    U_reduce = U[:, 0:K]
    Z = np.dot(X_norm, U_reduce)
    return Z


def apply_DIM(I_row, SI_row, template_bbox, pad, pad1, image, numaddtemplates):
    I = preprocess(I_row)
    SI = preprocess(SI_row)
    template, oddTbox = imcrop_odd(I, template_bbox, True)
    targetKeypoints = [
        oddTbox[1] + (oddTbox[3] - 1) / 2,
        oddTbox[0] + (oddTbox[2] - 1) / 2,
    ]
    addtemplates = extract_additionaltemplates(
        I, template, numaddtemplates, np.array([targetKeypoints])
    )
    if len(addtemplates):
        templates = torch.cat((template, addtemplates), 0)
    else:
        templates = template
    print("Numtemplates=", len(templates))
    print("Preprocess done,start matching...")
    similarity = DIM_matching(SI, templates, 6)[
        pad[0] : pad[0] + I.shape[2], pad[1] : pad[1] + I.shape[3]
    ]
    # post processing
    similarity = cv2.resize(similarity, (image.shape[1], image.shape[0]))
    scale = 0.025
    region = (
        torch.from_numpy(
            ellipse(round(max(1, scale * pad1[1])), round(max(1, scale * pad1[0])))
        )
        .type(torch.FloatTensor)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    similarity = (
        conv2_same(torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0), region)
        .squeeze()
        .numpy()
    )

    return similarity
