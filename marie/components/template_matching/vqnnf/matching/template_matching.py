import time

import colorcet as cc
import cv2
import numpy as np
import seaborn as sns
import torch
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from torch import nn

from .gauss_haar_filters import GaussHaarFilters

# from fast_pytorch_kmeans import KMeans
from .kmeans import KMeans


class VQNNFMatcher:
    def __init__(
        self,
        template: torch.Tensor,
        pca_dims: int = None,
        n_code: int = 128,
        filters_cat: str = "haar",
        filter_params: dict = None,
        verbose: bool = False,
        code_weights: np.array = None,
    ) -> None:
        self.device = template.device
        self.eps = 1e-6
        self.n_chunks = 8000  # can be increased if GPU memory allows

        c, self.t_w, self.t_h = template.shape
        template_flatten = template.reshape(c, self.t_w * self.t_h).transpose(1, 0)
        t1 = time.time()

        if pca_dims is not None:
            # print(f"Performing PCA with {pca_dims} dimensions")
            U, S, V = torch.pca_lowrank(template_flatten, q=pca_dims)
            self.v = V
            template_flatten = template_flatten @ V
        else:
            self.v = None

        self.n_code = (
            n_code if template_flatten.shape[0] > n_code else template_flatten.shape[0]
        )

        clusterer = KMeans(
            n_clusters=self.n_code, max_iter=25, device=torch.device(self.device)
        )
        template_nnf = clusterer.fit_predict(template_flatten)
        self.codebook = clusterer.centroids

        self.kmeans_time = time.time() - t1

        self.pool1d = nn.MaxPool1d(kernel_size=self.n_code, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=self.n_code)

        one_hot_nnf = (
            torch.nn.functional.one_hot(
                template_nnf.reshape(self.t_w, self.t_h), num_classes=self.n_code
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        ).float()
        cumsum_onehot = one_hot_nnf.cumsum(dim=2).cumsum(dim=3)

        self.verbose = verbose
        if self.verbose:
            self.colors = sns.color_palette(cc.glasbey, self.n_code)
            self.template_nnf = label2rgb(
                one_hot_nnf.argmax(dim=1).squeeze(0).cpu().numpy(), colors=self.colors
            )
        else:
            self.template_nnf = None
        # cv2.imwrite("/tmp/dim/template_nnf.png", self.template_nnf * 255)

        self.code_weights = (
            np.ones(self.n_code) / self.n_code if code_weights is None else code_weights
        )

        if filters_cat == "haar":
            filter_params.update({"n_channels": self.n_code})
            filter_params.update({"device": self.device})
            filter_params.update({"template_shape": (self.t_w, self.t_h)})
            filter_params.update({"channel_weights": self.code_weights})
            self.filtering_layer = GaussHaarFilters(**filter_params)

        self.filtering_layer.get_template_features(cumsum_onehot)

    def get_nnf(self, x: torch.Tensor):
        d, w, h = x.shape
        x = x.reshape(d, w * h).transpose(1, 0)
        chunks = torch.split(x, self.n_chunks, dim=0)

        nnf_idxs = []
        sim_vals = []
        one_hots = []
        sim_one_hots = []
        for i, chunk in enumerate(chunks):
            dist = -torch.cdist(self.codebook, chunk)
            sim, idx = self.pool1d(dist.transpose(0, 1))
            one_hot = torch.ones_like(sim)
            one_hots.append(self.unpool1d(one_hot, idx))
            sim_one_hots.append(self.unpool1d(sim, idx))
            sim_vals.append(sim)
            nnf_idxs.append(idx)

        nnf_idxs = torch.cat(nnf_idxs).transpose(0, 1).reshape(w, h)
        sim_vals = -torch.cat(sim_vals).transpose(0, 1).reshape(1, w, h)
        sim_one_hots = -torch.cat(sim_one_hots).transpose(0, 1).reshape(-1, w, h)
        one_hots = torch.cat(one_hots).transpose(0, 1).reshape(-1, w, h)

        if self.verbose:
            nnf = one_hots.argmax(dim=0)
            nnf = label2rgb(nnf.cpu().numpy(), colors=self.colors)

        return one_hots, nnf_idxs, sim_vals

    def get_heatmap(self, x: torch.Tensor):
        if self.v is not None:
            x = (x.permute(1, 2, 0) @ self.v).permute(2, 0, 1)

        with torch.no_grad():
            one_hots, query_nnf, _ = self.get_nnf(x)
            integral_onehot = one_hots.cumsum(dim=1).cumsum(dim=2)
            distrib_sim, all_filt_sim = self.filtering_layer.get_query_map(
                integral_onehot.unsqueeze(0)
            )

        query_XXXX = None
        self.verbose = False
        if self.verbose:
            t = query_nnf
            query_nnf = query_nnf.cpu().numpy()

            all_filt_sim = [
                rescale_intensity(filt_sim, out_range=(0, 1))
                for filt_sim in all_filt_sim
            ]

            # GB MOD
            self.colors = sns.color_palette(cc.glasbey, self.n_code)
            one_hot_nnf = (
                torch.nn.functional.one_hot(
                    t.reshape(t.shape[0], t.shape[1]),
                    num_classes=self.n_code,
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            ).float()

            query_XXXX = label2rgb(
                one_hot_nnf.argmax(dim=1).squeeze(0).cpu().numpy(), colors=self.colors
            )

        return (
            distrib_sim.cpu().numpy(),
            all_filt_sim,
            self.template_nnf,
            query_XXXX,
        )

    # return (distrib_sim.cpu().numpy(), all_filt_sim, self.template_nnf, query_nnf)
