import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gausian import GaussianNoise


from .swish import Swish


class NLayerDiscriminatorWithSpectralNorm(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminatorWithSpectralNorm, self).__init__()

        print('NLayerDiscriminatorWithSpectralNorm')
        self.std = 0.1
        self.std_decay_rate = 0

        use_bias = True
        kw = 4
        padw = 1
        sequence = [
            #  GaussianNoise(self.std, self.std_decay_rate),
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias)), Swish()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # GaussianNoise(self.std, self.std_decay_rate),
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                # Swish()
                Swish()
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            # GaussianNoise(self.std, self.std_decay_rate),
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            # Swish()
            Swish()
        ]

        sequence += [
            # GaussianNoise(self.std, self.std_decay_rate),
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
