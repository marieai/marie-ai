import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

from .gausian import GaussianNoise
from .spectral_discriminator import NLayerDiscriminatorWithSpectralNorm


class Swish(nn.Module):
    """
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3),
                                nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0)),
                                norm_layer(ngf_global), Swish(),
                                nn.utils.spectral_norm(
                                    nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1)),
                                norm_layer(ngf_global * 2), Swish()]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample

            if False:
                model_upsample += [nn.Upsample(scale_factor=2, mode="nearest"),
                                   nn.utils.spectral_norm(
                                       nn.Conv2d(in_channels=ngf_global, out_channels=ngf_global * 2, kernel_size=3,
                                                 stride=2, padding=1)),
                                   norm_layer(ngf_global),
                                   Swish()]

            print(ngf_global)
            # https://www.programcreek.com/python/?code=alterzero%2FSTARnet%2FSTARnet-master%2Fbase_networks.py
            if True:
                model_upsample += [
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)),
                    norm_layer(ngf_global), Swish()]

                # Pix2PixHD: use instance norm for the last conv layer

            # upsample using PixelShuffle
            # model_upsample += [
            #         nn.utils.spectral_norm(nn.Conv2d(in_channels=ngf_global * 2, out_channels=ngf_global, kernel_size=3, stride=2, padding=1)),
            #     ]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3),
                                   nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

            # Replacing AvgPool2d with Conv2d aka Strided Convolution
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.downsample = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = Swish()
        # activation = nn.LeakyReLU(0.2, True)

        self.std = 0.1
        self.std_decay_rate = 0

        model = [
            # GaussianNoise(self.std, self.std_decay_rate),
            nn.ReflectionPad2d(3), nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)),
            norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                #   GaussianNoise(self.std, self.std_decay_rate),
                nn.utils.spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)),
                norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)

            model += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.utils.spectral_norm(
                    nn.Conv2d(in_channels=ngf * mult, out_channels=int(ngf * mult / 2), kernel_size=3, stride=1,
                              padding=1)),
                activation]

            if False:
                model += [nn.utils.spectral_norm(
                    nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                       output_padding=1)),
                          norm_layer(int(ngf * mult / 2)), activation]

        model += [
            nn.ReflectionPad2d(3), nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)),
            nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

    # Define a resnet block


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=Swish(), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):

        self.std = 0.1
        self.std_decay_rate = 0

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # nn.utils.spectral_norm
        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p)),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # nn.utils.spectral_norm
        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p)),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            # netD = NLayerDiscriminatorWithSpectralNorm(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)

            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # self.downsample = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input

        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                # print(f"downsampling... B: {input_downsampled.shape}")
                input_downsampled = self.downsample(input_downsampled)
                # print(f"downsampling... A: {input_downsampled.shape}")
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        print('NLayerDiscriminator')
        self.std = 0.1
        self.std_decay_rate = 0

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        # sequence = [[
        #     nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
        #     ]]

        sequence = [[
            # GaussianNoise(self.std, self.std_decay_rate),
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), Swish()
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                # GaussianNoise(self.std, self.std_decay_rate),
                nn.utils.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(nf), Swish()
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            # GaussianNoise(self.std, self.std_decay_rate),
            nn.utils.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(nf),
            Swish()
        ]]

        sequence += [[
            nn.utils.spectral_norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))
        ]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
