import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample_(x, h, w):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, [h, w], mode="nearest")
#    return F.interpolate(x, scale_factor=2, mode="nearest")

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Upsampler_subp(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias, stride=1))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias, stride=1))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResNetEncoder(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_channels=4, net_type='CNN_TOOL3'):
        super(ResNetEncoder, self).__init__(block, layers)
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
                    nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.flag_deep = False
        if net_type == 'CNN_TOOL8' :
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.flag_deep = True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x) : 
        # feature encoding
        x1 = self.head(x)
        self.features = []
        self.features.append(x1)
        self.features.append(self.layer1(self.features[-1]))
        self.features.append(self.layer2(self.features[-1]))
        if self.flag_deep : 
            self.features.append(self.layer3(self.features[-1]))
            self.features.append(self.layer4(self.features[-1]))
        return self.features

class ResNetDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec, num_output_channels=1, use_skips=True):
        super(ResNetDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.num_block = len(self.num_ch_dec)-1

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_block, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_block else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upsampling
            self.convs[("upsample", i, 0)] = nn.ConvTranspose2d(num_ch_out, num_ch_out, 2, stride=2)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips:
                num_ch_in += self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.tail = nn.Sequential(
                    Conv3x3(self.num_ch_dec[0], self.num_output_channels),
                    nn.Sigmoid()
                    )

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        # decoder
        x = input_features[-1]
        for i in range(self.num_block, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [self.convs[("upsample", i, 0)](x)]
            if self.use_skips:
                x += [input_features[i]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        outputs = self.tail(x)
        return outputs

class CNN_TOOL_V2(nn.Module):
    def __init__(self, converter=None, num_channels=4, use_skips=True, network='CNN_TOOL3'):
        super(CNN_TOOL_V2, self).__init__()

        self.color_cvt, self.color_dcvt = self.set_colorspace(converter)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512]) # head, l1, l2, l3, l4
        self.num_ch_dec = np.array([32, 64, 128, 256]) # 

        if network == 'CNN_TOOL3' : 
            n_resblocks=2
        elif network == 'CNN_TOOL8' : 
            n_resblocks=4

        self.encoder = ResNetEncoder(models.resnet.BasicBlock, [1, 1, 1, 1], num_channels=num_channels, net_type=network)
        self.decoder = ResNetDecoder(self.num_ch_enc[:n_resblocks+1], self.num_ch_dec[:n_resblocks], num_output_channels=num_channels, use_skips=use_skips)

    def set_colorspace(self, converter) :
        if converter == 'hsv' : 
            color_cvt  = cc.rgb_to_hsv
            color_dcvt = cc.hsv_to_rgb
        elif converter == 'hls' : 
            color_cvt  = cc.rgb_to_hls
            color_dcvt = cc.hls_to_rgb
        elif converter == 'luv' : 
            color_cvt  = cc.rgb_to_luv
            color_dcvt = cc.luv_to_rgb
        elif converter == 'xyz' : 
            color_cvt  = cc.rgb_to_xyz
            color_dcvt = cc.xyz_to_rgb
        elif converter == 'ycbcr' : 
            color_cvt  = cc.rgb_to_ycbcr
            color_dcvt = cc.ycbcr_to_rgb
        elif converter == 'yuv' : 
            color_cvt  = cc.rgb_to_yuv
            color_dcvt = cc.yuv_to_rgb
        elif converter == 'lab' : 
            color_cvt  = cc.rgb_to_lab
            color_dcvt = cc.lab_to_rgb
        else: 
            color_cvt  = self.rgb_to_rgb
            color_dcvt = self.rgb_to_rgb
        return color_cvt, color_dcvt

    def rgb_to_rgb(self, x) :
        return x

    def forward(self, x):
        x = self.color_cvt(x)
        features = self.encoder(x)
        outputs = self.decoder(features)
        x_out = self.color_dcvt(outputs)
        return x_out