from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

import timm


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample_rate, use_batchnorm=True):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.upsample_rate, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, encoder_reduction, decoder_channels, use_batchnorm=True):
        super().__init__()
        assert len(encoder_channels) == len(encoder_reduction) == len(decoder_channels), \
            f"Encoder-decoder channel/reduction rate mismatch"

        encoder_channels = encoder_channels[::-1]
        encoder_reduction = encoder_reduction[::-1] + [1]
        upsample_rate = [i // j for i, j in zip(encoder_reduction[:-1], encoder_reduction[1:])]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        blocks = [
            UNetDecoderBlock(in_ch, skip_ch, out_ch, up_rate, use_batchnorm=use_batchnorm)
            for in_ch, skip_ch, out_ch, up_rate in zip(in_channels, skip_channels, out_channels, upsample_rate)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]  # reverse channels to start from head of encoder
        x = features[0]
        skips = features[1:]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class UNetHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, activation=None) -> None:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            activation if activation is not None else nn.Identity()
        ]
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = None,
                 num_classes=1):
        super().__init__()
        self.encoder = encoder
        encoder_channels = self.encoder.feature_info.channels()
        encoder_reduction = self.encoder.feature_info.reduction()
        decoder_channels = [2 ** i for i in reversed(range(6, 6 + len(encoder_channels)))] \
            if decoder_channels is None else decoder_channels

        self.decoder = UNetDecoder(encoder_channels=encoder_channels,
                                   encoder_reduction=encoder_reduction,
                                   decoder_channels=decoder_channels,
                                   use_batchnorm=decoder_use_batchnorm)

        self.head = UNetHead(decoder_channels[-1], num_classes, kernel_size=3)
        # TODO: add aux head

    def forward(self, x):
        result = OrderedDict()
        features = self.encoder(x)
        x = self.decoder(*features)
        result['out'] = self.head(x)
        # if self.aux_classifier:
        #     result['aux'] = self.aux_head(x)
        return result


def get_unet(backbone_name, num_classes, pretrained=True):
    backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
    return UNet(encoder=backbone, num_classes=num_classes)
