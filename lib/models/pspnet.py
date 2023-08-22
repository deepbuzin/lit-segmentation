from collections import OrderedDict

import timm
import torch
from torch import nn
from torch.nn import functional as F


class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=not use_bathcnorm),
            nn.BatchNorm2d(out_channels) if use_bathcnorm else nn.Identity(),
            nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_bathcnorm=use_bathcnorm)
                for size in sizes])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(self, encoder_channels, use_batchnorm=True, out_channels=512, dropout=0.2):
        super().__init__()

        self.psp = PSPModule(in_channels=encoder_channels[-1],
                             sizes=(1, 2, 3, 6),
                             use_bathcnorm=use_batchnorm)
        self.conv = nn.Sequential(
            nn.Conv2d(encoder_channels[-1] * 2, out_channels, 1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class PSPHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, upsampling=1, activation=None) -> None:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
            activation if activation is not None else nn.Identity()
        ]
        super().__init__(*layers)


class PSPNet(nn.Module):
    def __init__(self,
                 encoder,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        encoder_channels = self.encoder.feature_info.channels()
        encoder_reduction = self.encoder.feature_info.reduction()

        self.decoder = PSPDecoder(
            encoder_channels=encoder_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout)

        self.head = PSPHead(in_channels=psp_out_channels,
                            out_channels=num_classes,
                            kernel_size=3,
                            upsampling=encoder_reduction[-1])
        # TODO add aux head

    def forward(self, x):
        result = OrderedDict()
        features = self.encoder(x)
        x = self.decoder(*features)
        result['out'] = self.head(x)
        # if self.aux_classifier:
        #     result['aux'] = self.aux_head(x)
        return result


def get_pspnet(backbone_name, num_classes, pretrained=True):
    backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
    return PSPNet(encoder=backbone, num_classes=num_classes)
