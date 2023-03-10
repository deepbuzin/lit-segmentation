from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import timm


class FCNAuxWrapper(nn.Module):
    """
    Wraps a model to add an auxiliary classifier for deep supervision.
    TODO Will only work if the model has resnet-like structure.
    """
    def __init__(self, model: nn.Module, aux_classifier: bool = False):
        super().__init__()
        self.model = model
        self.aux_classifier = aux_classifier

        module_names = [name for name, _ in list(self.model.named_children())]
        self.out_layers = {module_names[-1]: 'out', module_names[-2]: 'aux'}

        x = torch.randn(1, 3, 224, 224)
        out_channels = []
        for name, module in self.model.named_children():
            x = module(x)
            if name in self.out_layers:
                out_channels.append(x.shape[1])
        self.aux_channels, self.out_channels = out_channels

    def forward(self, x):
        result = OrderedDict()
        for name, module in self.model.named_children():
            x = module(x)
            if name in self.out_layers:
                result[self.out_layers[name]] = x
        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class FCN(nn.Module):
    def __init__(self, backbone: FCNAuxWrapper, num_classes, aux_classifier: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = FCNHead(backbone.out_channels, num_classes)
        self.aux_classifier = aux_classifier
        self.aux_head = FCNHead(backbone.aux_channels, num_classes) if aux_classifier else None

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        result = OrderedDict()
        x = self.head(features["out"])
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        if self.aux_classifier:
            aux = self.aux_head(features["aux"])
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux
        return result


def get_fcn(backbone_name="resnet34", num_classes=1, pretrained=True, aux_classifier=True):
    backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
    backbone = FCNAuxWrapper(backbone, aux_classifier=aux_classifier)
    return FCN(backbone, num_classes, aux_classifier=aux_classifier)





