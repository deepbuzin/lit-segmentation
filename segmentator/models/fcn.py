from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import timm


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
    def __init__(self, encoder: nn.Module, num_classes, aux_classifier: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = FCNHead(encoder.feature_info.channels()[-1], num_classes)
        self.aux_classifier = aux_classifier
        self.aux_head = FCNHead(encoder.feature_info.channels()[-2], num_classes) if aux_classifier else None

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        result = OrderedDict()
        x = self.head(features[-1])
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        if self.aux_classifier:
            aux = self.aux_head(features[-2])
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux
        return result


def get_fcn(backbone_name="resnet34", num_classes=1, pretrained=True, aux_classifier=False):
    backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
    return FCN(backbone, num_classes, aux_classifier=aux_classifier)





