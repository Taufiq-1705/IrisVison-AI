"""Deep learning model architectures for Iris Vision.

Supports pretrained MobileNetV2 (`IMAGENET1K_V2`) and EfficientNet-B0 (`IMAGENET1K_V1`)
paired with a custom two-stage regularized classification head.
"""

from typing import Iterator
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
)

from src.config import BACKBONE_NAME, NUM_CLASSES


class EyeDiseaseClassifier(nn.Module):
    """Multi-class (2, 4, or 5-class) ocular classifier with transferable backbone and custom head."""

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower().strip()
        self.num_classes = num_classes
        self._backbone_frozen = False

        if self.backbone_name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            self.features = base_model.features
            in_features = base_model.classifier[1].in_features
        elif self.backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
            self.features = base_model.features
            in_features = base_model.classifier[0].in_features
        elif self.backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            self.features = base_model.features
            in_features = base_model.classifier[1].in_features
        elif self.backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.resnet50(weights=weights)
            # Wrap stem + residual stages so freeze/unfreeze & Grad-CAM (features[-1] = layer4)
            # behave identically across all backbones.
            self.features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
            )
            in_features = base_model.fc.in_features  # 2048
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "Supported backbones: 'mobilenet_v2', 'mobilenet_v3_small', 'efficientnet_b0', 'resnet50'."
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone feature extractor for Phase 1 head warm-up."""
        self._backbone_frozen = True
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze full network for Phase 2 end-to-end fine-tuning."""
        self._backbone_frozen = False
        for param in self.features.parameters():
            param.requires_grad = True
        self.features.train()

    def train(self, mode: bool = True) -> "EyeDiseaseClassifier":
        """Keep frozen backbone in eval mode during Phase 1 warm-up."""
        super().train(mode)
        if mode and self._backbone_frozen:
            self.features.eval()
        return self

    def get_backbone_params(self) -> Iterator[nn.Parameter]:
        """Return parameter iterator for the backbone feature extractor."""
        return self.features.parameters()

    def get_head_params(self) -> Iterator[nn.Parameter]:
        """Return parameter iterator for the classification head."""
        return self.head.parameters()

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return final convolutional block for Grad-CAM visualization."""
        return self.features[-1]


def build_model(
    backbone_name: str = BACKBONE_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> EyeDiseaseClassifier:
    """Factory function to construct EyeDiseaseClassifier."""
    return EyeDiseaseClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )

