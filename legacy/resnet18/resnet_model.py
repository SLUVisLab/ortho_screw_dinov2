"""Dual-view ResNet18 classifier (the model deployed to the HF Space until November 2026).

Two separate ImageNet-pretrained ResNet18 backbones, one per view, each producing a
512-dim feature; the concatenated 1024-dim vector feeds a small fusion head.
"""
import torch
import torch.nn as nn
import torchvision


class DualViewClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.backbone_ap = torchvision.models.resnet18(weights=weights)
        self.backbone_lat = torchvision.models.resnet18(weights=weights)
        num_features = self.backbone_ap.fc.in_features  # 512
        self.backbone_ap.fc = nn.Identity()
        self.backbone_lat.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_ap, x_lat):
        combined = torch.cat((self.backbone_ap(x_ap), self.backbone_lat(x_lat)), dim=1)
        return self.classifier(combined)
