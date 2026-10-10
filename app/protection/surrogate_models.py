"""
PyTorch Face Recognition Surrogates for Strong Mode PGD
======================================================
Provides gradient-enabled white-box face recognition backbones:
  1. ArcFace: IResNet50 backbone (MS1MV3 weights from ~/.cache/varman/arcface_r50_backbone.pth)
  2. FaceNet: InceptionResnetV1 (VGGFace2 weights via facenet_pytorch)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class _IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class _IResNet(nn.Module):
    def __init__(self, block: type, layers: list[int], dropout: float = 0.0, num_features: int = 512):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


class DualSurrogateSuite(nn.Module):
    """
    Combined ArcFace and FaceNet surrogate for gradient-based adversarial disruption.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # 1. ArcFace (IResNet50)
        self.arcface = _IResNet(_IBasicBlock, [3, 4, 14, 3]).eval().to(device)
        weights_path = Path.home() / ".cache" / "varman" / "arcface_r50_backbone.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"ArcFace backbone weights missing at {weights_path}")
        
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v.float() for k, v in state.items()}
        self.arcface.load_state_dict(state, strict=False)
        for p in self.arcface.parameters():
            p.requires_grad = False

        # 2. FaceNet (InceptionResnetV1)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.facenet.parameters():
            p.requires_grad = False

    def forward(self, x_rgb_112: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts L2-normalized embeddings for both models from aligned 112x112 RGB tensors in [0, 1].

        Returns:
            (arcface_emb_512, facenet_emb_512)
        """
        # ArcFace input: (pixel * 255 - 127.5) / 128.0
        x_af = (x_rgb_112 * 255.0 - 127.5) / 128.0
        emb_af = self.arcface(x_af)
        emb_af = nn.functional.normalize(emb_af, p=2, dim=1)

        # FaceNet input: 160x160 bilinear, (pixel * 255 - 127.5) / 128.0
        x_fn = nn.functional.interpolate(x_rgb_112, size=(160, 160), mode="bilinear", align_corners=False)
        x_fn = (x_fn * 255.0 - 127.5) / 128.0
        emb_fn = self.facenet(x_fn)
        emb_fn = nn.functional.normalize(emb_fn, p=2, dim=1)

        return emb_af, emb_fn
