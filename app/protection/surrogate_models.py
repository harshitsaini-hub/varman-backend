import os
import torch
import torch.nn as nn
import torchvision.models as models
import open_clip


# ── Legacy CLIP+ResNet Ensemble ───────────────────────────────────────────────

class SurrogateEnsemble:
    def __init__(self, device="cpu"):
        self.device = device

        # Load CLIP ViT-B/32
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
        )
        self.clip_model.eval()

        # Load ResNet50 as a generic feature extractor
        self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_model.to(device)
        self.resnet_model.eval()

        # Freeze models
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.resnet_model.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        """
        Extract concatenated features from the ensemble.
        x: (B, C, H, W) tensor in [0, 1]
        """
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        x_norm    = (x - mean) / std
        x_resized = torch.nn.functional.interpolate(x_norm, size=(224, 224), mode='bilinear', align_corners=False)

        clip_features   = self.clip_model.encode_image(x_resized)
        resnet_features = self.resnet_model(x_resized)
        return torch.cat([clip_features, resnet_features], dim=1)


# ── FaceNet (VGGFace2) Surrogate ─────────────────────────────────────────────

class FaceNetSurrogate:
    def __init__(self, device="cpu"):
        from facenet_pytorch import InceptionResnetV1
        self.device = device

        # InceptionResnetV1 trained on VGGFace2 — 512-dim embedding
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        for param in self.facenet.parameters():
            param.requires_grad = False

    def extract_features(self, x, face_bbox=None):
        """
        Extract identity embedding.
        x:         (B, C, H, W) tensor in [0, 1]
        face_bbox: optional (x1, y1, x2, y2) pixel coords — crop face before embedding.
                   Matches what MTCNN does in real face-recognition pipelines.
        """
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            x = x[:, :, y1:y2, x1:x2]

        x_resized = torch.nn.functional.interpolate(
            x, size=(160, 160), mode='bilinear', align_corners=False
        )
        # VGGFace2 normalisation: (pixel - 127.5) / 128  ≡  (pixel - 0.5) / 0.5
        x_norm = (x_resized - 0.5) / 0.5
        return self.facenet(x_norm)


# ── ArcFace IResNet50 backbone (pure PyTorch) ────────────────────────────────
# Architecture: IResNet50 as used in ArcFace (CVPR 2019) / InsightFace.
# Weights:      ms1mv3_arcface_r50 — locally cached, 475 verified keys.
# Cache path:   ~/.cache/varman/arcface_r50_backbone.pth

class _IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1      = nn.BatchNorm2d(in_planes)
        self.conv1    = nn.Conv2d(in_planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(planes)
        self.prelu    = nn.PReLU(planes)
        self.conv2    = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn3      = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride   = stride

    def forward(self, x):
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
    """IResNet backbone as used in ArcFace (CVPR 2019)."""

    def __init__(self, block, layers, dropout=0.0, num_features=512):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.prelu  = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2      = nn.BatchNorm2d(512)
        self.dropout  = nn.Dropout(p=dropout)
        self.fc       = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
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


def _iresnet50():
    return _IResNet(_IBasicBlock, [3, 4, 14, 3])


class ArcFaceSurrogate:
    """
    IResNet50 ArcFace surrogate — pure PyTorch, gradient-compatible.

    Weights: ms1mv3_arcface_r50_fp16 — manually verified (475 keys).
    Cache:   ~/.cache/varman/arcface_r50_backbone.pth  (~166 MB)

    Input:  (B, 3, H, W) in [0, 1] — face-cropped recommended
    Output: (B, 512) L2-normalised embedding
    Norm:   (x * 255 - 127.5) / 128.0  (ArcFace standard)
    """

    WEIGHTS_PATH = os.path.join(
        os.path.expanduser("~"), ".cache", "varman", "arcface_r50_backbone.pth"
    )

    def __init__(self, device="cpu"):
        self.device = device
        self.model  = _iresnet50().eval().to(device)

        if not os.path.exists(self.WEIGHTS_PATH):
            raise FileNotFoundError(
                f"ArcFace weights not found at {self.WEIGHTS_PATH}.\n"
                "Download ms1mv3_arcface_r50_fp16.pth and place it there."
            )

        # Load fp16 checkpoint → cast to fp32 for gradient computation
        state = torch.load(self.WEIGHTS_PATH, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        # Strip DataParallel 'module.' prefix if present
        state = {k.replace("module.", ""): v.float() for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, x, face_bbox=None):
        """
        x:         (B, C, H, W) tensor in [0, 1]
        face_bbox: optional (x1, y1, x2, y2) pixel coords — crop before embed.
        """
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            x = x[:, :, y1:y2, x1:x2]

        # ArcFace standard: 112×112 input, norm = (pixel*255 - 127.5) / 128
        x_resized = torch.nn.functional.interpolate(
            x, size=(112, 112), mode='bilinear', align_corners=False
        )
        x_norm = (x_resized * 255.0 - 127.5) / 128.0
        emb = self.model(x_norm)
        # L2-normalise — matches how ArcFace embeddings are compared
        return torch.nn.functional.normalize(emb, p=2, dim=1)


# ── Dual Surrogate Ensemble: FaceNet + ArcFace ────────────────────────────────

class FaceNetArcFaceEnsemble:
    """
    Dual-surrogate ensemble: FaceNet (VGGFace2) + ArcFace (IResNet50, MS1MV3).

    Perturbing against two architecturally distinct models trained on different
    datasets maximises transferability to unseen real-world face recognition
    systems (deepfake pipelines, reverse-image-search, etc.).

    extract_features() returns a TUPLE: (facenet_emb, arcface_emb)

    Loss in engine:
        fn_adv,  arc_adv  = ensemble.extract_features(x_adv,  face_bbox=bbox)
        fn_orig, arc_orig = ensemble.extract_features(x_orig, face_bbox=bbox)
        loss = (cosine_similarity(fn_adv, fn_orig) +
                cosine_similarity(arc_adv, arc_orig)).mean()
    """

    def __init__(self, device="cpu"):
        self.facenet = FaceNetSurrogate(device=device)
        self.arcface = ArcFaceSurrogate(device=device)

    def extract_features(self, x, face_bbox=None):
        """Returns (facenet_embedding, arcface_embedding) — both (B, 512) tensors."""
        fn_emb  = self.facenet.extract_features(x, face_bbox=face_bbox)
        arc_emb = self.arcface.extract_features(x, face_bbox=face_bbox)
        return fn_emb, arc_emb


# ── VAE Surrogate (kept for reference — not used in active pipeline) ──────────

class VAESurrogate:
    def __init__(self, device="cpu"):
        from diffusers import AutoencoderKL

        self.device = device
        local_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "sd-vae-ft-mse"
        )
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(
                f"VAE model not found at {local_model_path}. Run download_vae.py first."
            )
        self.vae = AutoencoderKL.from_pretrained(local_model_path, local_files_only=True)
        self.vae.eval().to(device)
        for param in self.vae.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        """x: (B, C, H, W) in [0, 1] → latent mean (B, 4, H/8, W/8)"""
        x_scaled  = (x * 2.0) - 1.0
        x_resized = torch.nn.functional.interpolate(
            x_scaled, size=(512, 512), mode='bilinear', align_corners=False
        )
        return self.vae.encode(x_resized).latent_dist.mean
