import torch
import torchvision.models as models
import open_clip

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
        # Normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        x_norm = (x - mean) / std
        
        # Resize to 224x224
        x_resized = torch.nn.functional.interpolate(x_norm, size=(224, 224), mode='bilinear', align_corners=False)
        
        clip_features = self.clip_model.encode_image(x_resized)
        resnet_features = self.resnet_model(x_resized)
        
        return torch.cat([clip_features, resnet_features], dim=1)

class FaceNetSurrogate:
    def __init__(self, device="cpu"):
        from facenet_pytorch import InceptionResnetV1
        self.device = device
        
        # Load FaceNet (InceptionResnetV1 trained on VGGFace2)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Freeze model
        for param in self.facenet.parameters():
            param.requires_grad = False

    def extract_features(self, x, face_bbox=None):
        """
        Extract identity embedding.
        x: (B, C, H, W) tensor in [0, 1]
        face_bbox: optional (x1, y1, x2, y2) in pixel coords to crop face first.
                   If provided, crop the face region before resizing to 160x160.
                   This matches what MTCNN does in real face-recognition pipelines.
        """
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            # Crop face region from the tensor (B, C, H, W)
            x_face = x[:, :, y1:y2, x1:x2]
        else:
            x_face = x

        # Resize to 160x160 as expected by FaceNet
        x_resized = torch.nn.functional.interpolate(
            x_face, size=(160, 160), mode='bilinear', align_corners=False
        )
        
        # Normalization for VGGFace2 (mean=127.5/255=0.5, std=128/255=0.5) roughly
        x_norm = (x_resized - 0.5) / 0.5
        
        return self.facenet(x_norm)

class VAESurrogate:
    def __init__(self, device="cpu"):
        import os
        from diffusers import AutoencoderKL
        
        self.device = device
        
        # Load from local directory
        local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "sd-vae-ft-mse")
        
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"VAE model not found at {local_model_path}. Please run download_vae.py first.")
            
        self.vae = AutoencoderKL.from_pretrained(local_model_path, local_files_only=True)
        self.vae.eval().to(device)
        
        # Freeze model
        for param in self.vae.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        """
        Encode image into the Diffusion Latent Space.
        x: (B, C, H, W) tensor in [0, 1]
        """
        # VAE usually expects images scaled to [-1, 1]
        x_scaled = (x * 2.0) - 1.0
        
        # Resize to 512x512 (divisible by 8)
        x_resized = torch.nn.functional.interpolate(x_scaled, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Use the mean of the latent distribution as our embedding E(x)
        latent = self.vae.encode(x_resized).latent_dist.mean
        return latent
