import os
import sys

# Ensure Varman backend is in path
sys.path.insert(0, r"D:\JavaProjects\Varman\Varman-backend")

from app.protection.engine import protect_image_pipeline

def main():
    artifact_dir = r"C:\Users\ASUS F17\.gemini\antigravity\brain\4e9437cb-0438-4359-8592-4929a00a3092"
    
    bases = [
        "photo_6183939895160673175_y",
        "photo_6183939895160673176_y",
        "photo_6183939895160673177_y",
        "photo_6183939895160673178_y"
    ]
    
    for base in bases:
        orig_path = os.path.join(artifact_dir, f"{base}.jpg")
        prot_path = os.path.join(artifact_dir, f"{base}_pgd.png")
        
        print(f"Processing {base}...")
        try:
            res = protect_image_pipeline(orig_path, prot_path)
            print(f"  -> SSIM: {res['ssim']:.4f}")
            print(f"  -> CLIP cos: {res['clip_cosine_final']:.4f}")
            print(f"  -> ResNet cos: {res['resnet_cosine_final']:.4f}")
        except Exception as e:
            print(f"Failed on {base}: {e}")

if __name__ == "__main__":
    main()
