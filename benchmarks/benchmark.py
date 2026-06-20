"""
Varman Benchmark: Does protection measurably alter facial identity embeddings?

Usage:
    python benchmark.py --image path/to/face.jpg

What it does:
    1. Takes your original image
    2. Runs it through Varman's protection engine
    3. Extracts FaceNet embeddings from BOTH images
    4. Computes cosine distance between them
    5. Reports the verdict

What counts as success:
    - distance(original, original) should be ~0.0
    - distance(original, protected) should be SIGNIFICANTLY higher
    - If it's not: current perturbation strategy is ineffective. We stop.
"""

import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def extract_embedding(model, mtcnn, image_path, device):
    """Extract FaceNet embedding from an image. Returns None if no face found."""
    img = Image.open(image_path).convert("RGB")

    # Detect and align face
    face = mtcnn(img)
    if face is None:
        return None

    # face shape: (3, 160, 160), add batch dim
    face = face.unsqueeze(0).to(device)

    # Extract embedding
    with torch.no_grad():
        embedding = model(face)

    return embedding.cpu().numpy().flatten()


def cosine_distance(a, b):
    """Cosine distance: 0 = identical, 2 = opposite."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 2.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def main():
    parser = argparse.ArgumentParser(description="Varman Benchmark: Identity Embedding Distance")
    parser.add_argument("--image", type=str, required=True, help="Path to a face image (jpg/png)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[FAIL] Image not found: {args.image}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input:  {args.image}")
    print("-" * 60)

    # ── Load FaceNet ──────────────────────────────────────────
    print("Loading FaceNet (InceptionResnetV1)...")
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # ── Extract embedding from ORIGINAL ───────────────────────
    print("Extracting embedding from original...")
    emb_original = extract_embedding(model, mtcnn, args.image, device)
    if emb_original is None:
        print("[FAIL] No face detected in original image.")
        print("       Use an image with a clearly visible face.")
        sys.exit(1)

    # ── Self-distance sanity check ────────────────────────────
    self_cos = cosine_distance(emb_original, emb_original)
    self_euc = euclidean_distance(emb_original, emb_original)
    print(f"  Self-distance (sanity check): cosine={self_cos:.6f}, euclidean={self_euc:.6f}")

    # ── Run Varman protection ─────────────────────────────────
    protected_path = os.path.join(
        os.path.dirname(os.path.abspath(args.image)),
        "protected_" + os.path.basename(args.image),
    )

    print(f"\nRunning Varman protection engine...")
    start = time.time()

    from app.protection.engine import protect_image_pipeline

    result = protect_image_pipeline(args.image, protected_path)
    elapsed = time.time() - start

    print(f"  Engine result: {result}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved: {protected_path}")

    # ── Extract embedding from PROTECTED ──────────────────────
    print("\nExtracting embedding from protected image...")
    emb_protected = extract_embedding(model, mtcnn, protected_path, device)
    if emb_protected is None:
        print("[FAIL] No face detected in protected image.")
        print("       The perturbation may have destroyed the face region.")
        sys.exit(1)

    # ── Compute distances ─────────────────────────────────────
    cos_dist = cosine_distance(emb_original, emb_protected)
    euc_dist = euclidean_distance(emb_original, emb_protected)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Cosine distance  (original vs protected): {cos_dist:.6f}")
    print(f"  Euclidean distance (original vs protected): {euc_dist:.6f}")
    print()

    # ── Interpret ─────────────────────────────────────────────
    # FaceNet typical thresholds:
    #   cosine < 0.4  = same person
    #   cosine > 0.4  = different person
    #   euclidean < 1.0 = same person (on VGGFace2)
    print("INTERPRETATION:")
    print(f"  FaceNet 'same person' threshold: cosine < 0.40")
    print()

    if cos_dist >= 0.40:
        print("  [PASS] Identity embedding significantly altered.")
        print("         Varman IS disrupting facial identity recognition.")
        print("         The perturbation changes who AI thinks this person is.")
    elif cos_dist >= 0.20:
        print("  [PARTIAL] Identity embedding moderately altered.")
        print("            Some disruption detected, but may not be enough")
        print("            to fool all face-recognition systems.")
    else:
        print("  [FAIL] Identity embedding NOT significantly altered.")
        print("         Current perturbation strategy is ineffective.")
        print("         The noise is invisible to FaceNet.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
