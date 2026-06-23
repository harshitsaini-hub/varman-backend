"""
Varman Black-Box Transferability Benchmark
==========================================
Tests whether adversarial perturbations generated against FaceNet (white-box)
also fool ArcFace (black-box) — proving the protection generalises to unseen
real-world face recognition systems.

Usage:
    python varman_benchmark.py original.jpg protected.jpg [downloaded_instagram.jpg]

Arguments:
    original.jpg             — Unprotected source image
    protected.jpg            — Varman-protected image (pre-compression, local)
    downloaded_instagram.jpg — Optional: image re-downloaded from Instagram after upload
                               (tests survival through platform JPEG compression)

Output:
    Benchmark table printed to console with PASS/FAIL verdicts.

VRAM strategy:
    FaceNet is loaded → distances measured → model explicitly deleted → VRAM freed.
    ArcFace is then loaded → distances measured → model deleted.
    Peak VRAM never exceeds ~500MB — safe for RTX 2050 (4GB).
"""

import sys
import gc
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Allow running from the benchmarks/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Thresholds ────────────────────────────────────────────────────────────────
FACENET_THRESHOLD = 0.40   # Cosine distance above this → "different person" (FaceNet)
ARCFACE_THRESHOLD = 0.40   # Cosine distance above this → "different person" (ArcFace)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load image as (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0).to(device)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance in [0, 2]. 0 = identical, 2 = opposite directions."""
    sim = torch.nn.functional.cosine_similarity(a, b).item()
    return 1.0 - sim


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item()


def free_vram(obj):
    """Aggressively free VRAM after a model is no longer needed."""
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def detect_face_bbox(image_path: str):
    """
    Use MTCNN to detect the primary face bounding box.
    Returns (x1, y1, x2, y2) in pixel coords, or None if no face found.
    """
    try:
        from facenet_pytorch import MTCNN
        img   = Image.open(image_path).convert("RGB")
        mtcnn = MTCNN(keep_all=False, device="cpu")  # CPU only — tiny model
        boxes, _ = mtcnn.detect(img) # type:ignore
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            return (max(0, x1), max(0, y1), x2, y2)
    except Exception as e:
        print(f"  [MTCNN] Face detection failed ({e}) — using full image for embedding.")
    return None


def crop_face(tensor: torch.Tensor, bbox) -> torch.Tensor:
    """Crop face region from (1, C, H, W) tensor using pixel bbox."""
    if bbox is None:
        return tensor
    x1, y1, x2, y2 = bbox
    return tensor[:, :, y1:y2, x1:x2]


# ── FaceNet measurement ───────────────────────────────────────────────────────

def measure_facenet(original_path, protected_path, downloaded_path, device):
    """
    Load FaceNet (white-box surrogate), measure cosine/euclidean distances,
    return results dict, then free VRAM.
    """
    print("\n[1/2] Loading FaceNet (VGGFace2) ...")
    from facenet_pytorch import InceptionResnetV1

    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in facenet.parameters():
        p.requires_grad = False

    def embed(image_path, bbox):
        tensor = load_image_tensor(image_path, device)
        face   = crop_face(tensor, bbox)
        face_r = torch.nn.functional.interpolate(
            face, size=(160, 160), mode="bilinear", align_corners=False
        )
        face_n = (face_r - 0.5) / 0.5
        with torch.no_grad():
            return facenet(face_n)

    print("  Detecting face with MTCNN ...")
    bbox = detect_face_bbox(original_path)
    print(f"  Face bbox: {bbox}" if bbox else "  No face detected — using full image.")

    orig_emb = embed(original_path,  bbox)
    prot_emb = embed(protected_path, bbox)

    results = {
        "fn_cos_orig_prot": cosine_distance(orig_emb, prot_emb),
        "fn_euc_orig_prot": euclidean_distance(orig_emb, prot_emb),
    }

    if downloaded_path:
        dl_emb = embed(downloaded_path, bbox)
        results["fn_cos_orig_dl"]  = cosine_distance(orig_emb, dl_emb)
        results["fn_euc_orig_dl"]  = euclidean_distance(orig_emb, dl_emb)
        results["fn_cos_prot_dl"]  = cosine_distance(prot_emb, dl_emb)

    print("  FaceNet done. Freeing VRAM ...")
    free_vram(facenet)
    return results


# ── ArcFace measurement ───────────────────────────────────────────────────────

def measure_arcface(original_path, protected_path, downloaded_path, device):
    """
    Load ArcFace (IResNet50, black-box — never seen the noise),
    measure distances, return results dict, then free VRAM.
    """
    print("\n[2/2] Loading ArcFace (IResNet50 / MS1MV3) ...")
    from app.protection.surrogate_models import ArcFaceSurrogate

    arcface = ArcFaceSurrogate(device=str(device))

    def embed(image_path, bbox):
        tensor = load_image_tensor(image_path, device)
        return arcface.extract_features(tensor, face_bbox=bbox)

    print("  Detecting face with MTCNN ...")
    bbox = detect_face_bbox(original_path)
    print(f"  Face bbox: {bbox}" if bbox else "  No face detected — using full image.")

    orig_emb = embed(original_path,  bbox)
    prot_emb = embed(protected_path, bbox)

    results = {
        "arc_cos_orig_prot": cosine_distance(orig_emb, prot_emb),
        "arc_euc_orig_prot": euclidean_distance(orig_emb, prot_emb),
    }

    if downloaded_path:
        dl_emb = embed(downloaded_path, bbox)
        results["arc_cos_orig_dl"]  = cosine_distance(orig_emb, dl_emb)
        results["arc_euc_orig_dl"]  = euclidean_distance(orig_emb, dl_emb)
        results["arc_cos_prot_dl"]  = cosine_distance(prot_emb, dl_emb)

    print("  ArcFace done. Freeing VRAM ...")
    free_vram(arcface)
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def verdict(distance, threshold):
    if distance > threshold:
        return f"✅ PASS  ({distance:.4f} > {threshold})"
    return f"❌ FAIL  ({distance:.4f} ≤ {threshold})"


def print_report(fn: dict, arc: dict, has_instagram: bool):
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  VARMAN BLACK-BOX TRANSFERABILITY BENCHMARK")
    print(f"  White-box surrogate : FaceNet (VGGFace2) — used to GENERATE noise")
    print(f"  Black-box target    : ArcFace (IResNet50 / MS1MV3) — NEVER seen noise")
    print(f"{'═' * 72}")

    print(f"\n{'MODEL':<12} {'COMPARISON':<32} {'COS DIST':>10}  VERDICT")
    print(sep)

    print(f"{'FaceNet':<12} {'Original → Protected':<32} "
          f"{fn['fn_cos_orig_prot']:>10.4f}  "
          f"{verdict(fn['fn_cos_orig_prot'], FACENET_THRESHOLD)}")

    if has_instagram:
        print(f"{'FaceNet':<12} {'Original → Downloaded (IG)':<32} "
              f"{fn['fn_cos_orig_dl']:>10.4f}  "
              f"{verdict(fn['fn_cos_orig_dl'], FACENET_THRESHOLD)}")
        print(f"{'FaceNet':<12} {'Protected → Downloaded (IG)':<32} "
              f"{fn['fn_cos_prot_dl']:>10.4f}  (compression drift)")

    print(sep)

    print(f"{'ArcFace':<12} {'Original → Protected':<32} "
          f"{arc['arc_cos_orig_prot']:>10.4f}  "
          f"{verdict(arc['arc_cos_orig_prot'], ARCFACE_THRESHOLD)}")

    if has_instagram:
        print(f"{'ArcFace':<12} {'Original → Downloaded (IG)':<32} "
              f"{arc['arc_cos_orig_dl']:>10.4f}  "
              f"{verdict(arc['arc_cos_orig_dl'], ARCFACE_THRESHOLD)}")
        print(f"{'ArcFace':<12} {'Protected → Downloaded (IG)':<32} "
              f"{arc['arc_cos_prot_dl']:>10.4f}  (compression drift)")

    print(sep)

    fn_pass  = fn["fn_cos_orig_prot"]  > FACENET_THRESHOLD
    arc_pass = arc["arc_cos_orig_prot"] > ARCFACE_THRESHOLD

    print(f"\n  TRANSFERABILITY RESULT:")
    if fn_pass and arc_pass:
        print(f"  ✅ FULL TRANSFER — Protection fools BOTH FaceNet AND ArcFace.")
        print(f"     Perturbation generalises to architecturally distinct models.")
        print(f"     Real-world deepfake pipelines would be disrupted.")
    elif fn_pass and not arc_pass:
        print(f"  ⚠️  PARTIAL — Fools FaceNet (white-box) but NOT ArcFace (black-box).")
        print(f"     Transferability insufficient. Consider ensemble surrogate training.")
    elif not fn_pass and arc_pass:
        print(f"  ⚠️  ANOMALY — ArcFace fooled but FaceNet (training model) is not.")
        print(f"     Something is wrong upstream — check the pipeline.")
    else:
        print(f"  ❌ FAIL — Protection does not fool either model.")
        print(f"     Verify cosine loss is decreasing during training.")

    print(f"{'═' * 72}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    original_path  = sys.argv[1]
    protected_path = sys.argv[2]
    downloaded_path = sys.argv[3] if len(sys.argv) > 3 else None
    has_instagram   = downloaded_path is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Varman Benchmark] Device: {device}")
    if torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"[Varman Benchmark] VRAM:   {vram_mb} MB")

    print(f"[Varman Benchmark] Original:  {original_path}")
    print(f"[Varman Benchmark] Protected: {protected_path}")
    if has_instagram:
        print(f"[Varman Benchmark] Downloaded (IG): {downloaded_path}")

    # Sequential: free VRAM between models — safe on 4GB RTX 2050
    fn_results  = measure_facenet(original_path, protected_path, downloaded_path, device)
    arc_results = measure_arcface(original_path, protected_path, downloaded_path, device)

    print_report(fn_results, arc_results, has_instagram)


if __name__ == "__main__":
    main()
