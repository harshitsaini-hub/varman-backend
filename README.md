---
title: Varman Backend
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Varman: Adversarial Facial Privacy Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Status: Closed](https://img.shields.io/badge/Research_Status-Closed_(2026--10--17)-red.svg)]()

Varman is a research-grade adversarial protection engine designed to safeguard personal facial imagery against unauthorized automated facial recognition (FR). Operating on PyTorch and ONNX Runtime, Varman optimizes adversarial perturbations directly within aligned facial embedding spaces using Expectation-over-Transformations Projected Gradient Descent (EOT-PGD), enabling robustness against downstream JPEG re-encoding.

> **Research Closure Notice (October 17, 2026):**  
> The research, benchmarking, and tuning phase of Project Varman is officially **closed**. No further hyperparameter sweeps, surrogate architecture modifications, or PGD iterations are planned. This repository serves as the definitive reference implementation of the Varman 3.0 architecture. For the complete 125-page historical monograph and empirical audit, see [`Varman_Research_History.pdf`](../Varman-research/reports/history_doc/Varman_Research_History.pdf).

---

## 1. Scope: Capabilities and Non-Goals

### What Varman Does
- **EOT-PGD Optimization:** Perturbs facial feature representations in the latent space of deep convolutional recognition backbones (FaceNet, ArcFace/Glint360k) using differentiable color, scale, and affine transformations.
- **Robustness to Compression:** Employs differentiable DCT-based JPEG approximations during optimization so perturbations survive real-world JPEG recompression across quality factors Q50 to Q95.
- **Paste-Aware Bounding:** Calculates gradient projections directly on the aligned $112 \times 112$ crop and smoothly composites the adversarial face back into the source canvas using an elliptical alpha mask.
- **Empirical Protection:** Prevents 1:1 facial verification across standard verification thresholds calibrated at 1% False Accept Rate (FAR), achieving roughly 40% to 56% protection against commercial-grade deep models under JPEG compression.

### What Varman Does NOT Do
- **Not Inconspicuous to Humans:** Varman does **not** make faces invisible or imperceptible to human observers. Adversarial grain is perceptible upon scrutiny (mean ratings 1.30 to 2.00 on a 0–2 scale across tested settings).
- **Not 100% Evasion ("Bulletproof"):** Protection is mathematically bounded. Empirical protection plateaus at approximately 50–56% under the strongest tested perturbation settings ($\varepsilon=8/255$, mask scale 1.30). It does not guarantee evasion against unknown black-box ensembles or multimodal foundation models (e.g., CLIP).
- **Not Robust to Downscaling:** Adversarial high-frequency patterns are largely destroyed if the image is resized below the native alignment resolution ($112 \times 112$ pixels) or aggressively blurred.
- **Not In-the-Wild Certified:** Empirical verification is established on LFW front-facing portraits with clean backgrounds; performance on severe multi-person occlusions or extreme yaw angles ($>45^\circ$) is unverified.

---

## 2. Project Evolution: Three Generations

| Generation | Era | Optimization Pipeline | Target Models | Protection Rate (1% FAR, JPEG) | Root Outcome / Status |
|---|---|---|---|---|---|
| **AMOR 1.0** | May–Jun 2026 | Full-canvas PGD ($512 \times 512$), Caffe detector | PyTorch FaceNet (Inception) | **0.00% [M]** | **Complete Failure:** Full-canvas noise was discarded by downstream face alignment and cropping. |
| **Varman 2.0** | Jun–Oct 2026 | CPU heuristic noise, frequency DCT, SSIM guards | PyTorch FaceNet | **0.00% [M]** | **Complete Failure:** Heuristic noise was filtered out by deep embedding layers. Zero evasion. |
| **Varman 3.0** (`adaptive-strength`) | Oct 2026 | Aligned EOT-PGD ($112 \times 112$), differentiable JPEG, mask composite | FaceNet, ArcFace, SFace, MobileFaceNet | **40% – 56% [M]** | **Empirical Breakthrough:** First verified evasion under JPEG compression. Closed as reference. |

---

## 3. Evaluation Setup & Verification Models

Evaluation follows the standard 1:1 verification paradigm on the Labeled Faces in the Wild (LFW) dataset. All similarity scores use the Cosine Distance metric:
$$D_C(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
A pair is matched as the same identity if $D_C(u, v) \le \tau$. Adversarial evasion (protection success) is achieved when $D_C(u_{adv}, v) > \tau$.

Thresholds are strictly calibrated to enforce a **1.00% False Accept Rate (FAR)** against negative pairs:

| Model | Architecture | Backbone / Pre-Training | Input Shape | 1% FAR Threshold ($\tau$) | Role in Evaluation |
|---|---|---|---|---|---|
| **FaceNet** | Inception-ResNet-v1 | CASIA-WebFace | $112 \times 112 \times 3$ | **0.3754 [M]** | Optimization Surrogate & Evaluation |
| **ArcFace** | ResNet-50 (`w600k_r50`) | Glint360k | $112 \times 112 \times 3$ | **0.1476 [M]** | Optimization Surrogate & Evaluation |
| **SFace** | SFace / OpenCV Zoo | CASIA-WebFace subset | $112 \times 112 \times 3$ | **0.2920 [M]** | **Held-Out Model** (Never seen during PGD) |
| **MobileFaceNet** | MobileFaceNet (Lightweight) | CASIA-WebFace | $112 \times 112 \times 3$ | **0.1688 [M]** | **Held-Out Model** (Never seen during PGD) |

---

## 4. Empirical Tradeoff Results

Summary of multi-quality protection rates (150 LFW pairs) and human conspicuity ratings (10 source groups, 0–2 scale: 0 = None, 1 = Slight, 2 = Clear):

| Preset / Condition | $\varepsilon$ | Mask Scale | 4-Model Ensemble Protection | Held-Out Models Protection | Human Conspicuity Rating | Mean Face SSIM (Def B) | Recommended Operating Point |
|---|---|---|---|---|---|---|---|
| **F0 (Conservative)** | $8/255$ | 1.00 | Q90: 42.03%<br>Q50: 36.39% | Q90: 32.87%<br>Q50: 30.39% | **1.30 / 2.00 [M]**<br>(7 Slight, 3 Clear) | 0.90149 (Q50)<br>0.91583 (Q90) | Lower artifact profile, modest protection. |
| **F5 (Balanced)** | $8/255$ | 1.15 | **Q90: 47.84%**<br>**Q50: 45.73%** | **Q90: 39.58%**<br>**Q50: 39.23%** | **1.50 / 2.00 [M]**<br>(5 Slight, 5 Clear) | 0.88381 (Q50)<br>0.89538 (Q90) | **Authoritative Recommended Preset.** |
| **F1 (Aggressive)** | $8/255$ | 1.30 | Q90: 55.59%<br>Q50: 49.95% | Q90: 45.24%<br>Q50: 42.42% | **2.00 / 2.00 [M]**<br>(0 Slight, 10 Clear) | 0.88027 (Q50)<br>0.89410 (Q90) | Ceiling protection; obvious grain halos. |
| **Unprotected Control** | 0 | 0.00 | Q90: 0.35%<br>Q50: 0.35% | Q90: 0.35%<br>Q50: 0.35% | **0.00 / 2.00 [M]**<br>(10 None, 0 Clear) | 0.97089 (Q50)<br>0.98540 (Q90) | Baseline control. |

*Note: Differences of 1–2 percentage points across adjacent JPEG qualities (e.g. Q75 vs Q70) fall within empirical sampling noise (~0.7 percentage points per pair per model).*

---

## 5. Recommended Configuration: Preset F5

Preset `F5` provides the optimal balance between facial verification evasion and visual artifacting:
- **Perturbation Bound ($\varepsilon$):** $8 / 255 \approx 0.03137$
- **Mask Scale Factor:** `1.15` (covers core facial geometry while feathering into jawline and hairline)
- **PGD Step Size ($\alpha$):** $1.5 / 255 \approx 0.00588$
- **Iterations ($N$):** 20 steps
- **EOT Transforms:** Affine jittering ($\pm 2^\circ$, $\pm 2\%$ scale), brightness jittering ($\pm 5\%$), differentiable JPEG simulation.

```python
from app.presets import get_preset_config
from app.engine.protection import AdversarialProtectionEngine

# Load the authoritative balanced preset
config = get_preset_config("F5")
engine = AdversarialProtectionEngine(config)

# Apply protection to an input RGB image
protected_image = engine.protect(source_image_bgr)
```

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended for PGD optimization, fallback to CPU supported)
- Windows / Linux OS

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/<org>/Varman-backend.git
cd Varman-backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### Running Verification Tests
All unit and regression tests should be executed with a hard timeout to prevent hanging:
```bash
pytest --maxfail=1 --timeout=60 tests/
```

---

## 7. Known Limitations & Research Boundary

1. **Human Evaluation Sample Size:** Visual conspicuity ratings are based on 10 standardized source-image groups evaluated by a single expert evaluator under consistent display conditions.
2. **SSIM Decoupling:** Structural Similarity (SSIM Definition B: bounding-box crop) does not strictly track human visibility rankings. For example, F0 exhibits higher SSIM than F5 (0.901 vs 0.884), yet both can exhibit clearly perceptible patterns depending on individual skin texture.
3. **Smooth-Skin Hypothesis:** Qualitative inspection indicates that subjects with smooth, untextured skin reveal high-frequency adversarial noise more prominently than subjects with textured skin, facial hair, or complex backgrounds. This remains an unvalidated observational hypothesis.
4. **Compression Scope:** Validation was conducted exclusively under standard baseline JPEG compression (Q50–Q95). Behavioral characteristics under modern codecs (AVIF, WebP) or video compression (H.264/HEVC) have not been evaluated.

---

## 8. Historical Documentation

For exhaustive technical background, failure analyses, mathematical formulations, and reproduction logs, consult the companion research monograph:
- **Full History Document (125 Pages):** [`Varman_Research_History.pdf`](../Varman-research/reports/history_doc/Varman_Research_History.pdf)
- **Markdown Source:** [`Varman_Research_History.md`](../Varman-research/reports/history_doc/Varman_Research_History.md)
- **Untraceable Gaps Register:** [`GAPS.md`](../Varman-research/reports/history_doc/GAPS.md)
- **Source Audit:** [`SOURCE_INVENTORY.md`](../Varman-research/reports/history_doc/SOURCE_INVENTORY.md)

---
*Varman Research Project • 2026 • Reference Archive*
