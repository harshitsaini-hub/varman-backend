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
- **EOT-PGD Optimization:** Perturbs facial feature representations in the latent space of deep convolutional recognition backbones (FaceNet, ArcFace) using differentiable color, scale, and affine transformations.
- **Robustness to Compression:** Employs differentiable DCT-based JPEG approximations during optimization so perturbations survive real-world JPEG recompression across quality factors Q50 to Q95.
- **Paste-Aware Bounding:** Calculates gradient projections directly on the aligned $112 \times 112$ crop and smoothly composites the adversarial face back into the source canvas using an elliptical alpha mask.
- **Empirical Protection:** Prevents 1:1 facial verification across standard verification thresholds calibrated at 1% False Accept Rate (FAR), achieving verified protection per configuration under real JPEG compression (Q50–Q90): **F0** achieves ~36.4% to 42.0% 4-model mean (30.4% to 32.9% held-out mean); **F5** achieves ~45.7% to 48.5% 4-model mean (37.5% to 40.3% held-out mean); **F1** achieves ~50.0% to 55.6% 4-model mean (42.1% to 45.2% held-out mean) [M].

### What Varman Does NOT Do
- **Not Inconspicuous to Humans:** Varman does **not** make faces invisible or imperceptible to human observers. Adversarial grain is perceptible upon scrutiny (mean ratings 1.30 to 2.00 on a 0–2 scale across tested settings).
- **Not 100% Evasion ("Bulletproof"):** Protection is mathematically bounded. Empirical protection plateaus at approximately 50–56% under the strongest tested perturbation settings ($\varepsilon=8/255$, mask scale 1.30). It does not guarantee evasion against unknown black-box ensembles or multimodal foundation models (e.g., CLIP).
- **Not Robust to Downscaling:** Adversarial high-frequency patterns are largely destroyed if the image is resized below the native alignment resolution ($112 \times 112$ pixels) or aggressively blurred.
- **Not In-the-Wild Certified:** Empirical verification is established on LFW front-facing portraits with clean backgrounds; performance on severe multi-person occlusions or extreme yaw angles ($>45^\circ$) is unverified.

---

## 2. Project Evolution: Three Generations

| Generation | Era | Optimization Pipeline | Target Models | Protection Rate (1% FAR, JPEG) | Root Outcome / Status |
|---|---|---|---|---|---|
| **AMOR 1.0** | May–Jun 2026 | Full-canvas PGD ($512 \times 512$), Caffe detector | PyTorch FaceNet (Inception) | **0.00% [H]** | Recorded in handover notes (original test script not preserved, see GAP-01) as complete evasion failure. |
| **Varman 2.0** | Jun–Oct 2026 | CPU heuristic noise, frequency DCT, SSIM guards | PyTorch FaceNet | **~0.0% [H]** | Recorded in handover notes (original test script not preserved, see GAP-01) as zero evasion across models. |
| **Varman 3.0** (`adaptive-strength`) | Oct 2026 | Aligned EOT-PGD ($112 \times 112$), differentiable JPEG, mask composite | FaceNet, ArcFace, SFace, MobileFaceNet | **F0: 36–42%<br>F5: 46–49%<br>F1: 50–56% [M]** | **Empirical Breakthrough:** First verified evasion under JPEG compression. Closed as reference. |

---

## 3. Evaluation Setup & Verification Models

Evaluation follows the standard 1:1 verification paradigm on the Labeled Faces in the Wild (LFW) dataset. All similarity scores use the Cosine Distance metric:
$$D_C(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
A pair is matched as the same identity if $D_C(u, v) \le \tau$. Adversarial evasion (protection success) is achieved when $D_C(u_{adv}, v) > \tau$.

Thresholds are strictly calibrated to enforce a **1.00% False Accept Rate (FAR)** against negative pairs:

| Model | Architecture | Backbone / Pre-Training | Input Shape | 1% FAR Threshold ($\tau$) | Role in Evaluation |
|---|---|---|---|---|---|
| **FaceNet** | Inception-ResNet-v1 | CASIA-WebFace | $112 \times 112 \times 3$ | **0.3754 [M]** | Optimization Surrogate & Evaluation |
| **ArcFace** | ResNet-50 (`w600k_r50`) | InsightFace release | $112 \times 112 \times 3$ | **0.1476 [M]** | Optimization Surrogate & Evaluation |
| **SFace** | SFace / OpenCV Zoo | CASIA-WebFace subset | $112 \times 112 \times 3$ | **0.2920 [M]** | **Held-Out Model** (Never seen during PGD) |
| **MobileFaceNet** | MobileFaceNet (Lightweight) | CASIA-WebFace | $112 \times 112 \times 3$ | **0.1688 [M]** | **Held-Out Model** (Never seen during PGD) |

---

## 4. Empirical Tradeoff Results

Summary of multi-quality protection rates (150 LFW pairs) and human conspicuity ratings (10 source groups, 0–2 scale: 0 = None, 1 = Slight, 2 = Clear):

| Preset / Condition | $\varepsilon$ | Mask Scale | 4-Model Ensemble Protection | Held-Out Models Protection | Human Conspicuity Rating | Mean Face SSIM (Def B) | Recommended Operating Point |
|---|---|---|---|---|---|---|---|
| **F0 (Conservative)** | $8/255$ | 1.00 | Q90: 42.03%<br>Q50: 36.39% | Q90: 32.87%<br>Q50: 30.39% | **1.30 / 2.00 [M]**<br>(7 Slight, 3 Clear) | 0.90149 (Q50)<br>0.91583 (Q90) | Lower artifact profile, modest protection. |
| **F5 (Balanced)** | $8/255$ | 1.15 | **Q90: 47.84%**<br>**Q50: 45.73%** | **Q90: 39.58%**<br>**Q50: 39.23%** | **1.50 / 2.00 [M]**<br>(5 Slight, 5 Clear) | 0.88381 (Q50)<br>0.89538 (Q90) | **Authoritative Recommended Preset.** |
| **F1 (Aggressive)** | $8/255$ | 1.30 | Q90: 55.59%<br>Q50: 49.95% | Q90: 45.24%<br>Q50: 42.42% | **2.00 / 2.00 [M]**<br>(0 Slight, 10 Clear) | 0.88027 (Q50) | Ceiling protection; obvious grain halos. |
| **Unprotected Control** | 0 | 0.00 | Q90: 0.35%<br>Q50: 0.35% | Q90: 0.35%<br>Q50: 0.35% | **0.00 / 2.00 [M]**<br>(10 None, 0 Clear) | 0.97089 (Q50) | Baseline control. |

*Note: Differences of 1–2 percentage points across adjacent JPEG qualities (e.g. Q75 vs Q70) fall within empirical sampling noise (~0.7 percentage points per pair per model). Face SSIM (Definition B) is evaluated strictly within the fixed canonical face mask ($\ge 0.10$) on the aligned 112 crop, compared against the uncompressed source.*

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

> [!IMPORTANT]
> **Deployment Note:**
> - **Default API Engine:** The deployed FastAPI service (`/images/protect`) defaults to the fast CPU-based `"Balanced"` engine (`warp_scale=0.015`, `dct_strength=10.0`).
> - **Strong GPU Mode & F5 Preset:** Strong mode and the recommended `"F5"` preset execute iterative EOT-PGD and require a CUDA-capable GPU (`torch.cuda.is_available()`) along with locally cached surrogate weights (`~/.cache/varman/arcface_r50_backbone.pth`).
> - **API Boundary:** The `"F5"` preset is research-validated and evaluated via benchmark suites, but is **not** selectable through the `/images/protect` web endpoint.

> [!NOTE]
> **Security & Privacy Note:**
> Contributors and operators must adhere to the repository security standards outlined in [`docs/SECURITY_AND_PRIVACY_RULES.md`](docs/SECURITY_AND_PRIVACY_RULES.md). Never commit passwords, database files, salts, or personal emails. Pre-push repository hygiene tests can be verified via `pytest tests/test_repo_hygiene.py`.

### Running Verification Tests
All unit and regression tests should be executed with a hard timeout to prevent hanging:
```bash
pytest --maxfail=1 --timeout=60 tests/
```


---

## 7. Technology Stack

Built strictly from verified repository configurations, package manifests (`requirements.txt`, `package.json`), and Docker deployment definitions:

### Frontend Client & Zero-Knowledge Vault
- **Framework & Runtime:** React 19 (`react`, `react-dom`) with modern Functional Components & Hooks.
- **Build Tooling:** Vite 8 (`@vitejs/plugin-react`) with fast ESM HMR and tree-shaken rollups.
- **Client Routing:** React Router 7 (`react-router-dom`) with guarded private routes.
- **Styling & Motion:** Tailwind CSS, Lucide React icons (`lucide-react`), and interactive WebGL canvas shaders (`ogl`).
- **Client-Side Cryptography:** Web Crypto API (`window.crypto.subtle`):
  - PBKDF2 (SHA-256, 100,000 iterations) derives an ephemeral 256-bit symmetric encryption key from the user's password and persistent `vault_salt`.
  - AES-256-GCM authenticated encryption and decryption for local media protection.
- **Storage & State:**
  - `sessionStorage`: Ephemeral CryptoKey retention across tab reloads (keys never touch disk or backend).
  - IndexedDB: Local encrypted vault storage caching encrypted ciphertext blobs.
- **API Communication:** Axios (`axios`) with request interceptors for JWT Bearer token injection and response interceptors for graceful 401 session recovery.
- **Feedback & Notifications:** React Hot Toast (`react-hot-toast`).

### Backend Service Architecture
- **Web Framework:** FastAPI 0.115.6 with ASGI runner Uvicorn 0.34.0.
- **Database & Async ORM:** SQLAlchemy 2.0 (AsyncIO engine) with cross-backend support:
  - SQLite via `aiosqlite` (default local & container-ephemeral storage)
  - PostgreSQL via `asyncpg` (recommended production persistent storage)
  - Automatic declarative table bootstrap on startup via `Base.metadata.create_all`.
- **Authentication & Security:**
  - OAuth2 Password Bearer flow.
  - JWT Access & Refresh Tokens (`python-jose` with cryptography backend).
  - Password Hashing: Passlib with Bcrypt (`bcrypt==3.2.2`).
  - OTP Verification: Constant-time comparison (`hmac.compare_digest`) of HMAC-SHA256 OTP hashes.
- **Email Delivery:** Python standard library `smtplib` over STARTTLS / SSL with multipart MIME (HTML + plain text fallback) and development console fallback.
- **Configuration & Validation:** Pydantic Settings 2.7.0, `python-dotenv`, `email-validator`.
- **Image Processing & Vision:** OpenCV (`cv2`), Pillow (PIL), NumPy, SciPy, imagehash.

### Adversarial ML & Protection Engine
- **Deep Learning Framework:** PyTorch 2.2+ (CPU-optimized build in Docker, CUDA supported).
- **Optimization Algorithm:** Expectation-over-Transformations Projected Gradient Descent (EOT-PGD).
- **Differentiable JPEG Simulation:** PyTorch 8x8 block Discrete Cosine Transform (DCT) & Inverse DCT with differentiable quantization matrices (Q50–Q95).
- **Surrogate Vision Models:**
  - FaceNet: `InceptionResnetV1` (CASIA-WebFace pretrained via `facenet-pytorch`).
  - ArcFace: `ResNet-50` (`w600k_r50` backbone via ONNX Runtime 1.18+).
  - MobileFaceNet & SFace: Held-out evaluation backbones.
- **Face Localization & Alignment:** MTCNN landmark detection with affine warping to canonical $112 \times 112$ geometry.
- **Compositing:** Smooth elliptical feathering and paste-aware gradient masking.
- **Evaluation Metrics:** Scikit-learn, Scikit-image (SSIM Definition B, PSNR).

### Deployment & Cloud Infrastructure
- **Containerization:** Docker with `python:3.10-slim` base, non-root write permissions on `/code`, and PyTorch CPU-only wheels for lean image sizes.
- **Cloud Hosting:** Hugging Face Spaces (Docker SDK), serving Uvicorn on `0.0.0.0:8000`.
- **Persistence Model:** Containers run statelessly on free-tier compute. SQLite databases are ephemeral unless mounted to persistent storage or configured with an external managed PostgreSQL database URL (`DATABASE_URL`).

---

## 8. Authentication & Password Recovery Workflow

Varman provides self-contained authentication and an encrypted password reset pipeline:

1. **Atomic Registration (`POST /api/auth/register`):**
   - Atomically registers the user, generates a unique 64-character hexadecimal `vault_salt`, hashes the password with bcrypt, and returns an access token and `vault_salt` with `201 Created`.
   - Prevents duplicate registration with `409 Conflict`.
2. **Forgot Password OTP Request (`POST /api/auth/forgot-password/request`):**
   - Generates a cryptographically secure 6-digit one-time password via `secrets.randbelow(1000000)`.
   - Hashes the OTP using HMAC-SHA256 keyed with `JWT_SECRET` before persisting to `password_reset_otps` (raw OTP is never stored).
   - Enforces a 60-second cooldown, 3 requests/hour limit per email, and 10 requests/hour limit per IP address.
   - Configurable account enumeration protection via `REVEAL_UNREGISTERED_EMAIL` (`False` returns a generic 200 response; `True` returns 404).
   - Dispatches reset instructions via SMTP with STARTTLS.
   - For technical schema details and error codes, see [`docs/FORGOT_PASSWORD_API.md`](docs/FORGOT_PASSWORD_API.md).
3. **Password Reset (`POST /api/auth/forgot-password/reset`):**
   - Compares the submitted OTP using constant-time `hmac.compare_digest`.
   - Locks out after 5 consecutive failed attempts to thwart brute-force attacks.
   - Preserves `vault_salt` so user identity remains intact.
   - **Important Vault Notice:** Because vault media is encrypted on the client side using keys derived from the user's password, resetting the account password means previously encrypted vault images cannot be decrypted unless the previous password is known.

---

## 9. Known Limitations & Research Boundary

1. **Human Evaluation Sample Size:** Visual conspicuity ratings are based on 10 standardized source-image groups evaluated by a single expert evaluator under consistent display conditions.
2. **SSIM Decoupling:** Structural Similarity (SSIM Definition B: fixed canonical face mask $\ge 0.10$ on the aligned 112 crop, compared against the uncompressed source) does not strictly track human visibility rankings. For example, F0 exhibits higher SSIM than F5 (0.901 vs 0.884), yet both can exhibit clearly perceptible patterns depending on individual skin texture.
3. **Smooth-Skin Hypothesis:** Qualitative inspection indicates that subjects with smooth, untextured skin reveal high-frequency adversarial noise more prominently than subjects with textured skin, facial hair, or complex backgrounds. This remains an unvalidated observational hypothesis.
4. **Compression Scope:** Validation was conducted exclusively under standard baseline JPEG compression (Q50–Q95). Behavioral characteristics under modern codecs (AVIF, WebP) or video compression (H.264/HEVC) have not been evaluated.
5. **Training Lineage of Evaluated Models:** `w600k_r50` (ArcFace) and `w600k_mbf` (MobileFaceNet) appear to share an upstream training lineage within the InsightFace release family [H]. Held-out denotes that the model was unseen during PGD and has a different architecture, not that it was trained on independent data.

---

## 10. Historical Documentation

For exhaustive technical background, failure analyses, mathematical formulations, and reproduction logs, consult the companion research monograph:
- **Full History Document (125 Pages):** [`Varman_Research_History.pdf`](docs/Varman_Research_History.pdf)
- **Markdown Source:** [`Varman_Research_History.md`](../Varman-research/reports/history_doc/Varman_Research_History.md)
- **Untraceable Gaps Register:** [`GAPS.md`](../Varman-research/reports/history_doc/GAPS.md)
- **Source Audit:** [`SOURCE_INVENTORY.md`](../Varman-research/reports/history_doc/SOURCE_INVENTORY.md)

---
*Varman Research Project • 2026 • Reference Archive*
