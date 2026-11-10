# Varman — Hard Mode Production Integration Audit

**Document Title:** HARD_MODE_INTEGRATION_AUDIT.md  
**Date:** November 9, 2026  
**Auditor:** Antigravity (AI Engineering & Security Lead)  
**Audit Scope:** Varman Backend, Frontend Client, Hugging Face Spaces Deployment, Empirical Research Testbed, Benchmark Artifacts, and CI/CD Pipelines.  
**Audit Mode:** Strict Read-Only Investigation (No application code modified).  

---

## Executive Verdict & Status

```
FINAL STATUS: READY WITH BLOCKERS
```

### Direct Summary of Findings
1. **The "Hard Mode" Implementation Exists:**  
   In the Varman codebase and research history, "Hard Mode" is formally implemented as **"Strong Mode"** (`mode="strong"`, `StrongProtectionEngine` in `app/protection/strong_engine.py`) using GPU-accelerated 25-step EOT-PGD adversarial perturbation against dual surrogates (ArcFace IResNet50 + FaceNet InceptionResnetV1) with differentiable JPEG (`DiffJPEG`).
2. **Why It Was Never Integrated into Live Production:**  
   Hard/Strong Mode was left as an orphaned research asset because of **four compounding integration breaks**:
   - **Frontend Disconnection:** `Varman-frontend/src/pages/UploadPage.jsx` has no UI controls for mode selection (`fast` vs `strong`) and only presents the three fast CPU presets (`Light`, `Balanced`, `Aggressive`). It never transmits a `mode` parameter.
   - **API & Pipeline Severance:** The API route (`app/routes/images.py`), request schema (`app/routes/schemas.py`), and orchestrator (`app/protection/engine.py:protect_image_pipeline`) do not accept, validate, or forward the `mode` parameter.
   - **Internal Signature Crash (Bug in Existing Class):** `StrongProtectionEngine.PRESETS["F5"]` defines parameters (`mask_scale=1.15`, `w_fn=1.0`, `feather_sigma=4.0`), but `StrongProtectionEngine.protect_image()` was never updated to accept them. Passing the verified F5 configuration immediately throws `TypeError: StrongProtectionEngine.protect_image() got an unexpected keyword argument 'mask_scale'`.
   - **Deployment Infrastructure Incompatibility:** The live production deployment on Hugging Face Spaces (`ShadowKagenou/Varman`) is a free CPU-only tier (2 vCPU, 16 GB RAM). Its Dockerfile installs CPU-only PyTorch and does not package the surrogate weights (`arcface_r50_backbone.pth` [166.59 MB] and FaceNet checkpoint [106.71 MB]). On this container, any invocation of `StrongProtectionEngine` triggers an immediate fallback to fast CPU mode.

---

## Phase 1 — Inventory of Existing Implementation

### 1.1 Conceptual & Semantic Mapping
In developer and user communications, the terms "Hard Mode" and "Strong Mode" refer to the same adversarial defense layer:

| User / Developer Term | Production Code Symbol | Location | Core Technical Mechanism |
|---|---|---|---|
| **Hard Mode / Strong Mode** | `StrongProtectionEngine` | `app/protection/strong_engine.py` | 25-step EOT-PGD ($\mu=0.9$ MI-FGSM) on canonical 112×112 face crops |
| **Production Preset F5** | `STRONG_PRESETS["F5"]` | `app/protection/protection_engine.py` | $\varepsilon=8/255$, $\text{mask\_scale}=1.15$, $w_{fn}=1.0$, Pure Luminance |
| **Fast Mode (Default)** | `ProtectionEngine._protect_fast` | `app/protection/protection_engine.py` | YuNet 5-point landmark warping + 8×8 block Y-DCT perturbation |

### 1.2 Core Implementation Files
1. **`app/protection/strong_engine.py` (`StrongProtectionEngine`):**
   - Lines 116–129: Defines `PRESETS["F5"]`.
   - Lines 144–294: `protect_face_crop()` — Executes EOT-PGD using `DiffJPEG` quality sampling ($Q \in [50, 95]$), luminance-only projection (`chroma_scale=0.0`), and optional CIELAB penalty.
   - Lines 295–386: `protect_face_crop_adaptive()` — Bisection search over perturbation scale $s \in [0, 1]$ targeting perceptual CIEDE2000 thresholds.
   - Lines 388–469: `protect_image()` — Detects landmarks, aligns faces to canonical 112×112, calls crop perturbation, and inverse-composites back to original resolution.
2. **`app/protection/diff_jpeg.py` (`DiffJPEG`):**
   - Standalone differentiable JPEG approximation layer supporting batch computation, forward/backward YCbCr color transforms, 8×8 block DCT/IDCT, and Straight-Through Estimator (STE) quantization.
3. **`app/protection/surrogate_models.py` (`DualSurrogateSuite`):**
   - White-box feature extraction backbones: ArcFace (IResNet50, MS1MV3) and FaceNet (InceptionResnetV1, VGGFace2).
4. **`app/protection/face_aligner.py`:**
   - Canonical 5-point landmark partial affine alignment (`align_crop_112`) and soft Gaussian-feathered inverse composite (`composite_crop_back`).
5. **`app/protection/protection_engine.py`:**
   - Lines 47–60: Defines `STRONG_PRESETS["F5"]`.
   - Lines 69–74: Lazy initializer `_get_strong_engine()` avoiding PyTorch imports at global application boot.
   - Lines 269–299: `protect(..., mode="strong")` dispatch logic with CUDA detection and graceful CPU fallback.

### 1.3 Experimental Scripts & Benchmark Codebase
The research and empirical verification were conducted using specialized benchmark harnesses:
- **`benchmarks/run_coverage_frontier.py`:** Authoritative 6-condition coverage frontier (F0–F5) evaluated across 150 LFW genuine pairs ($N=150$) against 4 deep recognition models (ArcFace, FaceNet, SFace, MobileFaceNet) under real saved-and-reloaded JPEG $Q=50$.
- **`benchmarks/run_region_ablation.py`:** Defines `RegionAblationEngine`, implementing paste-aware differentiable PGD, scaled canonical masks (`build_scaled_mask`), and online landmark jitter.
- **`benchmarks/run_paste_aware_experiment.py`:** Proved that modeling the inverse affine transform and mask feathering inside the autograd graph eliminates the 50% post-paste disruption rebound.
- **`benchmarks/run_weight_ratio_followup.py`:** Established that setting FaceNet loss weight $w_{fn}=1.0$ (rather than $0.5$) maximizes multi-architecture transferability.

### 1.4 Git Branch & Commit Provenance
- **Original Development Branches:** `adaptive-strength` and `strong-visibility-sweep`.
- **Merge into Mainline:** Merged into `main` via commit `0354ace` (*"Merge Varman 3.0 research (adaptive-strength) into main"*).
- **Preset Formalization Commit:** Commit `c5a20a2` (*"feat(preset): add F5 named strong preset and config unit test"*), dated October 16, 2026.
- **Current Mainline State:** Fully merged and present on `main` at backend commit `2bdcb9e`.

---

## Phase 2 — Verification of Working Status

### 2.1 Test Suite Evidence
Three test suites in `Varman-backend/tests/` relate directly to Hard/Strong Mode:

| Test File | Test Functions | Execution Target | Local Verification Status |
|---|---|---|:---:|
| `tests/test_f5_preset.py` | `test_f5_preset_config_values` | Dictionary validation | **PASSED (Exit 0)** |
| `tests/test_diff_jpeg.py` | `test_diff_jpeg_vs_pillow_agreement`<br>`test_diff_jpeg_autograd_flow`<br>`test_diff_jpeg_monotonic_quality` | DiffJPEG mathematical invariants | **PASSED (Exit 0)** |
| `tests/test_pgd_convergence.py` | `test_deterministic_execution_under_fixed_seed`<br>`test_gradient_propagation_through_diffjpeg`<br>`test_perturbation_bounds_and_luminance_projection`<br>`test_pgd_optimizer_direction` | GPU EOT-PGD convergence on CUDA | **PASSED (Exit 0)** |

### 2.2 Live Local Hardware Verification
Executed live on local hardware (Local NVIDIA GPU, CUDA 12.x, Python 3.11):
1. **DiffJPEG Agreement with Real libjpeg:** `test_diff_jpeg_vs_pillow_agreement` passed across $Q \in \{50, 75, 95\}$ with $\text{SSIM} \ge 0.990$ and $\text{PSNR} \ge 40.0\text{ dB}$.
2. **Finite Non-Zero Gradients:** Gradients flowed through DiffJPEG back to input pixels without NaNs or Infs ($\|\nabla_\delta\| > 0$).
3. **Pure Luminance Constraint:** Verified $\delta_R = \delta_G = \delta_B$ with maximum cross-channel difference $< 10^{-7}$ and strict $\ell_\infty \le 8/255$.
4. **Surrogate Optimization Direction:** Proved loss monotonically decreases over 5 iterations under MI-FGSM ($\mu=0.9$).

### 2.3 The Critical Production Gap (Verified Bug in `strong_engine.py`)
While the mathematical modules work in isolation, **`StrongProtectionEngine.protect_image()` cannot execute its own preset**:
```python
# Executed via live Python interpreter:
from app.protection.strong_engine import StrongProtectionEngine
eng = StrongProtectionEngine()
eng.protect_image(img, faces, **eng.PRESETS['F5'])

# RESULT:
TypeError: StrongProtectionEngine.protect_image() got an unexpected keyword argument 'mask_scale'
```

#### Why did this happen?
- In `benchmarks/run_region_ablation.py`, the research engine was written as `RegionAblationEngine`, which accepted `mask_scale`, `w_fn`, and `jitter_params`.
- When commit `c5a20a2` introduced `PRESETS["F5"]` to `StrongProtectionEngine`, the author added the dictionary configuration and wrote `tests/test_f5_preset.py`.
- `tests/test_f5_preset.py` **only asserted dictionary key equality**. It never called `protect_image(..., **PRESETS["F5"])`.
- Furthermore, `StrongProtectionEngine.protect_face_crop()` has the FaceNet surrogate weight hardcoded to `0.5` (`loss = sim_af + 0.5 * sim_fn`), whereas the authoritative F5 preset requires $w_{fn} = 1.0$.

---

## Phase 3 — End-to-End Production Pipeline Trace

Tracing the live application path reveals where Hard/Strong Mode was dropped:

```
[User Browser]
      │
      ▼
1. UploadPage.jsx
   ├─ UI State: presetName in ['Light', 'Balanced', 'Aggressive']
   ├─ Mode Selector: MISSING (No toggle for Fast vs Strong/Hard)
   ├─ FormData: {'files': file, 'preset_name': presetName, 'enable_grid': ...}
   └─ 'mode' parameter: NOT SENT
      │
      ▼
2. API Layer (POST /api/images/protect)
   ├─ Route Handler: app/routes/images.py:protect_images()
   ├─ Form Parameters: protection_strength, preset_name, enable_grid
   ├─ 'mode' parameter: NOT DECLARED in FastAPI signature (Ignored if sent)
   └─ Request Schema: app/routes/schemas.py:ImageUploadParams has no 'mode'
      │
      ▼
3. Database Model (app/models/protected_image.py)
   ├─ Model: ProtectedImage
   ├─ Columns: preset_name (String), protection_strength (Float), ...
   └─ 'mode' Column: NOT PRESENT in database schema
      │
      ▼
4. Asynchronous Task Worker (app/routes/images.py:_run_protection_task)
   ├─ Dispatches: protect_image_pipeline(original_path, protected_path, preset_name, enable_grid)
   └─ 'mode' Parameter: NOT FORWARDED
      │
      ▼
5. Orchestrator Pipeline (app/protection/engine.py:protect_image_pipeline)
   ├─ Function Signature: protect_image_pipeline(..., preset_name="Balanced", enable_grid=False, ...)
   ├─ Calls: protection_eng.protect(img_bgr, preset_name=preset_str, enable_grid=enable_grid)
   └─ 'mode' Parameter: OMITTED (Defaults to mode="fast")
      │
      ▼
6. Protection Dispatcher (app/protection/protection_engine.py:protect)
   ├─ Defaults: mode="fast"
   └─ If preset_name="F5":
        preset = self.PRESETS.get("F5", self.PRESETS["Balanced"])
        --> SILENT FALLBACK TO BALANCED CPU PRESET!
```

### 3.1 Silent Exclusion Analysis
If a client attempts to bypass the frontend and manually sends `preset_name="F5"`:
1. `app/routes/images.py` accepts `preset_name="F5"`.
2. `protect_image_pipeline` capitalizes it to `"F5"`.
3. `protect_image_pipeline` calls `protection_eng.protect(..., preset_name="F5")` with default `mode="fast"`.
4. `_protect_fast` looks up `"F5"` in `self.PRESETS` (`Light`, `Balanced`, `Aggressive`).
5. Because `"F5"` is only in `self.STRONG_PRESETS`, the lookup returns `None` and falls back to `self.PRESETS["Balanced"]`.
6. The image is processed using standard fast CPU landmark warping + DCT perturbation. **The client receives a Balanced fast image, believing it ran F5.**

---

## Phase 4 — Identification of Integration Gaps

### 4.1 Summary of Exact Breakpoints

| Layer | File & Location | Current Behavior | Required Behavior for Hard Mode |
|---|---|---|---|
| **Frontend UI** | `Varman-frontend/src/pages/UploadPage.jsx` | Renders only `Light`, `Balanced`, `Aggressive` preset buttons. | Add a Mode Toggle (`Fast (CPU)` vs `Hard (GPU EOT-PGD)`) or expose `Hard (F5)` preset. |
| **Frontend API Call** | `Varman-frontend/src/pages/UploadPage.jsx:125` | `formData.append('preset_name', ...)` only. | Append `formData.append('mode', mode)` to multipart payload. |
| **API Route** | `Varman-backend/app/routes/images.py:151` | `protect_images` does not define `mode: str = Form(default="fast")`. | Add `mode: str = Form(default="fast")` parameter. |
| **Pydantic Schema** | `Varman-backend/app/routes/schemas.py:10` | `ImageUploadParams` lacks `mode` field. | Add `mode: str = Field(default="fast")`. |
| **Task Dispatcher** | `Varman-backend/app/routes/images.py:54` | `_run_protection_task` does not accept `mode`. | Accept `mode` and pass to `protect_image_pipeline`. |
| **Orchestrator** | `Varman-backend/app/protection/engine.py:38` | `protect_image_pipeline` signature lacks `mode`. | Accept `mode` and pass `mode=mode` to `protection_eng.protect()`. |
| **Strong Engine Class** | `Varman-backend/app/protection/strong_engine.py:388` | `protect_image()` crashes on `mask_scale`, `w_fn`, and `feather_sigma`. | Accept `mask_scale`, `w_fn`, and `feather_sigma`, or pass `**kwargs`. Update `protect_face_crop` to use `w_fn`. |
| **Face Aligner** | `Varman-backend/app/protection/face_aligner.py:46` | `composite_crop_back` hardcodes standard $1.0\times$ ellipse mask radii `(42, 50)`. | Support `mask_scale` parameter in `composite_crop_back` (`rx = int(42 * mask_scale)`). |
| **Deployment Docker** | `Varman-HF-Deploy/Dockerfile` | Installs CPU-only PyTorch; no GPU drivers; no surrogate weights. | Deploy to GPU-enabled space or accept transparent fallback metadata. |

---

## Phase 5 — Deployment & Hardware Constraints

### 5.1 Hugging Face Spaces vs. Local Execution

| Dimension | Local Research Environment | Production Hugging Face Space (`ShadowKagenou/Varman`) |
|---|---|---|
| **Hardware** | Local NVIDIA GPU (4 GB VRAM) | Free CPU Tier (2 vCPU, 16 GB RAM) |
| **CUDA Availability** | `torch.cuda.is_available() == True` | `torch.cuda.is_available() == False` |
| **PyTorch Flavor** | CUDA 12.x wheels with cuDNN | CPU-only wheels (`--extra-index-url .../cpu`) |
| **ArcFace Weights** | Present at `~/.cache/varman/arcface_r50_backbone.pth` (166.59 MB) | **MISSING** (Not copied into container image) |
| **FaceNet Weights** | Present at `~/.cache/torch/checkpoints/` (106.71 MB) | **MISSING** (Not pre-cached in container) |
| **Strong Mode Execution** | Runs in ~1.9s per image | **Impossible** (Will trigger `fallback_to_fast=True`) |

### 5.2 Architectural Decision Invariant (Monograph Page 20)
Varman's design deliberately rejects silent auto-routing:
> *"Decision 1 — Fast CPU vs. Strong GPU (No silent auto-routing): If a client requests strong mode and CUDA is unavailable, the system explicitly returns fallback metadata flags (`requested_mode="strong"`, `fallback_to_fast=True`, `fallback_reason="..."`)."*

In the current live deployment, because Hugging Face Spaces operates on free CPU instances, **Strong/Hard Mode cannot execute on the live container**. Even if the API pipeline were fully wired today, every cloud request would transparently return a CPU fallback response.

---

## Proposed Minimal Implementation Plan (When Approved)

*Note: In accordance with project constraints, no changes will be implemented until this audit is formally reviewed.*

### Step 1: Repair `StrongProtectionEngine` Signature & Weights (`strong_engine.py`, `face_aligner.py`)
1. In `app/protection/face_aligner.py`, update `composite_crop_back` to accept `mask_scale: float = 1.0` and scale ellipse radii: `(int(round(42 * mask_scale)), int(round(50 * mask_scale)))`.
2. In `app/protection/strong_engine.py`:
   - Update `protect_face_crop` to accept `w_fn: float = 1.0` and set `loss = sim_af + w_fn * sim_fn`.
   - Update `protect_image` signature to accept `mask_scale: float = 1.0`, `w_fn: float = 1.0`, `feather_sigma: float = 4.0`, and `**kwargs`.
   - Pass `mask_scale` and `feather_sigma` to `composite_crop_back`.

### Step 2: Wire `mode` into Orchestrator & API Route (`engine.py`, `images.py`, `schemas.py`)
1. In `app/protection/engine.py:protect_image_pipeline`:
   - Add `mode: str = "fast"` parameter.
   - If `mode == "strong"` and `preset_name == "Balanced"` (or preset not in `STRONG_PRESETS`), default preset to `"F5"`.
   - Pass `mode=mode` to `protection_eng.protect()`.
2. In `app/routes/images.py`:
   - Add `mode: str = Form(default="fast")` to `protect_images()`.
   - Forward `mode` into `_run_protection_task` and `protect_image_pipeline`.
3. In `app/routes/schemas.py`:
   - Add `mode: str = Field(default="fast", description="fast (CPU) or strong (GPU EOT-PGD)")` to `ImageUploadParams`.
   - Add `mode: str = "fast"` and `fallback_to_fast: bool = False` to `ImageResponse` and `ImageStatusResponse`.

### Step 3: Frontend Exposure (`UploadPage.jsx`)
1. Add an intuitive Mode Toggle in the Disruption Strength Card:
   - **Fast Mode (CPU):** Sub-second aesthetic landmark warping + DCT filter. Presets: `Light`, `Balanced`, `Aggressive`.
   - **Hard Mode (Adversarial GPU):** 25-step EOT-PGD optimization. Preset: `F5 (Frontier Production)`.
2. Append `formData.append('mode', selectedMode)` in `handleProtect()`.
3. Display clear badge if backend returns `fallback_to_fast: true` (e.g. *"Processed in Fast Mode (Server GPU unavailable)"*).

### Step 4: Regression Tests
1. Add `tests/test_strong_mode_pipeline.py`:
   - Test `protect_image_pipeline(..., mode="strong", preset_name="F5")` end-to-end on synthetic image.
   - Test API route `POST /api/images/protect` with `mode="strong"`.
   - Test that when CUDA is absent, `fallback_to_fast=True` metadata is returned cleanly without 500 errors.

---

## Conclusion

The Hard Mode implementation was developed, tested in mathematical isolation, and exhaustively benchmarked during the Varman 3.0 research phase. However, it was never integrated into the live application because:
1. The project prioritized a lightweight CPU architecture for zero-cost deployment on Hugging Face Spaces.
2. The partial preset integration in commit `c5a20a2` was shallow and never wired into the pipeline.
3. The production `StrongProtectionEngine` contained an unexercised keyword argument mismatch that prevented it from running preset F5.

Once reviewed, the proposed minimal changes above will safely and cleanly bridge Hard Mode into both the local development server and cloud deployment environments.
