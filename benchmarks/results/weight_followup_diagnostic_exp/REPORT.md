# Varman 3.0 — Weight Curve Follow-up and Stage Diagnostic Report

**Measurement pass only:** Strict empirical evaluation of surrogate loss weights ($w_{\text{fn}} \in \{0.50, 1.00, 2.00\}$, ArcFace fixed at 1.0) and 4-stage pipeline leakage diagnostic.

## PART 1: Complete Weight Curve Evaluation (150 LFW Genuine Pairs)

### 1. Configuration & Model Roles
- **Perturbation Budget:** $\epsilon = 8/255 \approx 0.03137$, 25 steps, $\alpha = \epsilon / 7$, $\mu = 0.9$, pure luminance projection ($\Delta a^*=0, \Delta b^*=0$).
- **Guardrail:** OFF. **Texture Weighting:** OFF.
- **White-Box Surrogates:** ArcFace (weight 1.0) and FaceNet (weight $w_{\text{fn}}$).
- **Held-Out Models:** SFace and MobileFaceNet (black-box transfer evaluation only).
- **Claim Threshold:** Strict $\ge 3.0$ percentage points threshold on the 4-model mean.

### 2. Mean Flip Rate Across All 4 Models

| Format | FAR Level | Row 0: Control | Row 1: Cand A Rerun ($w=0.50$) | Row 2: Cand I ($w=1.00$) | Delta (I - A) | Row 3: Cand J ($w=2.00$) | Delta (J - A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PNG** | 1% FAR | 0.00% | 22.00% | **25.50%** | +3.50% | **24.60%** | +2.61% |
| **PNG** | 0.1% FAR | 0.00% | 30.45% | **33.50%** | +3.05% | **33.53%** | +3.07% |
| **Q75** | 1% FAR | 0.35% | 20.41% | **24.61%** | +4.20% | **24.07%** | +3.66% |
| **Q75** | 0.1% FAR | 0.00% | 28.86% | **32.07%** | +3.22% | **32.28%** | +3.42% |
| **Q50** | 1% FAR | 0.17% | 17.25% | **21.28%** | +4.03% | **22.49%** | +5.24% |
| **Q50** | 0.1% FAR | 0.00% | 26.38% | **29.41%** | +3.03% | **28.36%** | +1.98% |

### 3. Detailed Per-Model Breakdown (1% and 0.1% FAR with Wilson 95% CIs)

#### Format: PNG (Measured)

| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand I ($w=1.00$) [95% CI] | Delta (I - A) | Cand J ($w=2.00$) [95% CI] | Delta (J - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | White-Box | 1% FAR | 39/143 (27.27% [20.6, 35.1]) | 59/143 (41.26% [33.5, 49.5]) | +13.99% | 67/143 (46.85% [38.9, 55.0]) | +19.58% |
| **FaceNet** | White-Box | 0.1% FAR | 62/140 (44.29% [36.3, 52.6]) | 78/140 (55.71% [47.4, 63.7]) | +11.42% | 91/140 (65.00% [56.8, 72.4]) | +20.71% |
| **ArcFace** | White-Box | 1% FAR | 29/142 (20.42% [14.6, 27.8]) | 25/142 (17.61% [12.2, 24.7]) | -2.81% | 16/142 (11.27% [7.1, 17.5]) | -9.15% |
| **ArcFace** | White-Box | 0.1% FAR | 36/142 (25.35% [18.9, 33.1]) | 31/142 (21.83% [15.8, 29.3]) | -3.52% | 21/142 (14.79% [9.9, 21.6]) | -10.56% |
| **SFace** | Held-Out | 1% FAR | 31/141 (21.99% [15.9, 29.5]) | 33/141 (23.40% [17.2, 31.0]) | +1.41% | 29/141 (20.57% [14.7, 28.0]) | -1.42% |
| **SFace** | Held-Out | 0.1% FAR | 40/139 (28.78% [21.9, 36.8]) | 44/139 (31.65% [24.5, 39.8]) | +2.87% | 41/139 (29.50% [22.6, 37.5]) | +0.72% |
| **MobileFaceNet** | Held-Out | 1% FAR | 26/142 (18.31% [12.8, 25.5]) | 28/142 (19.72% [14.0, 27.0]) | +1.41% | 28/142 (19.72% [14.0, 27.0]) | +1.41% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 33/141 (23.40% [17.2, 31.0]) | 35/141 (24.82% [18.4, 32.6]) | +1.42% | 35/141 (24.82% [18.4, 32.6]) | +1.42% |

#### Format: Q75 (Measured)

| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand I ($w=1.00$) [95% CI] | Delta (I - A) | Cand J ($w=2.00$) [95% CI] | Delta (J - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | White-Box | 1% FAR | 38/143 (26.57% [20.0, 34.4]) | 58/143 (40.56% [32.9, 48.8]) | +13.99% | 67/143 (46.85% [38.9, 55.0]) | +20.28% |
| **FaceNet** | White-Box | 0.1% FAR | 59/140 (42.14% [34.3, 50.4]) | 78/140 (55.71% [47.4, 63.7]) | +13.57% | 88/140 (62.86% [54.6, 70.4]) | +20.72% |
| **ArcFace** | White-Box | 1% FAR | 24/142 (16.90% [11.6, 23.9]) | 20/142 (14.08% [9.3, 20.8]) | -2.82% | 15/142 (10.56% [6.5, 16.7]) | -6.34% |
| **ArcFace** | White-Box | 0.1% FAR | 33/142 (23.24% [17.1, 30.8]) | 29/142 (20.42% [14.6, 27.8]) | -2.82% | 21/142 (14.79% [9.9, 21.6]) | -8.45% |
| **SFace** | Held-Out | 1% FAR | 28/141 (19.86% [14.1, 27.2]) | 33/141 (23.40% [17.2, 31.0]) | +3.54% | 29/141 (20.57% [14.7, 28.0]) | +0.71% |
| **SFace** | Held-Out | 0.1% FAR | 38/139 (27.34% [20.6, 35.3]) | 38/139 (27.34% [20.6, 35.3]) | +0.00% | 39/139 (28.06% [21.3, 36.0]) | +0.72% |
| **MobileFaceNet** | Held-Out | 1% FAR | 26/142 (18.31% [12.8, 25.5]) | 29/142 (20.42% [14.6, 27.8]) | +2.11% | 26/142 (18.31% [12.8, 25.5]) | +0.00% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 32/141 (22.70% [16.6, 30.3]) | 35/141 (24.82% [18.4, 32.6]) | +2.12% | 33/141 (23.40% [17.2, 31.0]) | +0.70% |

#### Format: Q50 (Measured)

| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand I ($w=1.00$) [95% CI] | Delta (I - A) | Cand J ($w=2.00$) [95% CI] | Delta (J - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | White-Box | 1% FAR | 34/143 (23.78% [17.5, 31.4]) | 52/143 (36.36% [28.9, 44.5]) | +12.58% | 67/143 (46.85% [38.9, 55.0]) | +23.07% |
| **FaceNet** | White-Box | 0.1% FAR | 58/140 (41.43% [33.6, 49.7]) | 71/140 (50.71% [42.5, 58.9]) | +9.28% | 81/140 (57.86% [49.6, 65.7]) | +16.43% |
| **ArcFace** | White-Box | 1% FAR | 17/142 (11.97% [7.6, 18.3]) | 13/142 (9.15% [5.4, 15.0]) | -2.82% | 10/142 (7.04% [3.9, 12.5]) | -4.93% |
| **ArcFace** | White-Box | 0.1% FAR | 23/142 (16.20% [11.0, 23.1]) | 25/142 (17.61% [12.2, 24.7]) | +1.41% | 17/142 (11.97% [7.6, 18.3]) | -4.23% |
| **SFace** | Held-Out | 1% FAR | 28/141 (19.86% [14.1, 27.2]) | 32/141 (22.70% [16.6, 30.3]) | +2.84% | 29/141 (20.57% [14.7, 28.0]) | +0.71% |
| **SFace** | Held-Out | 0.1% FAR | 35/139 (25.18% [18.7, 33.0]) | 38/139 (27.34% [20.6, 35.3]) | +2.16% | 34/139 (24.46% [18.1, 32.2]) | -0.72% |
| **MobileFaceNet** | Held-Out | 1% FAR | 19/142 (13.38% [8.7, 19.9]) | 24/142 (16.90% [11.6, 23.9]) | +3.52% | 22/142 (15.49% [10.5, 22.3]) | +2.11% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 32/141 (22.70% [16.6, 30.3]) | 31/141 (21.99% [15.9, 29.5]) | -0.71% | 27/141 (19.15% [13.5, 26.4]) | -3.55% |

### 4. Visibility & Latency (Measured)

| Candidate Row | FaceNet Weight ($w_{\text{fn}}$) | Face SSIM | CIEDE2000 Mean | CIEDE2000 p95 | CIEDE2000 p99 | Seconds / Image |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Row 0: Unprotected Control** | None | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00s |
| **Row 1: Candidate A Rerun (FaceNet weight 0.50)** | 0.50 | 0.9491 | 1.3521 | 2.8480 | 3.6287 | 2.19s |
| **Row 2: Candidate I (FaceNet weight 1.00)** | 1.00 | 0.9481 | 1.3711 | 2.8741 | 3.6617 | 2.16s |
| **Row 3: Candidate J (FaceNet weight 2.00)** | 2.00 | 0.9476 | 1.3857 | 2.8956 | 3.6903 | 2.17s |

---

## PART 2: Stage Diagnostic (40 LFW Genuine Pairs, Candidate A Settings)

Tracking cosine similarity against the clean reference face across four processing stages:
1. **Stage 1 (Optimizer Crop):** Inside optimizer on the canonical 112x112 crop at final step 25.
2. **Stage 2 (Pasted Float):** After pasting the crop back into the full canvas in float32 via inverse affine warp + Gaussian feathered elliptical mask blending.
3. **Stage 3 (PNG Reload):** After 8-bit uint8 rounding, saving to lossless PNG, and reloading with face detection.
4. **Stage 4 (JPEG Q50):** After saving as real JPEG at Q50 and reloading with face detection.

### Stage-by-Stage Cosine Similarity & Below-Threshold Fraction (1% FAR)

| Stage Description | ArcFace Mean Similarity | ArcFace Below 1% FAR ($\tau=0.1476$) | FaceNet Mean Similarity | FaceNet Below 1% FAR ($\tau=0.3754$) |
| :--- | :---: | :---: | :---: | :---: |
| **Reference: Unprotected Images** | 0.7038 | 0/40 (0.0%) | 0.7681 | 0/40 (0.0%) |
| **Stage 1: Inside Optimizer (112x112 crop)** | -0.2078 | 40/40 (100.0%) | 0.1241 | 35/40 (87.5%) |
| **Stage 2: Pasted into Full Image (Float32)** | -0.0288 | 36/40 (90.0%) | 0.3005 | 24/40 (60.0%) |
| **Stage 3: 8-bit PNG Saved & Reloaded** | 0.0505 | 28/40 (70.0%) | 0.3741 | 19/40 (47.5%) |
| **Stage 4: JPEG Q50 Saved & Reloaded** | 0.1470 | 19/40 (47.5%) | 0.4116 | 17/40 (42.5%) |

### Diagnostic Findings: Where Protection Gets Lost

1. **Stage 1 (Optimization on Crop):**
   - The optimizer successfully disrupts the 112x112 face crop, driving ArcFace similarity down from 0.7038 to -0.2078 (drop of 0.9116), and FaceNet from 0.7681 to 0.1241 (drop of 0.6440).
   - Inside the optimizer, 100.0% of pairs are below the 1% FAR threshold for ArcFace, and 87.5% for FaceNet.

2. **Stage 2 (Pasting Crop into Full Image in Float):**
   - Inverting the affine transformation and blending via Gaussian-feathered mask causes an immediate similarity rebound of **+0.1790** for ArcFace (from -0.2078 to -0.0288) and **+0.1764** for FaceNet (from 0.1241 to 0.3005).
   - Fraction below threshold collapses from 100.0% to 90.0% for ArcFace.

3. **Stage 3 (8-bit Quantization & Face Detection):**
   - Rounding float32 to uint8 and re-detecting face bounding box adds an incremental rebound of **+0.0793** for ArcFace and **+0.0736** for FaceNet.

4. **Stage 4 (JPEG Q50 Compression):**
   - Real JPEG Q50 DCT compression causes an additional similarity rebound of **+0.0965** for ArcFace (from 0.0505 to 0.1470) and **+0.0375** for FaceNet.

5. **Largest Leakage Stage:**
   - For ArcFace, **Stage 1 -> Stage 2 (Compositing / Feathering)** loses the most protection (+0.1790 similarity increase).
   - For FaceNet, **Stage 1 -> Stage 2 (Compositing / Feathering)** loses the most protection (+0.1764 similarity increase).
   - Across both models, the geometric pasting back into the canvas (affine warping interpolation and feathered boundary attenuation) represents the single largest point of protection degradation in the pipeline.
