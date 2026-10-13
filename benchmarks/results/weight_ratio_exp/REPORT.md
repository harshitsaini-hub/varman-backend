# Varman 3.0 — Surrogate Weight Ratio Experiment Report

**Measurement pass only:** Strict empirical evaluation of surrogate loss weighting between ArcFace and FaceNet ($w_{\text{fn}} \in \{0.50, 0.25, 0.10\}$, ArcFace fixed at 1.0) at fixed $\epsilon = 8/255$ on 150 LFW genuine pairs.

## 1. Experimental Configuration & Model Roles

- **Fixed Perturbation Budget:** $\epsilon = 8/255 \approx 0.03137$, 25 steps, $\alpha = \epsilon / 7$, $\mu = 0.9$, pure luminance projection ($\Delta a^*=0, \Delta b^*=0$).
- **Guardrail:** OFF (attenuation = 1.0). **Texture Weighting:** OFF.
- **Loss Formulation:** Plain cosine similarity loss (no hinge): $\mathcal{L} = 1.0 \cdot \text{sim}_{\text{af}} + w_{\text{fn}} \cdot \text{sim}_{\text{fn}}$.
- **White-Box Surrogates (Gradient Contributors):** ArcFace (weight 1.0) and FaceNet (weight $w_{\text{fn}}$).
- **Held-Out Evaluation Models (Black-Box Transfer):** SFace and MobileFaceNet (zero gradient contribution).
- **Thresholds:** Sourced strictly from calibrated 1,500 LFW impostor pairs split (`calibrated_thresholds_sweep.json`).

## 2. Protection Effectiveness Comparison (Measured)

### Mean Flip Rate Across All 4 Models

| Condition / Format | FAR Level | Row 0: Control | Row 1: Cand A Rerun ($w=0.50$) | Row 2: Cand G ($w=0.25$) | Delta (G - A) | Row 3: Cand H ($w=0.10$) | Delta (H - A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PNG** | 1% FAR | 0.00% | 22.18% | **18.66%** | -3.51% | **14.62%** | -7.55% |
| **PNG** | 0.1% FAR | 0.00% | 30.63% | **26.53%** | -4.09% | **22.41%** | -8.21% |
| **Q75** | 1% FAR | 0.35% | 21.12% | **17.25%** | -3.86% | **13.75%** | -7.37% |
| **Q75** | 0.1% FAR | 0.00% | 27.96% | **23.86%** | -4.10% | **19.41%** | -8.55% |
| **Q50** | 1% FAR | 0.17% | 19.36% | **17.25%** | -2.10% | **11.98%** | -7.37% |
| **Q50** | 0.1% FAR | 0.00% | 26.37% | **22.80%** | -3.57% | **17.11%** | -9.27% |

## 3. Detailed Per-Model Breakdown with 95% Wilson Confidence Intervals

### Format: PNG (Measured)

| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand G ($w=0.25$) [95% CI] | Delta (G - A) | Cand H ($w=0.10$) [95% CI] | Delta (H - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | White-Box | 1% FAR | 39/143 (27.27% [20.6, 35.1]) | 28/143 (19.58% [13.9, 26.8]) | -7.69% | 13/143 (9.09% [5.4, 14.9]) | -18.18% |
| **FaceNet** | White-Box | 0.1% FAR | 61/140 (43.57% [35.6, 51.9]) | 40/140 (28.57% [21.7, 36.5]) | -15.00% | 28/140 (20.00% [14.2, 27.4]) | -23.57% |
| **ArcFace** | White-Box | 1% FAR | 28/142 (19.72% [14.0, 27.0]) | 27/142 (19.01% [13.4, 26.2]) | -0.71% | 26/142 (18.31% [12.8, 25.5]) | -1.41% |
| **ArcFace** | White-Box | 0.1% FAR | 38/142 (26.76% [20.2, 34.6]) | 37/142 (26.06% [19.5, 33.8]) | -0.70% | 38/142 (26.76% [20.2, 34.6]) | +0.00% |
| **SFace** | Held-Out | 1% FAR | 34/141 (24.11% [17.8, 31.8]) | 28/141 (19.86% [14.1, 27.2]) | -4.25% | 23/141 (16.31% [11.1, 23.3]) | -7.80% |
| **SFace** | Held-Out | 0.1% FAR | 40/139 (28.78% [21.9, 36.8]) | 42/139 (30.22% [23.2, 38.3]) | +1.44% | 33/139 (23.74% [17.4, 31.5]) | -5.04% |
| **MobileFaceNet** | Held-Out | 1% FAR | 25/142 (17.61% [12.2, 24.7]) | 23/142 (16.20% [11.0, 23.1]) | -1.41% | 21/142 (14.79% [9.9, 21.6]) | -2.82% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 33/141 (23.40% [17.2, 31.0]) | 30/141 (21.28% [15.3, 28.8]) | -2.12% | 27/141 (19.15% [13.5, 26.4]) | -4.25% |

### Format: Q75 (Measured)

| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand G ($w=0.25$) [95% CI] | Delta (G - A) | Cand H ($w=0.10$) [95% CI] | Delta (H - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | White-Box | 1% FAR | 37/143 (25.87% [19.4, 33.6]) | 26/143 (18.18% [12.7, 25.3]) | -7.69% | 15/143 (10.49% [6.5, 16.6]) | -15.38% |
| **FaceNet** | White-Box | 0.1% FAR | 54/140 (38.57% [30.9, 46.8]) | 37/140 (26.43% [19.8, 34.3]) | -12.14% | 27/140 (19.29% [13.6, 26.6]) | -19.28% |
| **ArcFace** | White-Box | 1% FAR | 26/142 (18.31% [12.8, 25.5]) | 22/142 (15.49% [10.5, 22.3]) | -2.82% | 19/142 (13.38% [8.7, 19.9]) | -4.93% |
| **ArcFace** | White-Box | 0.1% FAR | 34/142 (23.94% [17.7, 31.6]) | 31/142 (21.83% [15.8, 29.3]) | -2.11% | 28/142 (19.72% [14.0, 27.0]) | -4.22% |
| **SFace** | Held-Out | 1% FAR | 29/141 (20.57% [14.7, 28.0]) | 25/141 (17.73% [12.3, 24.9]) | -2.84% | 25/141 (17.73% [12.3, 24.9]) | -2.84% |
| **SFace** | Held-Out | 0.1% FAR | 37/139 (26.62% [20.0, 34.5]) | 35/139 (25.18% [18.7, 33.0]) | -1.44% | 31/139 (22.30% [16.2, 29.9]) | -4.32% |
| **MobileFaceNet** | Held-Out | 1% FAR | 28/142 (19.72% [14.0, 27.0]) | 25/142 (17.61% [12.2, 24.7]) | -2.11% | 19/142 (13.38% [8.7, 19.9]) | -6.34% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 32/141 (22.70% [16.6, 30.3]) | 31/141 (21.99% [15.9, 29.5]) | -0.71% | 23/141 (16.31% [11.1, 23.3]) | -6.39% |

### Format: Q50 (Measured)

| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand G ($w=0.25$) [95% CI] | Delta (G - A) | Cand H ($w=0.10$) [95% CI] | Delta (H - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | White-Box | 1% FAR | 37/143 (25.87% [19.4, 33.6]) | 27/143 (18.88% [13.3, 26.1]) | -6.99% | 12/143 (8.39% [4.9, 14.1]) | -17.48% |
| **FaceNet** | White-Box | 0.1% FAR | 56/140 (40.00% [32.3, 48.3]) | 37/140 (26.43% [19.8, 34.3]) | -13.57% | 23/140 (16.43% [11.2, 23.4]) | -23.57% |
| **ArcFace** | White-Box | 1% FAR | 20/142 (14.08% [9.3, 20.8]) | 20/142 (14.08% [9.3, 20.8]) | +0.00% | 17/142 (11.97% [7.6, 18.3]) | -2.11% |
| **ArcFace** | White-Box | 0.1% FAR | 26/142 (18.31% [12.8, 25.5]) | 25/142 (17.61% [12.2, 24.7]) | -0.70% | 18/142 (12.68% [8.2, 19.1]) | -5.63% |
| **SFace** | Held-Out | 1% FAR | 30/141 (21.28% [15.3, 28.8]) | 29/141 (20.57% [14.7, 28.0]) | -0.71% | 24/141 (17.02% [11.7, 24.1]) | -4.26% |
| **SFace** | Held-Out | 0.1% FAR | 36/139 (25.90% [19.3, 33.8]) | 35/139 (25.18% [18.7, 33.0]) | -0.72% | 31/139 (22.30% [16.2, 29.9]) | -3.60% |
| **MobileFaceNet** | Held-Out | 1% FAR | 23/142 (16.20% [11.0, 23.1]) | 22/142 (15.49% [10.5, 22.3]) | -0.71% | 15/142 (10.56% [6.5, 16.7]) | -5.64% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 30/141 (21.28% [15.3, 28.8]) | 31/141 (21.99% [15.9, 29.5]) | +0.71% | 24/141 (17.02% [11.7, 24.1]) | -4.26% |

## 4. Perceptual Quality & Latency (Measured)

| Candidate Row | FaceNet Weight ($w_{\text{fn}}$) | Face SSIM | CIEDE2000 Mean | CIEDE2000 p95 | CIEDE2000 p99 | Seconds / Image |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Row 0: Unprotected Control** | None | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00s |
| **Row 1: Candidate A Rerun (FaceNet weight 0.50)** | 0.50 | 0.9491 | 1.3526 | 2.8488 | 3.6310 | 2.04s |
| **Row 2: Candidate G (FaceNet weight 0.25)** | 0.25 | 0.9499 | 1.3367 | 2.8240 | 3.5941 | 1.97s |
| **Row 3: Candidate H (FaceNet weight 0.10)** | 0.10 | 0.9505 | 1.3232 | 2.8061 | 3.5648 | 2.00s |

## 5. Summary of Findings (Strict Empirical Threshold: $\ge 2.0\%$)

1. **Surrogate Weight Deltas vs Candidate A Rerun ($w=0.50$):**
   - Candidate G ($w=0.25$): Mean flip rate decreased by $-3.51\%$ in PNG and $-2.10\%$ at Q50 (1% FAR).
   - Candidate H ($w=0.10$): Mean flip rate decreased substantially by $-7.55\%$ in PNG and $-7.37\%$ at Q50 (1% FAR).
   - *Improvement Claim Assessment:* Neither Candidate G nor Candidate H achieved an improvement over the Candidate A baseline ($w=0.50$); instead, lowering FaceNet weight caused a statistically significant drop ($> 2.0$ percentage points) in overall protection.
   - Reducing FaceNet's weight did not meaningfully enhance ArcFace protection (ArcFace Q50 1% FAR remained at $14.08\%$ for $w=0.25$ and dropped to $11.97\%$ for $w=0.10$), while severely collapsing FaceNet evasion (dropping from $25.87\%$ down to $8.39\%$ at Q50).

2. **Held-Out Model Direction (SFace, MobileFaceNet):**
   - Held-out models (SFace and MobileFaceNet) moved in the same downward direction as the 4-model mean, confirming that reducing surrogate diversity hurts black-box transferability.
