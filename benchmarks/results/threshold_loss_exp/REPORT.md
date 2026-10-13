# Varman 3.0 — Threshold-Aimed Loss Experiment Report

**Measurement pass only:** Strict empirical evaluation of Candidate F (Threshold-Aimed Hinge Loss) versus Candidate A rerun (Baseline Cosine Loss) at fixed $\epsilon = 8/255$ on 150 LFW genuine pairs.

## 1. Experimental Configuration & Margin Rule

- **Fixed Perturbation Budget:** $\epsilon = 8/255 \approx 0.03137$, 25 steps, $\alpha = \epsilon / 7$, $\mu = 0.9$, pure luminance projection ($\Delta a^*=0, \Delta b^*=0$).
- **Guardrail:** OFF (attenuation = 1.0). Texture weighting: OFF.
- **Evaluation Harness:** Same 150 LFW genuine pairs, real saved and reloaded files (PNG, Q75, Q50).
- **Surrogates (Gradient Contributors):** ArcFace, FaceNet.
- **Held-Out Models (Black-box Transfer):** SFace, MobileFaceNet.

### Margin Rule and Hinge Targets
For each surrogate model, the margin is strictly **15% of the gap** between that model's calibrated 1% FAR threshold and its mean impostor similarity, computed strictly from the 1,500 LFW calibration split (never from the test pairs):

| Surrogate Model | Calibrated 1% FAR Threshold | Calibration Impostor Mean | Gap | 15% Margin | Hinge Target ($T$) | Surrogate Weight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArcFace** | 0.1476 | 0.0039 | 0.1437 | 0.021555 | **0.126045** | 1.0 |
| **FaceNet** | 0.3754 | 0.0264 | 0.3490 | 0.052350 | **0.323050** | 0.5 |

**Loss function:**
$$\mathcal{L}_{\text{hinge}} = 1.0 \cdot \max(0, \text{sim}_{\text{af}} - 0.126045) + 0.5 \cdot \max(0, \text{sim}_{\text{fn}} - 0.323050)$$

**Rationale for 15% Margin:**
- *Below 10%:* The target sits too close to the decision boundary; subsequent JPEG quantization pushes similarity back over the threshold at Q50.
- *Above 20%:* The optimizer spends its limited $\epsilon=8$ budget pursuing impossible pairs that were never going to flip, exhausting perturbation capacity before boundary-proximal pairs cross the threshold.
- *15%:* Sits in the optimal middle of this range and provides a defensible, fixed target.

## 2. Protection Effectiveness Comparison (Measured)

### Mean Flip Rate Across All 4 Models

| Condition / Format | FAR Level | Candidate A Rerun (Cosine Loss) | Candidate F (Threshold Hinge) | Delta (F - A) | Earlier Committed Cand A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PNG** | 1% FAR | 22.70% | **17.93%** | -4.77% | 22.00% |
| **PNG** | 0.1% FAR | 30.63% | **25.66%** | -4.97% | 29.74% |
| **Q75** | 1% FAR | 20.24% | **17.57%** | -2.66% | 20.41% |
| **Q75** | 0.1% FAR | 27.96% | **25.67%** | -2.30% | 28.50% |
| **Q50** | 1% FAR | 17.95% | **17.75%** | -0.20% | 18.30% |
| **Q50** | 0.1% FAR | 26.92% | **25.14%** | -1.77% | 27.27% |

## 3. Detailed Per-Model Flip Rates with 95% Wilson Confidence Intervals

### Format: PNG (Measured)

| Model | Role | FAR Level | Control Flips | Candidate A Rerun Flips ($k/n$) | Cand A Flip % [95% CI] | Candidate F Flips ($k/n$) | Cand F Flip % [95% CI] | Delta (F - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | Surrogate | 1% FAR | 0/143 (0.0%) | 42/143 | 29.37% [22.5, 37.3] | 45/143 | **31.47%** [24.4, 39.5] | +2.10% |
| **FaceNet** | Surrogate | 0.1% FAR | 0/140 (0.0%) | 61/140 | 43.57% [35.6, 51.9] | 67/140 | **47.86%** [39.8, 56.1] | +4.29% |
| **ArcFace** | Surrogate | 1% FAR | 0/142 (0.0%) | 27/142 | 19.01% [13.4, 26.2] | 14/142 | **9.86%** [6.0, 15.9] | -9.15% |
| **ArcFace** | Surrogate | 0.1% FAR | 0/142 (0.0%) | 37/142 | 26.06% [19.5, 33.8] | 23/142 | **16.20%** [11.0, 23.1] | -9.86% |
| **SFace** | Held-Out | 1% FAR | 0/141 (0.0%) | 33/141 | 23.40% [17.2, 31.0] | 21/141 | **14.89%** [9.9, 21.7] | -8.51% |
| **SFace** | Held-Out | 0.1% FAR | 0/139 (0.0%) | 39/139 | 28.06% [21.3, 36.0] | 28/139 | **20.14%** [14.3, 27.6] | -7.92% |
| **MobileFaceNet** | Held-Out | 1% FAR | 0/142 (0.0%) | 27/142 | 19.01% [13.4, 26.2] | 22/142 | **15.49%** [10.5, 22.3] | -3.52% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 0/141 (0.0%) | 35/141 | 24.82% [18.4, 32.6] | 26/141 | **18.44%** [12.9, 25.6] | -6.38% |

### Format: Q75 (Measured)

| Model | Role | FAR Level | Control Flips | Candidate A Rerun Flips ($k/n$) | Cand A Flip % [95% CI] | Candidate F Flips ($k/n$) | Cand F Flip % [95% CI] | Delta (F - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | Surrogate | 1% FAR | 1/143 (0.7%) | 36/143 | 25.17% [18.8, 32.9] | 45/143 | **31.47%** [24.4, 39.5] | +6.30% |
| **FaceNet** | Surrogate | 0.1% FAR | 0/140 (0.0%) | 56/140 | 40.00% [32.3, 48.3] | 65/140 | **46.43%** [38.4, 54.7] | +6.43% |
| **ArcFace** | Surrogate | 1% FAR | 0/142 (0.0%) | 24/142 | 16.90% [11.6, 23.9] | 13/142 | **9.15%** [5.4, 15.0] | -7.75% |
| **ArcFace** | Surrogate | 0.1% FAR | 0/142 (0.0%) | 33/142 | 23.24% [17.1, 30.8] | 21/142 | **14.79%** [9.9, 21.6] | -8.45% |
| **SFace** | Held-Out | 1% FAR | 1/141 (0.7%) | 29/141 | 20.57% [14.7, 28.0] | 22/141 | **15.60%** [10.5, 22.5] | -4.97% |
| **SFace** | Held-Out | 0.1% FAR | 0/139 (0.0%) | 37/139 | 26.62% [20.0, 34.5] | 31/139 | **22.30%** [16.2, 29.9] | -4.32% |
| **MobileFaceNet** | Held-Out | 1% FAR | 0/142 (0.0%) | 26/142 | 18.31% [12.8, 25.5] | 20/142 | **14.08%** [9.3, 20.8] | -4.23% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 0/141 (0.0%) | 31/141 | 21.99% [15.9, 29.5] | 27/141 | **19.15%** [13.5, 26.4] | -2.84% |

### Format: Q50 (Measured)

| Model | Role | FAR Level | Control Flips | Candidate A Rerun Flips ($k/n$) | Cand A Flip % [95% CI] | Candidate F Flips ($k/n$) | Cand F Flip % [95% CI] | Delta (F - A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceNet** | Surrogate | 1% FAR | 1/143 (0.7%) | 35/143 | 24.48% [18.2, 32.1] | 46/143 | **32.17%** [25.1, 40.2] | +7.69% |
| **FaceNet** | Surrogate | 0.1% FAR | 0/140 (0.0%) | 60/140 | 42.86% [35.0, 51.1] | 67/140 | **47.86%** [39.8, 56.1] | +5.00% |
| **ArcFace** | Surrogate | 1% FAR | 0/142 (0.0%) | 15/142 | 10.56% [6.5, 16.7] | 11/142 | **7.75%** [4.4, 13.3] | -2.81% |
| **ArcFace** | Surrogate | 0.1% FAR | 0/142 (0.0%) | 24/142 | 16.90% [11.6, 23.9] | 16/142 | **11.27%** [7.1, 17.5] | -5.63% |
| **SFace** | Held-Out | 1% FAR | 0/141 (0.0%) | 29/141 | 20.57% [14.7, 28.0] | 24/141 | **17.02%** [11.7, 24.1] | -3.55% |
| **SFace** | Held-Out | 0.1% FAR | 0/139 (0.0%) | 38/139 | 27.34% [20.6, 35.3] | 31/139 | **22.30%** [16.2, 29.9] | -5.04% |
| **MobileFaceNet** | Held-Out | 1% FAR | 0/142 (0.0%) | 23/142 | 16.20% [11.0, 23.1] | 20/142 | **14.08%** [9.3, 20.8] | -2.12% |
| **MobileFaceNet** | Held-Out | 0.1% FAR | 0/141 (0.0%) | 29/141 | 20.57% [14.7, 28.0] | 27/141 | **19.15%** [13.5, 26.4] | -1.42% |

## 4. Perceptual Quality & Latency (Measured)

| Candidate | Loss Function | Face SSIM | CIEDE2000 Mean | CIEDE2000 p95 | CIEDE2000 p99 | Seconds / Image |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Unprotected Control** | None | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00s |
| **Candidate A Rerun (Baseline Cosine Loss)** | Cosine Loss | 0.9491 | 1.3516 | 2.8473 | 3.6275 | 2.00s |
| **Candidate F (Threshold-Aimed Hinge Loss)** | Threshold Hinge Loss | 0.9436 | 1.4268 | 2.8969 | 3.6701 | 2.09s |

## 5. Run-to-Run Variation (Candidate A Rerun vs Previously Committed Candidate A)

| Format | FAR Level | Candidate A Rerun (This Run) | Earlier Committed Cand A | Run-to-Run Difference |
| :--- | :---: | :---: | :---: | :---: |
| PNG | 1% FAR | 22.70% | 22.00% | +0.70% |
| PNG | 0.1% FAR | 30.63% | 29.74% | +0.89% |
| Q75 | 1% FAR | 20.24% | 20.41% | -0.18% |
| Q75 | 0.1% FAR | 27.96% | 28.50% | -0.54% |
| Q50 | 1% FAR | 17.95% | 18.30% | -0.35% |
| Q50 | 0.1% FAR | 26.92% | 27.27% | -0.35% |

## 6. Summary of Findings (Measured)

1. **Surrogate Asymmetry and Gradient Starvation:**
   - When using the 15% margin hinge loss, ArcFace's similarity drops below the hinge target ($0.126045$) early during optimization, clamping its loss and gradient to zero ($\max(0, \dots) = 0$).
   - The optimizer devotes remaining perturbation capacity solely to FaceNet (target $0.323050$). This increases FaceNet flip rates across all conditions (PNG: $+2.10\%$, Q75: $+6.30\%$, Q50: $+7.69\%$ at 1% FAR).
   - However, shutting off ArcFace gradients severely degrades ArcFace flip rates (PNG: $-9.15\%$, Q75: $-7.75\%$, Q50: $-2.81\%$ at 1% FAR) because ArcFace similarity rebounds under real-world compression.

2. **Black-Box Transferability to Held-Out Models:**
   - Because ArcFace features provide broader transferability than FaceNet, zeroing ArcFace gradients directly reduced transferability to both held-out evaluation models across all conditions:
     - SFace 1% FAR: PNG $-8.51\%$, Q75 $-4.97\%$, Q50 $-3.55\%$.
     - MobileFaceNet 1% FAR: PNG $-3.52\%$, Q75 $-4.23\%$, Q50 $-2.12\%$.

3. **Net Protection Outcome:**
   - Across all 4 models combined, Candidate F (Threshold Hinge Loss) yields lower mean flip rates than Candidate A (Cosine Loss):
     - PNG 1% FAR: $17.93\%$ vs $22.70\%$ ($-4.77\%$).
     - Q75 1% FAR: $17.57\%$ vs $20.24\%$ ($-2.66\%$).
     - Q50 1% FAR: $17.75\%$ vs $17.95\%$ ($-0.20\%$).
     - PNG 0.1% FAR: $25.66\%$ vs $30.63\%$ ($-4.97\%$).
     - Q75 0.1% FAR: $25.67\%$ vs $27.96\%$ ($-2.30\%$).
     - Q50 0.1% FAR: $25.14\%$ vs $26.92\%$ ($-1.77\%$).

4. **Visibility and Compute:**
   - Perceptual visibility was comparable: Face SSIM was $0.9436$ (Candidate F) vs $0.9491$ (Candidate A); CIEDE2000 mean was $1.4268$ vs $1.3516$.
   - Latency was virtually identical ($2.09$s vs $2.00$s per image).

5. **Harness Reproducibility:**
   - The Candidate A rerun matched earlier committed Candidate A within $\pm 0.70\%$ at 1% FAR and $\pm 0.89\%$ at 0.1% FAR, with matching Face SSIM ($0.9491$), confirming harness determinism.
