# Varman 3.0 — Paste-Aware Optimization Experiment Report

**Measurement-Only Engineering Benchmark**
- Date: 2026-10-14 19:58:29
- Epsilon: fixed `8/255` (0.03137), Luminance-Only Perturbation
- White-Box Surrogate Models: ArcFace (IResNet50), FaceNet (InceptionResnetV1)
- Strictly Held-Out Evaluation Models: SFace, MobileFaceNet
- LFW Genuine Pairs Evaluated: 150
- Calibration Source: `benchmarks/results/sweep_evidence/calibrated_thresholds_sweep.json`

## 1. Step 2: Equivalence Check (Differentiable Paste vs OpenCV)

| Image Index | Mean Abs Pixel Diff (/255) | Max Abs Pixel Diff (/255) | Status |
|:---:|:---:|:---:|:---:|
| Image 0 | 0.0642 [measured] | 2.0 [measured] | PASS |
| Image 1 | 0.0642 [measured] | 2.0 [measured] | PASS |
| Image 2 | 0.0661 [measured] | 2.0 [measured] | PASS |
| Image 3 | 0.0724 [measured] | 3.0 [measured] | PASS |
| Image 4 | 0.0635 [measured] | 2.0 [measured] | PASS |

**Equivalence Conclusion:** Max mean difference = 0.0724 <= 1.0 threshold. PyTorch differentiable paste-back accurately replicates OpenCV composite math.

## 2. Step 3: Experiment Results Across Conditions

### Format: PNG

| Candidate | 4-Model Mean (1% FAR) | Held-Out Mean (1% FAR) | ArcFace | FaceNet | SFace (Held) | MobileFaceNet (Held) | Face SSIM | CIEDE2000 Mean | Latency (s) | Peak VRAM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 0: Unprotected Control** | **0.00%** [measured] | **0.00%** [measured] | 0/142 (0.0%) | 0/143 (0.0%) | 0/141 (0.0%) | 0/142 (0.0%) | 1.0000 | 0.0000 | 0.0s | 0.399 GB |
| **Row 1: Crop-only, FaceNet weight 1.0** | **25.85%** [measured] | **22.27%** [measured] | 24/142 (16.9%) | 60/143 (41.96%) | 34/141 (24.11%) | 29/142 (20.42%) | 0.9481 | 1.3704 | 2.31s | 0.489 GB |
| **Row 2: Paste-aware, FaceNet weight 1.0** | **33.57%** [measured] | **25.09%** [measured] | 37/142 (26.06%) | 83/143 (58.04%) | 37/141 (26.24%) | 34/142 (23.94%) | 0.9432 | 1.4673 | 1.83s | 0.494 GB |
| **Row 3: Paste-aware + Landmark Jitter** | **40.09%** [measured] | **33.92%** [measured] | 46/142 (32.39%) | 86/143 (60.14%) | 50/141 (35.46%) | 46/142 (32.39%) | 0.9427 | 1.4850 | 1.89s | 0.494 GB |

### Format: Q75

| Candidate | 4-Model Mean (1% FAR) | Held-Out Mean (1% FAR) | ArcFace | FaceNet | SFace (Held) | MobileFaceNet (Held) | Face SSIM | CIEDE2000 Mean | Latency (s) | Peak VRAM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 0: Unprotected Control** | **0.35%** [measured] | **0.35%** [measured] | 0/142 (0.0%) | 1/143 (0.7%) | 1/141 (0.71%) | 0/142 (0.0%) | 1.0000 | 0.0000 | 0.0s | 0.399 GB |
| **Row 1: Crop-only, FaceNet weight 1.0** | **23.38%** [measured] | **20.50%** [measured] | 18/142 (12.68%) | 57/143 (39.86%) | 29/141 (20.57%) | 29/142 (20.42%) | 0.9481 | 1.3704 | 2.31s | 0.489 GB |
| **Row 2: Paste-aware, FaceNet weight 1.0** | **32.16%** [measured] | **24.38%** [measured] | 32/142 (22.54%) | 82/143 (57.34%) | 35/141 (24.82%) | 34/142 (23.94%) | 0.9432 | 1.4673 | 1.83s | 0.494 GB |
| **Row 3: Paste-aware + Landmark Jitter** | **39.57%** [measured] | **32.87%** [measured] | 46/142 (32.39%) | 86/143 (60.14%) | 49/141 (34.75%) | 44/142 (30.99%) | 0.9427 | 1.4850 | 1.89s | 0.494 GB |

### Format: Q50

| Candidate | 4-Model Mean (1% FAR) | Held-Out Mean (1% FAR) | ArcFace | FaceNet | SFace (Held) | MobileFaceNet (Held) | Face SSIM | CIEDE2000 Mean | Latency (s) | Peak VRAM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 0: Unprotected Control** | **0.17%** [measured] | **0.00%** [measured] | 0/142 (0.0%) | 1/143 (0.7%) | 0/141 (0.0%) | 0/142 (0.0%) | 1.0000 | 0.0000 | 0.0s | 0.399 GB |
| **Row 1: Crop-only, FaceNet weight 1.0** | **21.10%** [measured] | **19.45%** [measured] | 13/142 (9.15%) | 52/143 (36.36%) | 31/141 (21.99%) | 24/142 (16.9%) | 0.9481 | 1.3704 | 2.31s | 0.489 GB |
| **Row 2: Paste-aware, FaceNet weight 1.0** | **30.22%** [measured] | **23.68%** [measured] | 24/142 (16.9%) | 81/143 (56.64%) | 35/141 (24.82%) | 32/142 (22.54%) | 0.9432 | 1.4673 | 1.83s | 0.494 GB |
| **Row 3: Paste-aware + Landmark Jitter** | **36.57%** [measured] | **30.74%** [measured] | 37/142 (26.06%) | 84/143 (58.74%) | 46/141 (32.62%) | 41/142 (28.87%) | 0.9427 | 1.4850 | 1.89s | 0.494 GB |

### Full-Image Modification and Canvas Leakage Metrics

| Candidate | Max Abs Pixel Change (Mean) | Max Abs Pixel Change (Max) | Changed Pixels Outside Face Mask (Mean Frac) | Changed Pixels Outside Face Mask (Max Frac) |
|:---|:---:|:---:|:---:|:---:|
| **Row 0: Unprotected Control** | 0.0 [measured] | 0.0 [measured] | 0.000000 [measured] | 0.000000 [measured] |
| **Row 1: Crop-only, FaceNet weight 1.0** | 26.25 [measured] | 53.0 [measured] | 0.003479 [measured] | 0.014156 [measured] |
| **Row 2: Paste-aware, FaceNet weight 1.0** | 26.62 [measured] | 57.0 [measured] | 0.003584 [measured] | 0.014376 [measured] |
| **Row 3: Paste-aware + Landmark Jitter** | 26.51 [measured] | 53.0 [measured] | 0.003593 [measured] | 0.015403 [measured] |

## 3. Paired Comparison: Row 2 (Paste-Aware) vs Row 1 (Crop-Only)

| Format | Model | Total Valid Pairs | Flipped ONLY by Row 2 | Flipped ONLY by Row 1 | Flipped by BOTH | Flipped by NEITHER | Net Gain (Row 2 - Row 1) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PNG | FaceNet | 150 | 28 [measured] | 5 [measured] | 55 [measured] | 62 [measured] | +23 |
| PNG | ArcFace | 150 | 16 [measured] | 3 [measured] | 21 [measured] | 110 [measured] | +13 |
| PNG | SFace | 150 | 8 [measured] | 5 [measured] | 29 [measured] | 108 [measured] | +3 |
| PNG | MobileFaceNet | 150 | 11 [measured] | 6 [measured] | 23 [measured] | 110 [measured] | +5 |
| Q75 | FaceNet | 150 | 30 [measured] | 5 [measured] | 52 [measured] | 63 [measured] | +25 |
| Q75 | ArcFace | 150 | 17 [measured] | 3 [measured] | 15 [measured] | 115 [measured] | +14 |
| Q75 | SFace | 150 | 10 [measured] | 4 [measured] | 25 [measured] | 111 [measured] | +6 |
| Q75 | MobileFaceNet | 150 | 9 [measured] | 4 [measured] | 25 [measured] | 112 [measured] | +5 |
| Q50 | FaceNet | 150 | 32 [measured] | 3 [measured] | 49 [measured] | 66 [measured] | +29 |
| Q50 | ArcFace | 150 | 15 [measured] | 4 [measured] | 9 [measured] | 122 [measured] | +11 |
| Q50 | SFace | 150 | 7 [measured] | 3 [measured] | 28 [measured] | 112 [measured] | +4 |
| Q50 | MobileFaceNet | 150 | 11 [measured] | 3 [measured] | 21 [measured] | 115 [measured] | +8 |

## 4. Step 4: Fixed Stage Diagnostic (40 Genuine Pairs)

Tracks similarity rebounds and fraction below 1% FAR threshold across processing stages:

### Row 1: Crop-Only (w=1.0)

| Stage | ArcFace Mean Sim | ArcFace Below Thresh | FaceNet Mean Sim | FaceNet Below Thresh |
|:---|:---:|:---:|:---:|:---:|
| **Reference (Unprotected)** | 0.7038 [measured] | 0/40 (0.0%) [measured] | 0.7681 [measured] | 0/40 (0.0%) [measured] |
| **Stage 1: Inside Optimizer (112x112 Crop)** | -0.1490 [measured] | 39/40 (97.5%) [measured] | -0.0211 [measured] | 39/40 (97.5%) [measured] |
| **Stage 2: Pasted into Canvas (Float32, No Quant/Save)** | 0.0294 [measured] | 31/40 (77.5%) [measured] | 0.1827 [measured] | 31/40 (77.5%) [measured] |
| **Stage 3a: 8-Bit Quantization (Original Landmarks)** | 0.0304 [measured] | 31/40 (77.5%) [measured] | 0.1833 [measured] | 31/40 (77.5%) [measured] |
| **Stage 3b: PNG Save & Reload (YuNet Re-detection)** | 0.0989 [measured] | 26/40 (65.0%) [measured] | 0.2828 [measured] | 23/40 (57.5%) [measured] |
| **Stage 4: Real JPEG Q50 & Reload (YuNet Re-detection)** | 0.1732 [measured] | 14/40 (35.0%) [measured] | 0.2971 [measured] | 24/40 (60.0%) [measured] |

### Row 2: Paste-Aware (w=1.0)

| Stage | ArcFace Mean Sim | ArcFace Below Thresh | FaceNet Mean Sim | FaceNet Below Thresh |
|:---|:---:|:---:|:---:|:---:|
| **Reference (Unprotected)** | 0.7038 [measured] | 0/40 (0.0%) [measured] | 0.7681 [measured] | 0/40 (0.0%) [measured] |
| **Stage 1: Inside Optimizer (112x112 Crop)** | -0.0666 [measured] | 37/40 (92.5%) [measured] | 0.0692 [measured] | 35/40 (87.5%) [measured] |
| **Stage 2: Pasted into Canvas (Float32, No Quant/Save)** | -0.0378 [measured] | 34/40 (85.0%) [measured] | 0.1146 [measured] | 35/40 (87.5%) [measured] |
| **Stage 3a: 8-Bit Quantization (Original Landmarks)** | -0.0368 [measured] | 34/40 (85.0%) [measured] | 0.1152 [measured] | 35/40 (87.5%) [measured] |
| **Stage 3b: PNG Save & Reload (YuNet Re-detection)** | 0.0425 [measured] | 31/40 (77.5%) [measured] | 0.2105 [measured] | 30/40 (75.0%) [measured] |
| **Stage 4: Real JPEG Q50 & Reload (YuNet Re-detection)** | 0.1006 [measured] | 25/40 (62.5%) [measured] | 0.2477 [measured] | 29/40 (72.5%) [measured] |

### Diagnostic Findings: Loss Decomposition Across Stages

**Row 1 (Crop-Only) Delta Breakdown (Similarity Rebound):**
- Stage 1 -> Stage 2 (Paste & Feathering Rebound): ArcFace **+0.1784** [measured], FaceNet **+0.2038** [measured]
- Stage 2 -> Stage 3a (8-Bit Quantization): ArcFace **+0.0010** [measured], FaceNet **+0.0006** [measured]
- Stage 3a -> Stage 3b (YuNet Detection Jitter): ArcFace **+0.0685** [measured], FaceNet **+0.0995** [measured]
- Stage 3b -> Stage 4 (JPEG Q50 DCT Compression): ArcFace **+0.0743** [measured], FaceNet **+0.0143** [measured]

**Row 2 (Paste-Aware) Delta Breakdown (Similarity Rebound):**
- Stage 1 -> Stage 2 (Paste & Feathering Rebound): ArcFace **+0.0288** [measured], FaceNet **+0.0454** [measured]
- Stage 2 -> Stage 3a (8-Bit Quantization): ArcFace **+0.0010** [measured], FaceNet **+0.0006** [measured]
- Stage 3a -> Stage 3b (YuNet Detection Jitter): ArcFace **+0.0793** [measured], FaceNet **+0.0953** [measured]
- Stage 3b -> Stage 4 (JPEG Q50 DCT Compression): ArcFace **+0.0581** [measured], FaceNet **+0.0372** [measured]

## 5. Decision Rule Evaluation & Step 5 Gated Follow-up

- **Decision Rule Condition 1:** Row 2 beats Row 1 by at least 3.0 percentage points on the 4-model mean at Q50 and 1% FAR.
  - Row 1 Q50 4-model mean: **20.93%** [measured]
  - Row 2 Q50 4-model mean: **30.05%** [measured]
  - Difference (Row 2 - Row 1): **+9.12%** [measured] (Threshold: >= +3.00%)
- **Decision Rule Condition 2:** Held-out mean at Q50 and 1% FAR does not drop.
  - Row 1 Q50 held-out mean: **19.09%** [measured]
  - Row 2 Q50 held-out mean: **24.02%** [measured]
  - Difference (Row 2 - Row 1): **+4.93%** [measured] (Threshold: >= 0.00%)
- **Decision Outcome:** **GO**

**Step 5 Follow-up:** Go rule passed. Row 3 (Paste-aware + Landmark Jitter) was executed and evaluated.
