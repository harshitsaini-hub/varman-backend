# Varman: YuNet Landmark Displacement Controlled Pilot Report (30 LFW Pairs)

**Measurement-Only Engineering Benchmark**

- **Date:** 2026-10-15
- **Target Pairs:** 30 LFW Genuine Pairs (indices 0 to 29) [Measured]
- **Configuration:** Row 3 (Paste-Aware PGD + Online Landmark Jitter)
- **Epsilon:** 8/255 (0.03137) (luminance-only)
- **Hardware:** NVIDIA GeForce RTX 2050 | Peak VRAM: 0.494 GB [Measured] | Avg Latency: 2.03s/img [Measured]
- **Protected Image Location:** `D:\JavaProjects\Varman-research\pilot_landmark_2026-10-15` (Outside repo, not committed)

> [!NOTE]
> **Seeding Notice:** `torch.manual_seed(2026 + pair_index)` and `torch.cuda.manual_seed_all(2026 + pair_index)` were invoked at the start of each pair along with deterministic NumPy EOT quality and landmark jitter RNGs. This ensures input-level determinism, though atomic operations in PyTorch CUDA backward kernels do not guarantee bit-exact floating-point results across differing GPU architectures.

---

## 1. Executive Summary & Pre-Registered Decision

**Pre-Registered Reading Verdict:**
> Reading 3: Protected displacement is statistically larger than clean control (CI excludes zero: [0.4354, 1.5115]) and a meaningful fraction exceeds 2 px aligned-crop space (31.33% > 2px, 43.33% scale > 3%). The current jitter range (+/-2px, +/-3%) may be too narrow, justifying a future wider-jitter experiment [Hypothesis].

| Metric | Clean JPEG Q50 (Control) | Protected Lossless PNG | Protected JPEG Q50 | Paired Difference (Prot Q50 - Clean Q50) | Bootstrap 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Mean Euclidean Drift (Crop 112x112 px)** | 1.003 px [Measured] | 1.7991 px [Measured] | 1.9672 px [Measured] | **+0.9643 px** [Measured] | **[0.4354, 1.5115]** [Recomputed from artifacts] |
| **Median Euclidean Drift (Crop px)** | 0.8564 px [Measured] | 1.2854 px [Measured] | 1.3417 px [Measured] | — | — |
| **P90 Euclidean Drift (Crop px)** | 2.0738 px [Measured] | 3.4947 px [Measured] | 4.1621 px [Measured] | — | — |
| **P95 Euclidean Drift (Crop px)** | 2.8649 px [Measured] | 4.5049 px [Measured] | 4.8112 px [Measured] | — | — |
| **Max Euclidean Drift (Crop px)** | 3.7354 px [Measured] | 7.8871 px [Measured] | 8.5312 px [Measured] | — | — |
| **Mean Euclidean Drift (Full Image px)** | 1.2013 px [Measured] | 2.1515 px [Measured] | 2.3366 px [Measured] | **+1.1354 px** [Measured] | **[0.4943, 1.8115]** [Recomputed from artifacts] |
| **Instances with Drift > 1 px (Crop)** | 38.67% [Measured] | 64.67% [Measured] | 71.33% [Measured] | — | — |
| **Instances with Drift > 2 px (Crop)** | 12.0% [Measured] | 30.67% [Measured] | 31.33% [Measured] | — | — |
| **Pairs with Scale Shift > 3%** | 23.33% [Measured] | 43.33% [Measured] | 43.33% [Measured] | — | — |

---

## 2. Per-Landmark Crop Displacement Breakdown (Aligned 112x112 px)

| Landmark | Clean Q50 Mean [Med, P95] | Prot PNG Mean [Med, P95] | Prot Q50 Mean [Med, P95] |
|:---|:---:|:---:|:---:|
| **Right Eye** | 1.0122 px [0.8465, 2.1081] [Measured] | 1.48 px [1.0445, 3.4903] [Measured] | 1.8795 px [1.5866, 4.0446] [Measured] |
| **Left Eye** | 0.7104 px [0.8265, 1.6407] [Measured] | 1.5415 px [1.0758, 3.9013] [Measured] | 1.8911 px [1.3767, 4.9579] [Measured] |
| **Nose Tip** | 0.9342 px [0.8712, 1.8239] [Measured] | 1.9839 px [1.7161, 3.8528] [Measured] | 1.9463 px [1.3861, 4.2098] [Measured] |
| **Right Mouth Corner** | 1.313 px [0.9661, 3.3331] [Measured] | 2.0519 px [1.3767, 5.61] [Measured] | 2.1259 px [1.3417, 4.7161] [Measured] |
| **Left Mouth Corner** | 1.0451 px [0.8785, 2.3569] [Measured] | 1.9383 px [1.6782, 4.7786] [Measured] | 1.9933 px [1.2178, 5.7729] [Measured] |

---

## 3. Displacement Covariance & Detection Stability

### Displacement Covariance Matrix (Aligned Crop Space)
- **Protected JPEG Q50 $\boldsymbol{\Sigma}_{\Delta x, \Delta y}$:**
  ```
  [[3.9161, 0.4400],
   [0.4400, 2.1989]]
  ```

### Detection Failures & Anomalies Across 30 Pairs
| Condition | No Detection Failures | Multiple Face Detections | Bounding Box Shift > 10% | Alignment Scale Change > 3% |
|:---|:---:|:---:|:---:|:---:|
| **clean_q50** | 0 / 30 [Measured] | 3 / 30 [Measured] | 0 / 30 [Measured] | 7 / 30 [Measured] |
| **prot_png** | 0 / 30 [Measured] | 3 / 30 [Measured] | 0 / 30 [Measured] | 13 / 30 [Measured] |
| **prot_q50** | 0 / 30 [Measured] | 3 / 30 [Measured] | 0 / 30 [Measured] | 13 / 30 [Measured] |

---

## 4. Verification Flip Rates on n=30 Pilot (1% and 0.1% FAR)

> [!WARNING]
> **Statistical Caveat:** Flip rates below reflect the $N=30$ pilot subset only and must NOT be directly compared to the $N=150$ benchmark figure (36.57%) without a paired confidence interval.

| Condition | Model | Role | Mean Sim | 1% FAR Flips (k/n) | Flip Rate (1% FAR) | Wilson 95% CI | 0.1% FAR Flips (k/n) | Flip Rate (0.1% FAR) | Wilson 95% CI |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| clean_control | FaceNet | White-Box | 0.7793 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_control | ArcFace | White-Box | 0.6813 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_control | SFace | Held-Out | 0.7017 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_control | MobileFaceNet | Held-Out | 0.6032 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_q50 | FaceNet | White-Box | 0.775 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_q50 | ArcFace | White-Box | 0.6722 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_q50 | SFace | Held-Out | 0.692 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| clean_q50 | MobileFaceNet | Held-Out | 0.5921 [Measured] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] | 0/30 [Measured] | 0.0% [Measured] | [0.0, 11.35] |
| prot_png | FaceNet | White-Box | 0.3821 [Measured] | 10/30 [Measured] | 33.33% [Measured] | [19.23, 51.22] | 13/30 [Measured] | 43.33% [Measured] | [27.38, 60.8] |
| prot_png | ArcFace | White-Box | 0.237 [Measured] | 7/30 [Measured] | 23.33% [Measured] | [11.79, 40.93] | 11/30 [Measured] | 36.67% [Measured] | [21.87, 54.49] |
| prot_png | SFace | Held-Out | 0.3722 [Measured] | 7/30 [Measured] | 23.33% [Measured] | [11.79, 40.93] | 10/30 [Measured] | 33.33% [Measured] | [19.23, 51.22] |
| prot_png | MobileFaceNet | Held-Out | 0.2721 [Measured] | 8/30 [Measured] | 26.67% [Measured] | [14.18, 44.45] | 9/30 [Measured] | 30.0% [Measured] | [16.66, 47.88] |
| prot_q50 | FaceNet | White-Box | 0.3918 [Measured] | 10/30 [Measured] | 33.33% [Measured] | [19.23, 51.22] | 13/30 [Measured] | 43.33% [Measured] | [27.38, 60.8] |
| prot_q50 | ArcFace | White-Box | 0.2689 [Measured] | 7/30 [Measured] | 23.33% [Measured] | [11.79, 40.93] | 8/30 [Measured] | 26.67% [Measured] | [14.18, 44.45] |
| prot_q50 | SFace | Held-Out | 0.3813 [Measured] | 6/30 [Measured] | 20.0% [Measured] | [9.51, 37.31] | 8/30 [Measured] | 26.67% [Measured] | [14.18, 44.45] |
| prot_q50 | MobileFaceNet | Held-Out | 0.282 [Measured] | 8/30 [Measured] | 26.67% [Measured] | [14.18, 44.45] | 9/30 [Measured] | 30.0% [Measured] | [16.66, 47.88] |

---

## 5. Artifact Preservation Manifest

- **Protected & Control Image Store:** `D:\JavaProjects\Varman-research\pilot_landmark_2026-10-15` (Preserved outside repo)
- **Displacement CSV:** `D:\JavaProjects\Varman\Varman-backend\benchmarks\results\landmark_drift_pilot\landmark_displacements.csv`
- **Per-Pair Flip Records:** `D:\JavaProjects\Varman\Varman-backend\benchmarks\results\landmark_drift_pilot\per_pair_flips.json`
- **Final JSON:** `D:\JavaProjects\Varman\Varman-backend\benchmarks\results\landmark_drift_pilot\landmark_drift_pilot_final.json`
