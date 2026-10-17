# Varman Region-Size Ablation Benchmark Results

**Date:** 2026-10-15  
**Pairs:** 150 LFW genuine pairs  
**Hardware:** NVIDIA GeForce RTX 2050  

## Visibility vs. Protection Frontier Table (Real JPEG Q50, 1% FAR)

| Row ID | Configuration | Face SSIM | CIEDE Mean | CIEDE p95 | Changed Area Frac | 4-Model Mean | Held-Out Mean |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **R0_control** | Row 0: Unprotected Control | 1.0000 | 0.0000 | 0.0000 | 64.5% | **0.17%** | **0.00%** |
| **R1_base_scale10** | Row 1: Baseline (eps=8/255, mask_scale=1.0) | 0.8753 | 2.1061 | 3.0599 | 64.5% | **36.57%** | **30.04%** |
| **R2_scale07** | Row 2: Smaller Mask (eps=8/255, mask_scale=0.7) | 0.9276 | 1.5088 | 3.0111 | 33.9% | **16.00%** | **13.79%** |
| **R3_scale13** | Row 3: Larger Mask (eps=8/255, mask_scale=1.3) | 0.8522 | 2.0649 | 3.0554 | 92.1% | **48.36%** | **40.64%** |
| **R4_eps6_scale10** | Row 4: Lower-Visibility Frontier (eps=6/255, mask_scale=1.0) | 0.9196 | 1.6232 | 2.2993 | 64.5% | **16.18%** | **14.50%** |

## Paired Exact McNemar Tests vs R1 Baseline (JPEG Q50, 1% FAR)

| Candidate vs R1 | Model | R1 Flips | Cand Flips | Discordant (b / c) | Exact p | Holm Adj p | Significant (< 0.05) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **R2_scale07** | FaceNet | 85 | 43 | 42 / 0 | 0.0000 | 0.0000 | **YES** |
| **R2_scale07** | ArcFace | 38 | 9 | 29 / 0 | 0.0000 | 0.0000 | **YES** |
| **R2_scale07** | SFace | 45 | 25 | 20 / 0 | 0.0000 | 0.0000 | **YES** |
| **R2_scale07** | MobileFaceNet | 40 | 14 | 26 / 0 | 0.0000 | 0.0000 | **YES** |
| **R3_scale13** | FaceNet | 85 | 104 | 0 / 19 | 0.0000 | 0.0000 | **YES** |
| **R3_scale13** | ArcFace | 38 | 56 | 2 / 20 | 0.0001 | 0.0002 | **YES** |
| **R3_scale13** | SFace | 45 | 60 | 0 / 15 | 0.0001 | 0.0002 | **YES** |
| **R3_scale13** | MobileFaceNet | 40 | 55 | 0 / 15 | 0.0001 | 0.0002 | **YES** |
| **R4_eps6_scale10** | FaceNet | 85 | 41 | 44 / 0 | 0.0000 | 0.0000 | **YES** |
| **R4_eps6_scale10** | ArcFace | 38 | 10 | 28 / 0 | 0.0000 | 0.0000 | **YES** |
| **R4_eps6_scale10** | SFace | 45 | 25 | 20 / 0 | 0.0000 | 0.0000 | **YES** |
| **R4_eps6_scale10** | MobileFaceNet | 40 | 16 | 24 / 0 | 0.0000 | 0.0000 | **YES** |
