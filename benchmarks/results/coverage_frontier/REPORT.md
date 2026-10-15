# Varman Coverage Frontier Benchmark Results

**Date:** 2026-10-15  
**Pairs:** 150 LFW genuine pairs  
**Hardware:** NVIDIA GeForce RTX 2050  
**Standard Visibility Definition:** Definition B (fixed canonical face mask on re-extracted composite crop)  

## Coverage Frontier Table (Real JPEG Q50, 1% FAR)

| Row ID | Epsilon | Mask Scale | Face SSIM (Def B) | CIEDE Mean | CIEDE p95 | Whole SSIM | Q50 1% FAR 4-Model Mean | Held-Out Mean |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F0_eps8_scale10** | 8/255 | 1.00 | 0.9192 | 1.5037 | 3.0674 | 0.9798 | **36.39%** | **30.39%** |
| **F1_eps8_scale13** | 8/255 | 1.30 | 0.8944 | 1.7722 | 3.2276 | 0.9649 | **49.95%** | **42.41%** |
| **F2_eps7_scale13** | 7/255 | 1.30 | 0.9129 | 1.5952 | 2.9206 | 0.9708 | **37.10%** | **30.39%** |
| **F3_eps6_scale13** | 6/255 | 1.30 | 0.9308 | 1.4138 | 2.6236 | 0.9765 | **26.55%** | **21.57%** |
| **F4_eps5_scale13** | 5/255 | 1.30 | 0.9471 | 1.2378 | 2.3314 | 0.9817 | **13.90%** | **13.43%** |
| **F5_eps8_scale115** | 8/255 | 1.15 | 0.8986 | 1.7418 | 3.1885 | 0.9727 | **45.73%** | **39.23%** |
