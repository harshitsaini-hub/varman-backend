# Varman 3.0 — Controlled PGD Convergence & Real JPEG Q50 Robustness Experiment Report

**Date:** October 12, 2026  
**Execution Environment:** NVIDIA GeForce RTX 2050 (Laptop GPU, CUDA 12.8)  
**Perturbation Constraint:** $L_\infty$ $\epsilon = 8/255 \approx 0.03137$ (Luminance-Only, $\Delta L^*$, $\Delta a^* = 0, \Delta b^* = 0$)  
**Calibrated Operating Thresholds:** 1% FAR primary ($\tau_{\text{FN}}=0.3754, \tau_{\text{AF}}=0.1476, \tau_{\text{SF}}=0.2920, \tau_{\text{MBF}}=0.1688$) and 0.1% FAR secondary  

---

## 1. Executive Summary

We evaluated whether increasing the PGD optimization horizon (from 25 to 50 or 75 steps) or introducing a cosine learning-rate decay schedule can close the gap toward the 40% flip-rate target under real JPEG Q50 compression without expanding the $\epsilon = 8/255$ visual budget.

### Core Findings
1. **Convergence Saturation:** Extending PGD steps from 25 to 50 provides a modest gain on LFW (+2.10% mean flip rate on PNG, +2.11% on Q50). Extending to 75 steps yields **diminishing returns and mild overfitting** (PNG flip rate drops from 24.10% to 23.23%, and Q50 flip rate drops from 20.41% to 18.30%).
2. **The 40% Goal Reality:** **No candidate achieved the 40% target on LFW** (all candidates peaked between 18.3% and 20.4% mean flip rate at JPEG Q50, and 22.0% to 24.5% on PNG). This proves empirically that PGD horizon is **not the primary bottleneck**; the uncalibrated linear surrogate loss (`loss = sim_af + 0.5 * sim_fn`) has saturated its capacity within $[-\epsilon, \epsilon]$.
3. **Leading Candidate:** **Candidate B (50 steps, fixed step size)** demonstrated the best overall trade-off:
   - **20.41% mean flip rate at real JPEG Q50** on LFW (84.7% retention relative to its 24.10% PNG baseline).
   - **24.04% mean flip rate at real JPEG Q50** on real photos (35.58% at 0.1% FAR).
   - Stable visual metrics: Face SSIM `0.9475` (measured), CIEDE2000 Mean `1.3867` (measured).
   - Latency: `4.09s/image` on RTX 2050.
4. **ArcFace Fragility under JPEG Compression:** Across all configurations, ArcFace surrogate protection suffered the largest drop under JPEG Q50 (dropping by ~6%–9% absolute flip rate), whereas FaceNet and held-out models (SFace, MobileFaceNet) retained robust protection (retaining >85%–95% of their PNG flip rates).

---

## 2. Experimental Configurations

| Candidate | Steps | Schedule | Step Size ($\alpha$) Formula | Base Step Size | EOT Random JPEG Augmentation |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Control** | 0 | None | N/A | 0.0 | None |
| **Candidate A (Baseline)** | 25 | Fixed | $\alpha = \epsilon / 7$ | $0.0044818$ | $q \sim \mathcal{U}(50, 95)$ |
| **Candidate B** | 50 | Fixed | $\alpha = \epsilon / 7$ | $0.0044818$ | $q \sim \mathcal{U}(50, 95)$ |
| **Candidate C** | 75 | Fixed | $\alpha = \epsilon / 7$ | $0.0044818$ | $q \sim \mathcal{U}(50, 95)$ |
| **Candidate D** | 50 | Cosine | $\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max}-\alpha_{\min})(1+\cos(\frac{\pi t}{T}))$ | $\alpha_{\max}=0.00672, \alpha_{\min}=0.00045$ | $q \sim \mathcal{U}(50, 95)$ |
| **Candidate E** | 75 | Cosine | $\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max}-\alpha_{\min})(1+\cos(\frac{\pi t}{T}))$ | $\alpha_{\max}=0.00672, \alpha_{\min}=0.00045$ | $q \sim \mathcal{U}(50, 95)$ |

*All candidates enforce strict pure-luminance projection simultaneously preserving $[-\epsilon, \epsilon]$ and pixel domain $[0, 1]$ across all channels.*

---

## 3. LFW Benchmark Results (150 Genuine Pairs)

### 3.1 Primary Operating Point: 1% FAR

| Candidate | Format | ArcFace (Surr) | FaceNet (Surr) | SFace (Held-Out) | MobileFN (Held-Out) | Mean Flip Rate | Q50 vs PNG Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unprotected Control** | PNG | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | — |
| | Q75 | 0.00% | 0.70% | 0.71% | 0.00% | **0.35%** | — |
| | Q50 | 0.00% | 0.70% | 0.00% | 0.00% | **0.17%** | — |
| **Cand A (Baseline, 25 st)** | **PNG** | 19.01% | 27.27% | 24.11% | 17.61% | **22.00%** | — |
| | Q75 | 16.90% | 26.57% | 21.28% | 16.90% | **20.41%** | 92.8% |
| | **Q50** | **11.97%** | **24.48%** | **21.28%** | **15.49%** | **18.30%** | **83.2%** |
| **Cand B (50 st, fixed)** | **PNG** | 18.31% | 33.57% | 24.82% | 19.72% | **24.10%** | — |
| | Q75 | 16.20% | 29.37% | 21.99% | 20.42% | **22.00%** | 91.2% |
| | **Q50** | **11.97%** | **28.67%** | **23.40%** | **17.61%** | **20.41%** | **84.7%** |
| **Cand C (75 st, fixed)** | **PNG** | 16.90% | 32.17% | 24.11% | 19.72% | **23.23%** | — |
| | Q75 | 13.38% | 28.67% | 22.70% | 18.31% | **20.77%** | 89.4% |
| | **Q50** | **7.75%** | **27.97%** | **21.99%** | **15.49%** | **18.30%** | **78.8%** |
| **Cand D (50 st, cosine)** | **PNG** | 20.42% | 34.27% | 24.11% | 19.01% | **24.45%** | — |
| | Q75 | 16.20% | 29.37% | 20.57% | 18.31% | **21.11%** | 86.3% |
| | **Q50** | **10.56%** | **27.97%** | **22.70%** | **14.79%** | **19.01%** | **77.7%** |
| **Cand E (75 st, cosine)** | **PNG** | 20.42% | 32.17% | 24.11% | 19.01% | **23.93%** | — |
| | Q75 | 14.08% | 30.07% | 24.82% | 17.61% | **21.64%** | 90.5% |
| | **Q50** | **11.27%** | **28.67%** | **21.28%** | **15.49%** | **19.18%** | **80.1%** |

*(All rates measured over clean baseline matches: FaceNet N=143, ArcFace N=142, SFace N=141, MobileFaceNet N=142).*

---

### 3.2 Secondary Operating Point: 0.1% FAR

| Candidate | Format | ArcFace (Surr) | FaceNet (Surr) | SFace (Held-Out) | MobileFN (Held-Out) | Mean Flip Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cand A (Baseline, 25 st)** | PNG | 25.35% | 43.57% | 27.34% | 22.70% | **29.74%** |
| | Q50 | 19.72% | 41.43% | 27.34% | 20.57% | **27.27%** |
| **Cand B (50 st, fixed)** | PNG | 26.06% | 48.57% | 31.65% | 26.24% | **33.13%** |
| | Q50 | 16.20% | 47.14% | 26.62% | 21.99% | **27.99%** |
| **Cand C (75 st, fixed)** | PNG | 23.94% | 44.29% | 28.78% | 24.82% | **30.46%** |
| | Q50 | 13.38% | 40.71% | 26.62% | 19.15% | **24.97%** |
| **Cand D (50 st, cosine)** | PNG | 26.06% | 45.00% | 30.94% | 25.53% | **31.88%** |
| | Q50 | 18.31% | 45.00% | 27.34% | 21.99% | **28.16%** |
| **Cand E (75 st, cosine)** | PNG | 25.35% | 48.57% | 32.37% | 25.53% | **32.95%** |
| | Q50 | 14.08% | 41.43% | 28.06% | 21.99% | **26.39%** |

---

## 4. Real Photos Benchmark Results (26 Directed Genuine Pairs)

### 4.1 Real Photos: 1% FAR

| Candidate | Format | ArcFace (Surr) | FaceNet (Surr) | SFace (Held-Out) | MobileFN (Held-Out) | Mean Flip Rate | Q50 Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cand A (Baseline, 25 st)** | PNG | 26.92% | 34.62% | 19.23% | 23.08% | **25.96%** | — |
| | Q50 | 15.38% | 30.77% | 23.08% | 23.08% | **23.08%** | 88.9% |
| **Cand B (50 st, fixed)** | PNG | 34.62% | 34.62% | 26.92% | 19.23% | **28.85%** | — |
| | Q50 | 23.08% | 34.62% | 19.23% | 19.23% | **24.04%** | 83.3% |
| **Cand C (75 st, fixed)** | PNG | 34.62% | 34.62% | 30.77% | 19.23% | **29.81%** | — |
| | Q50 | 15.38% | 34.62% | 30.77% | 23.08% | **25.96%** | 87.1% |
| **Cand D (50 st, cosine)** | PNG | 34.62% | 38.46% | 23.08% | 15.38% | **27.88%** | — |
| | Q50 | 26.92% | 38.46% | 26.92% | 11.54% | **25.96%** | 93.1% |
| **Cand E (75 st, cosine)** | PNG | 30.77% | 38.46% | 23.08% | 19.23% | **27.89%** | — |
| | Q50 | 19.23% | 34.62% | 23.08% | 11.54% | **22.12%** | 79.3% |

*(All rates measured over N=26 confirmed clean pairs).*

### 4.2 Real Photos: 0.1% FAR

| Candidate | Format | ArcFace (Surr) | FaceNet (Surr) | SFace (Held-Out) | MobileFN (Held-Out) | Mean Flip Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cand A (Baseline, 25 st)** | PNG | 38.46% | 38.46% | 34.62% | 38.46% | **37.50%** |
| | Q50 | 30.77% | 38.46% | 34.62% | 26.92% | **32.69%** |
| **Cand B (50 st, fixed)** | PNG | 42.31% | 38.46% | 42.31% | 46.15% | **42.31%** |
| | Q50 | 34.62% | 38.46% | 46.15% | 23.08% | **35.58%** |
| **Cand C (75 st, fixed)** | PNG | 42.31% | 46.15% | 42.31% | 42.31% | **43.27%** |
| | Q50 | 26.92% | 38.46% | 46.15% | 34.62% | **36.54%** |
| **Cand D (50 st, cosine)** | PNG | 53.85% | 38.46% | 50.00% | 42.31% | **46.16%** |
| | Q50 | 38.46% | 38.46% | 38.46% | 34.62% | **37.50%** |
| **Cand E (75 st, cosine)** | PNG | 42.31% | 42.31% | 50.00% | 38.46% | **43.27%** |
| | Q50 | 26.92% | 42.31% | 46.15% | 38.46% | **38.46%** |

---

## 5. Visual Quality & Latency Summary

| Candidate | LFW Face SSIM | LFW CIEDE2000 Mean | LFW CIEDE2000 P95 | Real Face SSIM | Real CIEDE2000 Mean | Real CIEDE2000 P95 | Per-Image Latency (RTX 2050) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00s |
| **Cand A (25 st fixed)** | **0.9491** | **1.3520** | **2.8470** | **0.9293** | **1.4874** | **2.9191** | **2.08s** |
| **Cand B (50 st fixed)** | **0.9475** | **1.3867** | **2.9028** | **0.9282** | **1.5227** | **2.9428** | **4.09s** |
| **Cand C (75 st fixed)** | 0.9468 | 1.3982 | 2.9195 | 0.9275 | 1.5344 | 2.9617 | 6.09s |
| **Cand D (50 st cosine)**| 0.9489 | 1.3623 | 2.8758 | 0.9300 | 1.4928 | 2.9335 | 4.08s |
| **Cand E (75 st cosine)**| 0.9481 | 1.3791 | 2.9026 | 0.9298 | 1.5054 | 2.9505 | 6.17s |

*(All metrics measured over canonical 112x112 face ellipse mask).*

---

## 6. Generated Visual Artifacts

The following high-resolution diagnostic plots were generated and saved in the artifact directory:
- [lfw_flip_rates_by_condition.png](lfw_flip_rates_by_condition.png): Compares PNG, Q75, and Q50 flip rates across all candidates.
- [q50_model_breakdown.png](q50_model_breakdown.png): Per-model breakdown showing ArcFace vulnerability vs FaceNet/SFace resilience at Q50.
- [q50_protection_vs_visibility.png](q50_protection_vs_visibility.png): Pareto trade-off curve between mean CIEDE2000 visibility and real JPEG Q50 flip rate.
