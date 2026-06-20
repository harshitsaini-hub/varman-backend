# Experiment: FaceNet White-Box Surrogate

**Image:** `433f31af-1678-4cb7-b8cc-4d953e825c87_original.jpeg`
**Surrogate Models:** FaceNet (InceptionResnetV1)
**Epsilon:** 8/255 (0.03137)
**Iterations:** 50
**Masking:** MediaPipe Face Mask
**Compression:** DiffJPEG (Q: 65-90)

## Results
- **FaceNet Distance (Cosine):** 0.021166
- **FaceNet Distance (Euclidean):** 0.205747
- **SSIM:** 0.945
- **PSNR:** 37.5 dB
- **Processing Time:** 11.7s

## Conclusion
[FAIL] Identity embedding NOT significantly altered.
Distance dropped slightly compared to the baseline (0.0418 -> 0.0211). 
This proves that the bottleneck is no longer the surrogate model. Even when conducting a pure white-box attack against the exact evaluation model, the severe constraints (8/255 maximum shift, strict facial mask, DiffJPEG compression, and downsampling to 160x160) completely neuter the adversarial gradients. The perturbation is destroyed before it can affect the identity embedding.
