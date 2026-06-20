# Baseline: CLIP + ResNet50

**Image:** `433f31af-1678-4cb7-b8cc-4d953e825c87_original.jpeg`
**Surrogate Models:** CLIP (ViT-B/32) + ResNet50
**Epsilon:** 8/255 (0.03137)
**Iterations:** 50
**Masking:** MediaPipe Face Mask
**Compression:** DiffJPEG (Q: 65-90)

## Results
- **FaceNet Distance (Cosine):** 0.041780
- **FaceNet Distance (Euclidean):** 0.289067
- **SSIM:** 0.944
- **PSNR:** 37.2 dB
- **Processing Time:** 26.2s

## Conclusion
[FAIL] Identity embedding NOT significantly altered.
The perturbation strategy is ineffective against face recognition models because it targets generalized vision features (texture/classification) rather than facial identity geometry.
