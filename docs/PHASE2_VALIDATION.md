# Varman Phase 2 Validation

This document records the Phase 2 validation harness and the measurements attempted in this environment.
It does not make claims about defeating any commercial deepfake, face-swap, or social-media compression system.
The measured values are only outputs of the locked Varman evaluation procedure.

## Validation harness

`benchmarks/phase2_validation.py` runs the existing locked pipeline without changing the protection algorithms:

1. Load an explicit YuNet ONNX model path.
2. Load multiple explicit real face image paths, or use `skimage.data.lfw_subset()` as an optional local source of real face crops when no image paths are supplied.
3. Validate that YuNet detects a face before including an image in the metric table.
4. Run Light, Balanced, and Aggressive presets.
5. Report per-preset averages for:
   - Global SSIM
   - Expanded face SSIM
   - PSNR
   - L∞
   - Y-NPR JPEG Q95/Q85/Q75/Q60
   - Y-NPR 0.75× resize + JPEG Q75
   - Landmark shift Δpx
   - Alignment error
6. Report measured timing buckets where available:
   - face detection
   - face warp
   - Y-DCT
   - grid
   - quality guardrail
   - evaluation
   - total latency

## YuNet/model acquisition attempt

The requested YuNet model was attempted from both official public locations:

- OpenCV Zoo GitHub raw URL for `face_detection_yunet_2023mar.onnx`
- Hugging Face `opencv/face_detection_yunet` resolve URL for `face_detection_yunet_2023mar.onnx`

Both attempts were blocked by the execution environment proxy with HTTP CONNECT `403 Forbidden` before any model bytes were downloaded.
Because no valid ONNX file was available on disk and `/weights` is intentionally not used for committed artifacts, the real YuNet validation matrix could not be produced in this run.

## Command result

```text
$ PYTHONPATH=. python benchmarks/phase2_validation.py \
    --yunet-model /tmp/varman_phase2/assets/yunet.onnx \
    --output-dir /tmp/varman_phase2/run \
    --json-output /tmp/varman_phase2/results.json
RuntimeError: YuNet model unavailable or OpenCV FaceDetectorYN missing: /tmp/varman_phase2/assets/yunet.onnx
```

## Measurement table

No real-YuNet measurements are reported here because the model download was blocked and reporting synthetic/fake-detector numbers as YuNet results would violate the scientific-integrity requirements.

```text
Metric                     Light       Balanced    Aggressive
------------------------------------------------------------
Global SSIM                n/a         n/a         n/a
Face SSIM                  n/a         n/a         n/a
PSNR                       n/a         n/a         n/a
L∞                         n/a         n/a         n/a
Y-NPR Q95                  n/a         n/a         n/a
Y-NPR Q85                  n/a         n/a         n/a
Y-NPR Q75                  n/a         n/a         n/a
Y-NPR Q60                  n/a         n/a         n/a
Resize+Q75                 n/a         n/a         n/a
Landmark Δpx               n/a         n/a         n/a
Alignment error            n/a         n/a         n/a
```

## Phase 2 regression coverage added

The automated Phase 2 tests added in `tests/test_phase2_validation.py` cover guardrail failure reporting at the minimum attenuation edge, no-face alignment-null behavior, and zero alignment error for identical 5-point landmark geometry.

## Environment used for local tests

- Python: 3.14.4
- OpenCV: 5.0.0
- Test command: `PYTHONPATH=. pytest -q`
- Result: 22 passed, 62 warnings
