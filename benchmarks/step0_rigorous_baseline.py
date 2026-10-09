"""
Varman Step 0 Rigorous Baseline Evaluation Harness
===================================================
Alias entrypoint delegating directly to benchmarks/evaluate_face_models.py.
Incorporates:
  1. Dynamic ONNX GPU provider selection (CUDAExecutionProvider with CPU fallback).
  2. Strict categorical pair accounting:
     {pair_id, is_clean_match, is_protected_flip, is_jpeg_control_flip, is_noise_control_flip}
  3. Strict denominator filtering: flip rates evaluated exclusively over confirmed clean matches.
  4. scipy.stats 95% Confidence Intervals for paired differences in proportions (Protected vs. Controls).
  5. Comprehensive experiment artifact export (meta, summaries, granular per-pair audit trail).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure Varman-backend is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.evaluate_face_models import run_face_recognition_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman Rigorous Baseline Benchmarking Suite")
    parser.add_argument("--pairs", type=int, default=500, help="Number of genuine LFW pairs (default: 500)")
    parser.add_argument("--preset", type=str, default="Balanced", choices=["Light", "Balanced", "Aggressive"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration on impostor pairs")
    parser.add_argument("--max-impostors", type=int, default=500, help="Max impostor pairs for calibration (default: 500)")
    parser.add_argument("--output", type=str, default="benchmarks/step0_baseline_results.json")
    args = parser.parse_args()

    run_face_recognition_benchmark(
        num_pairs=args.pairs,
        preset=args.preset,
        requested_device=args.device,
        seed=args.seed,
        recalibrate=args.recalibrate,
        max_impostors=args.max_impostors,
        output_json=Path(args.output),
    )
