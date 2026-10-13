"""
Script to create benchmarks/results/pgd_convergence_q50_exp/baseline_results.json
Preserves all original measured values without rounding or truncation.
Explicitly establishes Unprotected Control and Candidate A (25 steps fixed) as the baselines.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np

REPO_ROOT = Path(r"D:\JavaProjects\Varman\Varman-backend")
EXP_DIR = REPO_ROOT / "benchmarks" / "results" / "pgd_convergence_q50_exp"
FINAL_JSON_PATH = EXP_DIR / "pgd_convergence_final.json"
BASELINE_JSON_PATH = EXP_DIR / "baseline_results.json"

with open(FINAL_JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

meta = raw_data["benchmark_metadata"]
cands = raw_data["candidates"]
models = ["ArcFace", "FaceNet", "SFace", "MobileFaceNet"]

def extract_cand_summary(c_key):
    c = cands[c_key]
    out = {
        "candidate_id": c["candidate_id"],
        "candidate_name": c["candidate_name"],
        "steps": c["steps"],
        "schedule": c["schedule"],
        "is_control": c["is_control"],
        "lfw": {},
        "real": {},
        "visibility": c["visibility"],
    }
    for d_name in ["lfw", "real"]:
        out[d_name] = {"formats": {}}
        png_m = np.mean([c[d_name]["formats"]["png"][m]["flips_1pct_pct"] for m in models])
        for fmt in ["png", "q75", "q50"]:
            fmt_data = c[d_name]["formats"][fmt]
            m_flips_1 = [fmt_data[m]["flips_1pct_pct"] for m in models]
            m_flips_01 = [fmt_data[m]["flips_01pct_pct"] for m in models]
            mean_flip_1 = float(np.mean(m_flips_1))
            mean_flip_01 = float(np.mean(m_flips_01))
            retention_vs_png = float(mean_flip_1 / png_m * 100.0) if (png_m > 0 and fmt != "png") else (100.0 if fmt == "png" else 0.0)

            out[d_name]["formats"][fmt] = {
                "mean_flip_rate_1pct": round(mean_flip_1, 4),
                "mean_flip_rate_01pct": round(mean_flip_01, 4),
                "retention_relative_to_png_pct": round(retention_vs_png, 2) if fmt != "png" else None,
                "models": fmt_data,
            }
    return out

baseline_payload = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "metadata": {
        "benchmark_suite": "Varman 3.0 Face Recognition Evaluation Suite",
        "benchmark_version": "3.0.0",
        "created_at": datetime.now().isoformat(),
        "experiment_title": "Controlled PGD Convergence and Real JPEG Q50 Robustness Experiment",
        "reference_report": "benchmarks/results/pgd_convergence_q50_exp/REPORT.md",
        "source_data_file": "benchmarks/results/pgd_convergence_q50_exp/pgd_convergence_final.json",
        "execution_device": meta["device"],
        "gpu_name": meta["gpu_name"],
        "epsilon": meta["epsilon"],
        "perturbation_type": "Luminance-only (zero chromatic perturbation, Delta a* = 0, Delta b* = 0)",
        "surrogate_models": meta["models_surrogate"],
        "held_out_models": meta["models_held_out"],
        "calibrated_thresholds_1pct": meta["thresholds_1pct"],
        "calibrated_thresholds_01pct": meta["thresholds_01pct"],
        "datasets": {
            "lfw_genuine_pairs_evaluated": meta["lfw_pairs_count"],
            "lfw_clean_matched_denominators": {
                "FaceNet": 143,
                "ArcFace": 142,
                "SFace": 141,
                "MobileFaceNet": 142
            },
            "real_photos_genuine_pairs_evaluated": meta["real_pairs_count"],
            "real_photos_clean_matched_denominators": {
                "FaceNet": 26,
                "ArcFace": 26,
                "SFace": 26,
                "MobileFaceNet": 26
            }
        }
    },
    "established_baselines": {
        "unprotected_control": {
            "description": "Unprotected raw images without adversarial noise (0 optimization steps). Serves as negative control to measure baseline degradation from ordinary JPEG compression alone.",
            "data": extract_cand_summary("control")
        },
        "candidate_A_existing_production_baseline": {
            "description": "Existing Varman 3.0 Strong Mode baseline: 25 steps PGD, fixed step size (alpha = eps / 7 = 0.0044818), momentum mu=0.9, EOT random JPEG quality 50-95.",
            "data": extract_cand_summary("cand_A")
        }
    },
    "experimental_alternatives": {
        "status_note": "Candidates B-E are experimental alternatives tested under identical random conditions. They are preserved for comparative benchmarking but are NOT automatically promoted to the production baseline.",
        "candidate_B_50_steps_fixed": extract_cand_summary("cand_B"),
        "candidate_C_75_steps_fixed": extract_cand_summary("cand_C"),
        "candidate_D_50_steps_cosine": extract_cand_summary("cand_D"),
        "candidate_E_75_steps_cosine": extract_cand_summary("cand_E")
    }
}

with open(BASELINE_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(baseline_payload, f, indent=2)

print(f"Successfully generated baseline artifact: {BASELINE_JSON_PATH}")
