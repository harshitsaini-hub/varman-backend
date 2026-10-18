"""
Unit tests for Varman benchmark statistical helpers, threshold validation, and pair accounting.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("benchmarks.evaluate_face_models")
from benchmarks.evaluate_face_models import (
    compute_paired_proportion_difference_ci,
    compute_single_proportion_ci,
    generate_energy_matched_noise,
    validate_threshold_cache,
    save_threshold_cache,
)


def test_single_proportion_ci_bounds():
    # 0 out of 100
    low, high = compute_single_proportion_ci(0, 100)
    assert low == 0.0
    assert 0.0 < high < 5.0

    # 100 out of 100
    low, high = compute_single_proportion_ci(100, 100)
    assert 95.0 < low < 100.0
    assert high == 100.0

    # 50 out of 100
    low, high = compute_single_proportion_ci(50, 100)
    assert 39.0 < low < 41.0
    assert 59.0 < high < 61.0

    # Zero trials
    low, high = compute_single_proportion_ci(0, 0)
    assert low == 0.0 and high == 0.0


def test_paired_proportion_difference_identical_outcomes():
    # Both protected and control have 0 flips out of 100
    prot = [False] * 100
    ctrl = [False] * 100
    res = compute_paired_proportion_difference_ci(prot, ctrl)
    assert res["diff_pct"] == 0.0
    assert res["p_value"] == 1.0
    assert res["n_discordant_treatment"] == 0
    assert res["n_discordant_control"] == 0
    # Asymptotic CI covers 0.0 symmetrically
    assert res["ci_lower_pct"] < 0.0
    assert res["ci_upper_pct"] > 0.0
    assert abs(res["ci_lower_pct"] + res["ci_upper_pct"]) < 0.1


def test_paired_proportion_difference_perfect_correlation():
    # Both protected and control flip the exact same 5 pairs out of 100
    prot = [True] * 5 + [False] * 95
    ctrl = [True] * 5 + [False] * 95
    res = compute_paired_proportion_difference_ci(prot, ctrl)
    assert res["diff_pct"] == 0.0
    assert res["p_value"] == 1.0
    assert res["ci_lower_pct"] <= 0.0 <= res["ci_upper_pct"]


def test_paired_proportion_difference_few_discordant_pairs():
    # Treatment flips 2 pairs, Control flips 0 pairs out of 500
    prot = [True] * 2 + [False] * 498
    ctrl = [False] * 500
    res = compute_paired_proportion_difference_ci(prot, ctrl)
    assert pytest.approx(res["diff_pct"], 0.01) == 0.40
    assert res["n_discordant_treatment"] == 2
    assert res["n_discordant_control"] == 0
    assert 0.0 < res["p_value"] <= 1.0
    assert res["ci_lower_pct"] < res["diff_pct"] < res["ci_upper_pct"]


def test_paired_proportion_difference_large_effect():
    # Treatment flips 50 pairs, Control flips 5 pairs out of 100
    prot = [True] * 50 + [False] * 50
    ctrl = [True] * 5 + [False] * 95
    res = compute_paired_proportion_difference_ci(prot, ctrl)
    assert pytest.approx(res["diff_pct"], 0.1) == 45.0
    assert res["p_value"] < 0.001
    assert res["ci_lower_pct"] > 30.0


def test_energy_matched_noise_quantization_tracking():
    rng = np.random.default_rng(42)
    orig = np.full((100, 100, 3), 128, dtype=np.uint8)
    prot = orig.copy()
    prot[:20, :20] = 150  # perturbation

    noisy, stats = generate_energy_matched_noise(orig, prot, rng)
    assert stats["intended_l2_energy"] > 0
    assert stats["realized_l2_energy"] > 0
    # Ratio should be close to 1.0 (within quantization/clipping tolerance)
    assert 0.85 <= stats["energy_match_ratio"] <= 1.15


def test_threshold_cache_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "calibrated_thresholds.json"

        meta = {
            "benchmark_version": "2.0.0",
            "calibration_split": "lfw_train_impostors",
            "target_far": 0.01,
            "model_weights_hashes": {"FaceNet": "vggface2", "ArcFace": "abc123hash"},
            "preprocessing_config": {"arcface_ref": "v1", "facenet_margin": 0.15},
            "thresholds": {"FaceNet": 0.4427, "ArcFace": 0.1361, "SFace": 0.3079},
            "observed_far": {"FaceNet": 0.010, "ArcFace": 0.010, "SFace": 0.010},
        }
        save_threshold_cache(cache_file, meta)

        # Valid load
        loaded = validate_threshold_cache(
            cache_file,
            expected_version="2.0.0",
            expected_split="lfw_train_impostors",
            expected_target_far=0.01,
            expected_hashes={"FaceNet": "vggface2", "ArcFace": "abc123hash"},
            expected_prep={"arcface_ref": "v1", "facenet_margin": 0.15},
        )
        assert loaded is not None
        assert loaded["thresholds"]["FaceNet"] == 0.4427

        # Incompatible split triggers rejection
        incompatible_split = validate_threshold_cache(
            cache_file,
            expected_version="2.0.0",
            expected_split="lfw_test_impostors",  # different split!
            expected_target_far=0.01,
            expected_hashes={"FaceNet": "vggface2", "ArcFace": "abc123hash"},
            expected_prep={"arcface_ref": "v1", "facenet_margin": 0.15},
        )
        assert incompatible_split is None

        # Incompatible hash triggers rejection
        incompatible_hash = validate_threshold_cache(
            cache_file,
            expected_version="2.0.0",
            expected_split="lfw_train_impostors",
            expected_target_far=0.01,
            expected_hashes={"FaceNet": "vggface2", "ArcFace": "DIFFERENT_HASH"},
            expected_prep={"arcface_ref": "v1", "facenet_margin": 0.15},
        )
        assert incompatible_hash is None
