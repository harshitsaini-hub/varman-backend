"""
Unit test for F5 named preset configuration.
Verifies config values match the frontier2_2026-10-15 F5 benchmark artifacts.
"""

from app.protection.strong_engine import StrongProtectionEngine
from app.protection.protection_engine import ProtectionEngine


def test_f5_preset_config_values():
    assert "F5" in StrongProtectionEngine.PRESETS
    assert "F5" in ProtectionEngine.STRONG_PRESETS

    f5_strong = StrongProtectionEngine.PRESETS["F5"]
    f5_prot = ProtectionEngine.STRONG_PRESETS["F5"]

    for cfg in (f5_strong, f5_prot):
        assert abs(cfg["eps"] - (8.0 / 255.0)) < 1e-7
        assert cfg["mask_scale"] == 1.15
        assert cfg["steps"] == 25
        assert abs(cfg["alpha"] - ((8.0 / 255.0) / 7.0)) < 1e-7
        assert cfg["mu"] == 0.9
        assert cfg["w_fn"] == 1.0
        assert cfg["luminance_only"] is True
        assert cfg["chroma_scale"] == 0.0
        assert cfg["enable_guardrail"] is False
        assert cfg["feather_sigma"] == 4.0
