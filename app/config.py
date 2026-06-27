import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all Varman configuration.

    Values are loaded from environment variables first, then from a ``.env``
    file sitting next to this module (or in the working directory).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir: str = str(Path(__file__).resolve().parent.parent)
    storage_dir: str = ""  # resolved in model_post_init

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./varman.db"

    # ── Auth / JWT ─────────────────────────────────────────────────────────
    jwt_secret: str = ""  # MUST be set via env — startup guard will enforce
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    
    # ── Admin Operator Seed Credentials ────────────────────────────────────
    varman_admin_email: str = ""
    varman_admin_password: str = ""
    varman_admin_display_name: str = "Operator"

    # ── Upload Limits ──────────────────────────────────────────────────────
    max_upload_files: int = 10
    max_upload_bytes: int = 15 * 1024 * 1024  # 15 MB
    allowed_extensions: str = "jpg,jpeg,png,webp"

    # ── Protection Engine ──────────────────────────────────────────────────
    eot_iterations: int = 100
    epsilon_max: float = 0.016  # 4/255 — visually imperceptible L∞ bound
    ssim_min_threshold: float = 0.98
    device: str = "cpu"  # "cpu" or "cuda"

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173"

    # ── Derived helpers (not env-driven) ───────────────────────────────────

    def model_post_init(self, __context) -> None:
        """Resolve defaults that depend on other fields and ensure dirs exist."""
        if not self.storage_dir:
            self.storage_dir = os.path.join(self.base_dir, "storage")

        # Guarantee storage directories exist at import time
        for folder in (self.storage_dir,):
            os.makedirs(folder, exist_ok=True)

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",") if ext.strip()}

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Module-level singleton — import ``settings`` everywhere.
settings = Settings()
