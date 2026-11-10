import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models import Base


class ProtectedImage(Base):
    """A single image that has been (or is being) protected by Varman.

    Lifecycle:  pending → processing → completed | failed | cancelled

    The ``original_path`` and ``protected_path`` are *server-local*
    filesystem paths under ``settings.storage_dir``.
    """

    __tablename__ = "protected_images"

    # ── Identity ───────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # ── File metadata ──────────────────────────────────────────────────
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=True)
    protected_path: Mapped[str] = mapped_column(Text, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    original_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    protected_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)

    # ── Protection parameters ──────────────────────────────────────────
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="fast")  # fast | hard (synced with actual_mode)
    requested_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="fast")
    actual_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="fast")
    fallback_occurred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fallback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_user_accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False, default="4.0.0")
    protection_strength: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5
    )
    preset_name: Mapped[str] = mapped_column(String(20), nullable=False, default="Balanced")
    epsilon_used: Mapped[float] = mapped_column(Float, nullable=True)  # Legacy — kept for DB compat
    eot_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    eot_steps_executed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    eps_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    alpha_step: Mapped[float | None] = mapped_column(Float, nullable=True)
    w_fn: Mapped[float | None] = mapped_column(Float, nullable=True)
    mask_scale: Mapped[float | None] = mapped_column(Float, nullable=True)
    watermark_id: Mapped[str] = mapped_column(String(36), nullable=False, default="")
    watermark_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    vault_sealed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Quality metrics (filled on completion) ─────────────────────────
    ssim_score: Mapped[float] = mapped_column(Float, nullable=True)
    psnr_score: Mapped[float] = mapped_column(Float, nullable=True)
    face_ssim_score: Mapped[float] = mapped_column(Float, nullable=True)
    linf_score: Mapped[float] = mapped_column(Float, nullable=True)
    avg_y_npr: Mapped[float] = mapped_column(Float, nullable=True)
    guardrail_satisfied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attenuation_factor: Mapped[float] = mapped_column(Float, nullable=True)
    landmark_shift_delta_px: Mapped[float] = mapped_column(Float, nullable=True)
    alignment_error_pct_change: Mapped[float] = mapped_column(Float, nullable=True)

    # ── Processing state ───────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | processing | completed | failed | cancelled
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=True)
    processing_duration: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Timestamps ─────────────────────────────────────────────────────
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # ── Relationships ──────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="images")  # noqa: F821

    def __repr__(self) -> str:
        return f"<ProtectedImage {self.id} status={self.status} mode={self.mode}>"
