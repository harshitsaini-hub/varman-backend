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

    Lifecycle:  pending → processing → completed | failed

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
    original_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    protected_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    original_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    protected_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Protection parameters ──────────────────────────────────────────
    protection_strength: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5
    )
    epsilon_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    eot_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    watermark_id: Mapped[str] = mapped_column(String(36), nullable=False, default="")
    watermark_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    vault_sealed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Quality metrics (filled on completion) ─────────────────────────
    ssim_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    psnr_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Processing state ───────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | processing | completed | failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Timestamps ─────────────────────────────────────────────────────
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # ── Relationships ──────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="images")  # noqa: F821

    def __repr__(self) -> str:
        return f"<ProtectedImage {self.id} status={self.status}>"
