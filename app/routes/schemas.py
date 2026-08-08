"""Pydantic schemas for the image and analytics API routes."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


# ── Upload request ─────────────────────────────────────────────────────────

class ImageUploadParams(BaseModel):
    """Optional JSON body alongside the multipart file upload."""
    preset_name: str = Field(default="Balanced", description="Light, Balanced, or Aggressive")
    enable_grid: bool = Field(default=False, description="Enable optional low-intensity spatial grid")


# ── Single image response ─────────────────────────────────────────────────

class ImageResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    status: str
    protection_strength: float = 0.5
    preset_name: str = "Balanced"
    width: int
    height: int
    original_size_bytes: int
    protected_size_bytes: int | None = None
    ssim_score: float | None = None
    face_ssim_score: float | None = None
    psnr_score: float | None = None
    avg_y_npr: float | None = None
    guardrail_satisfied: bool = True
    landmark_shift_delta_px: float | None = None
    alignment_error_pct_change: float | None = None
    processing_time_ms: int | None = None
    error_message: str | None = None
    vault_sealed: bool = False
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Status polling ─────────────────────────────────────────────────────────

class ImageStatusResponse(BaseModel):
    id: uuid.UUID
    status: str
    ssim_score: float | None = None
    face_ssim_score: float | None = None
    psnr_score: float | None = None
    avg_y_npr: float | None = None
    guardrail_satisfied: bool = True
    landmark_shift_delta_px: float | None = None
    alignment_error_pct_change: float | None = None
    processing_time_ms: int | None = None
    error_message: str | None = None
    vault_sealed: bool = False

    model_config = {"from_attributes": True}


# ── Paginated list ─────────────────────────────────────────────────────────

class ImageListResponse(BaseModel):
    images: list[ImageResponse]
    total: int
    page: int
    page_size: int


# ── Analytics ──────────────────────────────────────────────────────────────

class AnalyticsResponse(BaseModel):
    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0
    pending_images: int = 0
    avg_ssim: float | None = None
    avg_psnr: float | None = None
    avg_y_npr: float | None = None
    total_processing_time_ms: int = 0
    total_original_bytes: int = 0
    total_protected_bytes: int = 0
