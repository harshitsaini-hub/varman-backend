"""Analytics endpoints — dashboard stats for the authenticated user."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.models.protected_image import ProtectedImage
from app.models.user import User
from app.routes.schemas import AnalyticsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/summary", response_model=AnalyticsResponse)
async def get_analytics(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Aggregate protection statistics for the logged-in user."""
    base = select(ProtectedImage).where(ProtectedImage.user_id == current_user.id)

    # Total count
    total_stmt = select(sa_func.count()).select_from(base.subquery())
    total_images = (await db.execute(total_stmt)).scalar() or 0

    # Status breakdown
    status_stmt = select(
        ProtectedImage.status, sa_func.count()
    ).where(
        ProtectedImage.user_id == current_user.id
    ).group_by(ProtectedImage.status)

    status_rows = (await db.execute(status_stmt)).all()
    status_map = {row[0]: row[1] for row in status_rows}

    # Quality averages (completed only)
    quality_stmt = select(
        sa_func.avg(ProtectedImage.ssim_score),
        sa_func.avg(ProtectedImage.psnr_score),
        sa_func.coalesce(sa_func.sum(ProtectedImage.processing_time_ms), 0),
        sa_func.coalesce(sa_func.sum(ProtectedImage.original_size_bytes), 0),
        sa_func.coalesce(sa_func.sum(ProtectedImage.protected_size_bytes), 0),
    ).where(
        ProtectedImage.user_id == current_user.id,
        ProtectedImage.status == "completed",
    )
    quality_row = (await db.execute(quality_stmt)).one()

    return AnalyticsResponse(
        total_images=total_images,
        completed_images=status_map.get("completed", 0),
        failed_images=status_map.get("failed", 0),
        pending_images=status_map.get("pending", 0) + status_map.get("processing", 0),
        avg_ssim=round(quality_row[0], 4) if quality_row[0] else None,  # type: ignore
        avg_psnr=round(quality_row[1], 2) if quality_row[1] else None,  # type: ignore
        total_processing_time_ms=quality_row[2],  # type: ignore
        total_original_bytes=quality_row[3],  # type: ignore
        total_protected_bytes=quality_row[4],  # type: ignore
    )
