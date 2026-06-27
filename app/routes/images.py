"""Image upload, protection, download, and listing endpoints.

Key design decisions:
  - A single ``asyncio.Semaphore(1)`` gates all PyTorch work so only ONE
    image is ever being processed at a time — critical for the 4 GB VRAM
    limit.
  - The heavy ``protect_image_pipeline`` runs inside a
    ``ThreadPoolExecutor`` via ``loop.run_in_executor`` so it never blocks
    the async event loop (the frontend can still poll ``/status``).
"""

import asyncio
import functools
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.config import settings
from app.database import get_db
from app.models.protected_image import ProtectedImage
from app.models.user import User
from app.protection.engine import protect_image_pipeline
from app.routes.schemas import ImageListResponse, ImageResponse, ImageStatusResponse
from app.uploads import get_image_dimensions, save_upload_with_limits, validate_upload_count

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Concurrency control ───────────────────────────────────────────────────
# Semaphore(1) = only one GPU job at a time.  The executor runs the
# synchronous PyTorch code off the event loop so /status polling still works.
_gpu_semaphore = asyncio.Semaphore(1)
_executor = ThreadPoolExecutor(max_workers=1)


# ── Helpers ────────────────────────────────────────────────────────────────

def _user_storage_dir(user_id: uuid.UUID) -> str:
    """Return (and create) the per-user storage directory."""
    path = os.path.join(settings.storage_dir, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


async def _run_protection_task(
    image_id: uuid.UUID,
    original_path: str,
    protected_path: str,
    strength: float,
) -> None:
    """Acquire the GPU semaphore, run protection off-thread, update the DB."""
    from app.database import async_session  # local to avoid circular

    async with _gpu_semaphore:
        loop = asyncio.get_running_loop()

        # Mark as "processing"
        async with async_session() as db:
            stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
            result = await db.execute(stmt)
            img = result.scalar_one_or_none()
            if img:
                img.status = "processing"  # type: ignore
                await db.commit()

        # Run the heavy pipeline off the event loop
        t0 = time.perf_counter()
        try:
            fn = functools.partial(
                protect_image_pipeline,
                original_path,
                protected_path,
                strength=strength,
            )
            metrics = await loop.run_in_executor(_executor, fn)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.exception("Protection failed for image %s", image_id)
            async with async_session() as db:
                stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
                result = await db.execute(stmt)
                img = result.scalar_one_or_none()
                if img:
                    img.status = "failed"
                    img.error_message = str(exc)[:500]
                    await db.commit()
            return

        # Persist results
        async with async_session() as db:
            stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
            result = await db.execute(stmt)
            img = result.scalar_one_or_none()
            if img:
                img.status = metrics.get("status", "completed") 
                img.ssim_score = metrics.get("ssim")  
                img.psnr_score = metrics.get("psnr") 
                img.epsilon_used = metrics.get("epsilon_used") 
                img.processing_time_ms = elapsed_ms
                if os.path.exists(protected_path):
                    img.protected_path = protected_path 
                    img.protected_size_bytes = os.path.getsize(protected_path) 
                await db.commit()

        logger.info(
            "Protection complete for %s — SSIM=%.3f  PSNR=%.1f  time=%dms",
            image_id,
            metrics.get("ssim", 0),
            metrics.get("psnr", 0),
            elapsed_ms,
        )


# ── Routes ─────────────────────────────────────────────────────────────────


@router.post("/protect", response_model=list[ImageResponse], status_code=status.HTTP_202_ACCEPTED)
async def protect_images(
    files: list[UploadFile],
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    protection_strength: float = Form(default=0.5),
):
    """Accept one or more image uploads, queue them for protection."""
    validate_upload_count(files)

    user_dir = _user_storage_dir(current_user.id)
    created: list[ProtectedImage] = []

    for file in files:
        image_id = uuid.uuid4()
        ext = (file.filename or "upload.jpg").rsplit(".", 1)[-1].lower()
        original_name = f"{image_id}_original.{ext}"
        protected_name = f"{image_id}_protected.png"
        original_path = os.path.join(user_dir, original_name)
        protected_path = os.path.join(user_dir, protected_name)

        # Stream to disk with validation
        await save_upload_with_limits(file, original_path)

        # Get dimensions
        width, height = get_image_dimensions(original_path)
        file_size = os.path.getsize(original_path)

        # Create DB row
        record = ProtectedImage(
            id=image_id,
            user_id=current_user.id,
            original_filename=file.filename or "unknown",
            original_path=original_path,
            protected_path=None,
            width=width,
            height=height,
            original_size_bytes=file_size,
            protection_strength=protection_strength,
            watermark_enabled=False,
            watermark_id="",
            eot_iterations=settings.eot_iterations,
            status="pending",
        )
        db.add(record)
        created.append(record)

    await db.commit()
    for r in created:
        await db.refresh(r)

    # Fire off background tasks (they will queue behind the semaphore)
    for record in created:
        asyncio.create_task(
            _run_protection_task(
                image_id=record.id,
                original_path=record.original_path,
                protected_path=os.path.join(user_dir, f"{record.id}_protected.png"),
                strength=record.protection_strength,
            )
        )

    return created


@router.get("/status/{image_id}", response_model=ImageStatusResponse)
async def get_image_status(
    image_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Poll the processing status of a single image."""
    stmt = select(ProtectedImage).where(
        ProtectedImage.id == image_id,
        ProtectedImage.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    img = result.scalar_one_or_none()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return img


@router.get("/list", response_model=ImageListResponse)
async def list_images(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    page_size: int = 20,
):
    """Paginated list of a user's protected images (newest first)."""
    offset = (max(1, page) - 1) * page_size

    # Total count
    count_stmt = select(sa_func.count()).select_from(ProtectedImage).where(
        ProtectedImage.user_id == current_user.id
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    # Page
    stmt = (
        select(ProtectedImage)
        .where(ProtectedImage.user_id == current_user.id)
        .order_by(ProtectedImage.created_at.desc())  # type: ignore
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    images = list(result.scalars().all())

    return ImageListResponse(images=images, total=total, page=page, page_size=page_size)  # type: ignore


@router.get("/download/{image_id}")
async def download_protected_image(
    image_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Download the protected (adversarial) version of an image."""
    stmt = select(ProtectedImage).where(
        ProtectedImage.id == image_id,
        ProtectedImage.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    img = result.scalar_one_or_none()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if img.status not in ["completed", "failed"]:
        raise HTTPException(status_code=409, detail=f"Image is still '{img.status}'")
    if not img.protected_path or not os.path.exists(img.protected_path):
        raise HTTPException(status_code=404, detail="Protected file missing from disk")

    # Derive the correct filename with .png extension
    base_name = os.path.splitext(img.original_filename)[0]
    return FileResponse(
        path=img.protected_path,
        filename=f"varman_{base_name}.png",
        media_type="image/png",
    )


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image(
    image_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Delete an image record and its files from disk."""
    stmt = select(ProtectedImage).where(
        ProtectedImage.id == image_id,
        ProtectedImage.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    img = result.scalar_one_or_none()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Clean up files
    for path in (img.original_path, img.protected_path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Failed to delete file: %s", path)

    await db.delete(img)
    await db.commit()
