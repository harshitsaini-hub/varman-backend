"""Image upload, protection, download, cancellation, and listing endpoints.

Key architectural & operational guarantees:
  - Strict bounded queue: Concurrency=1 for execution, max 5 pending jobs.
    Returns HTTP 429 Too Many Requests (Retry-After: 30) when saturated.
  - User-scoped idempotency: Deduplicates pending/processing submissions.
  - Cooperative cancellation: POST /api/images/cancel/{image_id} halts PGD safely.
  - Evidence-based per-job timeouts: 45s for GPU Hard Mode, 20s for CPU Fast Mode.
  - Mode dispatch: Fast Mode (CPU) vs Hard Mode (Server GPU).
"""

import asyncio
import functools
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func as sa_func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.config import settings
from app.database import get_db
from app.models.protected_image import ProtectedImage
from app.models.user import User
from app.protection.paste_aware_engine import TaskCancelledException
from app.routes.schemas import ImageListResponse, ImageResponse, ImageStatusResponse
from app.uploads import get_image_dimensions, save_upload_with_limits, validate_upload_count

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Concurrency & Queue Control ───────────────────────────────────────────
MAX_ERROR_MESSAGE_LENGTH = 500
MAX_PENDING_QUEUE = 5

_protection_semaphore = asyncio.Semaphore(1)
_executor = ThreadPoolExecutor(max_workers=1)
_queue_lock = asyncio.Lock()
_pending_jobs_count = 0
_cancel_events: Dict[uuid.UUID, threading.Event] = {}

ALLOWED_FAST_PRESETS = {"light", "balanced", "aggressive"}
ALLOWED_HARD_PRESETS = {"f5"}


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
    preset_name: str = "Balanced",
    enable_grid: bool = False,
    mode: str = "fast",
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Acquire the protection semaphore, run protection off-thread, update the DB."""
    global _pending_jobs_count
    from app.database import async_session

    try:
        async with _protection_semaphore:
            # Task is now transitioning from pending to processing
            async with _queue_lock:
                if _pending_jobs_count > 0:
                    _pending_jobs_count -= 1

            loop = asyncio.get_running_loop()

            # Check if task was already cancelled while in queue
            async with async_session() as db:
                stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
                result = await db.execute(stmt)
                img = result.scalar_one_or_none()
                if not img or img.status == "cancelled":
                    logger.info("Task %s was cancelled before execution.", image_id)
                    return
                img.status = "processing"
                await db.commit()

            # Timeout threshold: 45s for GPU Hard Mode, 20s for CPU Fast Mode
            timeout_sec = 45.0 if mode.lower() == "hard" else 20.0
            t0 = time.perf_counter()

            try:
                from app.protection.engine import protect_image_pipeline

                fn = functools.partial(
                    protect_image_pipeline,
                    original_path,
                    protected_path,
                    preset_name=preset_name,
                    enable_grid=enable_grid,
                    mode=mode,
                    cancel_event=cancel_event,
                )
                out_png_path, combined_metrics = await asyncio.wait_for(
                    loop.run_in_executor(_executor, fn),
                    timeout=timeout_sec,
                )
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                eval_m = combined_metrics.get("evaluation_metrics", {})
            except asyncio.TimeoutError:
                logger.error("Protection timeout exceeded for image %s after %.1fs", image_id, timeout_sec)
                async with async_session() as db:
                    stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
                    result = await db.execute(stmt)
                    img = result.scalar_one_or_none()
                    if img and img.status != "cancelled":
                        img.status = "failed"
                        img.error_message = f"Job processing timeout exceeded ({int(timeout_sec)}s limit)"
                        await db.commit()
                return
            except TaskCancelledException as exc:
                logger.warning("Task %s cancelled cooperatively: %s", image_id, exc)
                async with async_session() as db:
                    stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
                    result = await db.execute(stmt)
                    img = result.scalar_one_or_none()
                    if img:
                        img.status = "cancelled"
                        img.error_message = "Task cancelled by user"
                        await db.commit()
                return
            except Exception as exc:
                logger.exception("Protection failed for image %s", image_id)
                async with async_session() as db:
                    stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
                    result = await db.execute(stmt)
                    img = result.scalar_one_or_none()
                    if img and img.status != "cancelled":
                        img.status = "failed"
                        img.error_message = str(exc)[:MAX_ERROR_MESSAGE_LENGTH]
                        await db.commit()
                return

            # Persist completed results
            async with async_session() as db:
                stmt = select(ProtectedImage).where(ProtectedImage.id == image_id)
                result = await db.execute(stmt)
                img = result.scalar_one_or_none()
                if img and img.status != "cancelled":
                    prot_m = combined_metrics.get("protection_meta", {})
                    eval_m = combined_metrics.get("evaluation_metrics", {})

                    img.status = "completed"
                    img.ssim_score = eval_m.get("global_ssim")
                    img.face_ssim_score = eval_m.get("face_ssim_1_2x")
                    img.psnr_score = eval_m.get("psnr_db")
                    img.linf_score = (
                        float(eval_m.get("l_inf_max_shift", 0))
                        if eval_m.get("l_inf_max_shift") is not None
                        else None
                    ) # pyright: ignore[reportAttributeAccessIssue]
                    img.guardrail_satisfied = bool(prot_m.get("guardrail_satisfied", True))
                    img.attenuation_factor = (
                        float(prot_m.get("attenuation_factor", 1.0))
                        if prot_m.get("attenuation_factor") is not None
                        else None
                    ) # pyright: ignore[reportAttributeAccessIssue]
                    img.avg_y_npr = eval_m.get("avg_y_npr_percent")
                    img.landmark_shift_delta_px = eval_m.get("landmark_shift_delta_px")
                    img.alignment_error_pct_change = eval_m.get("alignment_error_pct_change")
                    img.eot_steps_executed = prot_m.get("eot_steps_executed")
                    img.eps_budget = prot_m.get("eps_budget")
                    img.alpha_step = prot_m.get("alpha_step")
                    img.w_fn = prot_m.get("w_fn")
                    img.mask_scale = prot_m.get("mask_scale")
                    img.processing_time_ms = elapsed_ms
                    img.processing_duration = round(elapsed_ms / 1000.0, 3)
                    img.engine_version = "4.0.0"
                    img.original_path = None # type: ignore
                    if os.path.exists(out_png_path):
                        img.protected_path = out_png_path
                        img.protected_size_bytes = os.path.getsize(out_png_path)
                    await db.commit()

            # Clean up original upload
            if os.path.exists(original_path):
                try:
                    os.remove(original_path)
                    logger.info("[Vault] Deleted original: %s", original_path)
                except OSError:
                    logger.warning("[Vault] Failed to delete original: %s", original_path)

            logger.info("Protection complete for %s (Mode: %s) in %dms", image_id, mode, elapsed_ms)
    finally:
        _cancel_events.pop(image_id, None)


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/protect", response_model=list[ImageResponse], status_code=status.HTTP_202_ACCEPTED)
async def protect_images(
    files: list[UploadFile],
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    mode: str = Form(default="fast"),
    protection_strength: float = Form(default=0.5),
    preset_name: str = Form(default="Balanced"),
    enable_grid: bool = Form(default=False),
    allow_cpu_fallback: bool = Form(default=False),
):
    """Accept one or more image uploads, validate parameters, and queue for protection."""
    global _pending_jobs_count
    validate_upload_count(files)

    clean_mode = str(mode).strip().lower()
    if clean_mode not in ("fast", "hard"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid protection mode '{mode}'. Must be 'fast' or 'hard'.",
        )

    clean_preset = str(preset_name).strip().lower()
    if clean_mode == "fast" and clean_preset not in ALLOWED_FAST_PRESETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid preset '{preset_name}' for Fast Mode. Must be Light, Balanced, or Aggressive.",
        )
    if clean_mode == "hard" and clean_preset not in ALLOWED_HARD_PRESETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid preset '{preset_name}' for Hard Mode. Must be F5.",
        )

    # ── Server-enforced 409 consent for CPU fallback ──────────────────
    requested_mode = clean_mode
    actual_mode = clean_mode
    fallback_occurred = False
    fallback_reason = None
    fallback_user_accepted = False

    if clean_mode == "hard":
        from app.protection.gpu_probe import get_cached_gpu_available
        if not get_cached_gpu_available():
            if not allow_cpu_fallback:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Hard Mode requires a CUDA GPU. Server is CPU-only. Explicit consent (allow_cpu_fallback=true) required to proceed with Fast Mode (CPU).",
                )
            actual_mode = "fast"
            clean_mode = "fast"
            fallback_occurred = True
            fallback_user_accepted = True
            fallback_reason = "Server is CPU-only; user consented to Fast Mode (CPU) fallback."
            preset_name = "Balanced"

    # ── Bounded queue capacity check ──────────────────────────────────
    async with _queue_lock:
        if _pending_jobs_count + len(files) > MAX_PENDING_QUEUE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Protection engine queue is full ({_pending_jobs_count} pending). Maximum capacity is {MAX_PENDING_QUEUE}.",
                headers={"Retry-After": "30"},
            )
        _pending_jobs_count += len(files)

    user_dir = _user_storage_dir(current_user.id)
    created: list[ProtectedImage] = []

    try:
        for file in files:
            image_id = uuid.uuid4()
            ext = (file.filename or "upload.jpg").rsplit(".", 1)[-1].lower()
            original_name = f"{image_id}_original.{ext}"
            original_path = os.path.join(user_dir, original_name)

            # Stream to disk with limits
            await save_upload_with_limits(file, original_path)

            width, height = get_image_dimensions(original_path)
            file_size = os.path.getsize(original_path)

            # ── Idempotency Check ──────────────────────────────────────
            # If user already has an identical pending/processing task, reuse it
            dup_stmt = select(ProtectedImage).where(
                ProtectedImage.user_id == current_user.id,
                ProtectedImage.original_filename == (file.filename or "unknown"),
                ProtectedImage.original_size_bytes == file_size,
                ProtectedImage.status.in_(["pending", "processing"]),
                ProtectedImage.mode == actual_mode,
            )
            dup_res = await db.execute(dup_stmt)
            existing_task = dup_res.scalars().first()
            if existing_task:
                logger.info("Idempotent hit: returning existing in-flight task %s", existing_task.id)
                if os.path.exists(original_path):
                    os.remove(original_path)
                async with _queue_lock:
                    if _pending_jobs_count > 0:
                        _pending_jobs_count -= 1
                created.append(existing_task)
                continue

            record = ProtectedImage(
                id=image_id,
                user_id=current_user.id,
                original_filename=file.filename or "unknown",
                original_path=original_path,
                protected_path=None,
                width=width,
                height=height,
                original_size_bytes=file_size,
                mode=actual_mode,
                requested_mode=requested_mode,
                actual_mode=actual_mode,
                fallback_occurred=fallback_occurred,
                fallback_reason=fallback_reason,
                fallback_user_accepted=fallback_user_accepted,
                engine_version="4.0.0",
                protection_strength=protection_strength,
                preset_name=preset_name,
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

        # Launch background tasks
        for record in created:
            if record.status == "pending":
                cancel_ev = threading.Event()
                _cancel_events[record.id] = cancel_ev
                asyncio.create_task(
                    _run_protection_task(
                        image_id=record.id,
                        original_path=record.original_path,  # type: ignore
                        protected_path=os.path.join(user_dir, f"{record.id}_protected.png"),
                        strength=record.protection_strength,
                        preset_name=record.preset_name,
                        enable_grid=enable_grid,
                        mode=record.actual_mode,
                        cancel_event=cancel_ev,
                    )
                )

        return created
    except Exception:
        # Revert queue counter on upload error
        async with _queue_lock:
            if _pending_jobs_count >= len(files):
                _pending_jobs_count -= len(files)
        raise


@router.post("/cancel/{image_id}", status_code=status.HTTP_200_OK)
async def cancel_protection_task(
    image_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Cancel a pending or in-flight protection task belonging to current user."""
    stmt = select(ProtectedImage).where(
        ProtectedImage.id == image_id,
        ProtectedImage.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    img = result.scalar_one_or_none()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    if img.status in ("completed", "failed", "cancelled"):
        return {"message": f"Task is already {img.status}", "status": img.status}

    if img.status == "pending":
        img.status = "cancelled"
        img.error_message = "Cancelled by user"
        await db.commit()
        return {"message": "Pending job successfully cancelled", "status": "cancelled"}

    # If in-flight processing, trigger cooperative cancellation event
    cancel_ev = _cancel_events.get(image_id)
    if cancel_ev:
        cancel_ev.set()
    img.status = "cancelled"
    img.error_message = "Cancellation initiated by user"
    await db.commit()
    return {"message": "In-flight job cancellation initiated", "status": "cancelled"}


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

    count_stmt = select(sa_func.count()).select_from(ProtectedImage).where(
        ProtectedImage.user_id == current_user.id
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = (
        select(ProtectedImage)
        .where(ProtectedImage.user_id == current_user.id)
        .order_by(ProtectedImage.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    images = list(result.scalars().all())
    image_responses = [ImageResponse.model_validate(image) for image in images]

    return ImageListResponse(
        images=image_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


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

    base_name = os.path.splitext(img.original_filename)[0]

    if img.vault_sealed:
        return FileResponse(
            path=img.protected_path,
            filename=f"varman_{base_name}.enc",
            media_type="application/octet-stream",
        )

    return FileResponse(
        path=img.protected_path,
        filename=f"varman_{base_name}.png",
        media_type="image/png",
    )


@router.post("/{image_id}/vault", status_code=status.HTTP_200_OK)
async def seal_vault_image(
    image_id: uuid.UUID,
    vault_blob: UploadFile,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Accept an encrypted blob from the frontend and replace the plaintext protected file."""
    stmt = select(ProtectedImage).where(
        ProtectedImage.id == image_id,
        ProtectedImage.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    img = result.scalar_one_or_none()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if img.vault_sealed:
        raise HTTPException(status_code=409, detail="Image is already vault-sealed")

    user_dir = _user_storage_dir(current_user.id)
    vault_path = os.path.join(user_dir, f"{image_id}_vault.enc")

    try:
        with open(vault_path, "wb") as out:
            while True:
                chunk = await vault_blob.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        if os.path.exists(vault_path):
            os.remove(vault_path)
        raise HTTPException(status_code=500, detail="Failed to save vault blob") from exc

    if img.protected_path and os.path.exists(img.protected_path):
        try:
            os.remove(img.protected_path)
            logger.info("[Vault] Deleted plaintext protected: %s", img.protected_path)
        except OSError:
            logger.warning("[Vault] Failed to delete plaintext: %s", img.protected_path)

    img.protected_path = vault_path
    img.vault_sealed = True
    img.protected_size_bytes = os.path.getsize(vault_path)
    await db.commit()

    logger.info("[Vault] Image %s sealed successfully", image_id)
    return {"status": "sealed", "image_id": str(image_id)}


@router.delete("/purge-all", status_code=status.HTTP_204_NO_CONTENT)
async def purge_all_images(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Delete all images belonging to the current user."""
    stmt = select(ProtectedImage).where(ProtectedImage.user_id == current_user.id)
    result = await db.execute(stmt)
    images = result.scalars().all()

    for img in images:
        for path in (img.original_path, img.protected_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        await db.delete(img)

    await db.commit()


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

    for path in (img.original_path, img.protected_path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Failed to delete file: %s", path)

    await db.delete(img)
    await db.commit()
