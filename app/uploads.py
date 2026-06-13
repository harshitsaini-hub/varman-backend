import os
import logging
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks


async def save_upload_with_limits(file: UploadFile, dest_path: str) -> None:
    """Stream an uploaded file to *dest_path* with strict guardrails.

    1. Validates the filename extension against the allow-list.
    2. Streams in 1 MB chunks, aborting if the cumulative size exceeds
       ``settings.max_upload_bytes``.
    3. After the full file is on disk, opens it with Pillow and calls
       ``Image.verify()`` to confirm it is a valid image.  This catches
       files that just have a ``.jpg`` extension but are actually ZIPs,
       EXEs, etc.
    4. On *any* failure the partially-written file is cleaned up before
       the exception propagates.
    """
    # ── Extension check ────────────────────────────────────────────────
    ext = _safe_extension(file.filename)
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '.{ext}' is not allowed. "
                f"Accepted: {', '.join(sorted(settings.allowed_extensions_set))}"
            ),
        )

    # ── Streamed write with size cap ───────────────────────────────────
    total_bytes = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    _safe_delete(dest_path)
                    mb_limit = settings.max_upload_bytes / (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {mb_limit:.0f} MB size limit.",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        _safe_delete(dest_path)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc

    # ── Image validity check (Pillow) ──────────────────────────────────
    try:
        with open(dest_path, "rb") as f:
            img = Image.open(BytesIO(f.read()))
            img.verify()  # checks header integrity
    except Exception:
        _safe_delete(dest_path)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )


def validate_upload_count(files: list[UploadFile]) -> None:
    """Reject requests that exceed the per-request file count limit."""
    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {settings.max_upload_files} per request.",
        )


def get_image_dimensions(path: str) -> tuple[int, int]:
    """Return (width, height) of the image at *path*."""
    with Image.open(path) as img:
        return img.size  # (width, height)


# ── Internal helpers ───────────────────────────────────────────────────────


def _safe_extension(filename: str | None) -> str:
    """Extract the lowercase extension from a filename, or 'unknown'."""
    if not filename or "." not in filename:
        return "unknown"
    return filename.rsplit(".", maxsplit=1)[-1].lower()


def _safe_delete(path: str) -> None:
    """Delete a file if it exists; swallow errors so cleanup never masks
    the original exception."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        logger.warning("Failed to clean up partial upload: %s", path)
