import os
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from core.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES_PER_FILE, MAX_UPLOAD_FILES


async def save_upload_with_limits(upload: UploadFile, destination_path: str) -> None:
    ext = Path(os.path.basename(upload.filename or "fallback.jpg")).suffix.lower().lstrip(".")
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    try:
        total = 0
        with open(destination_path, "wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES_PER_FILE:
                    raise HTTPException(status_code=413, detail="File exceeds maximum allowed size")
                output.write(chunk)

        _validate_saved_image(destination_path)
    except HTTPException:
        _delete_partial_upload(destination_path)
        raise
    except (OSError, UnidentifiedImageError) as exc:
        _delete_partial_upload(destination_path)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def validate_upload_count(files: list[UploadFile]) -> None:
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_UPLOAD_FILES} files per request")


def _validate_saved_image(path: str) -> None:
    with Image.open(path) as image:
        image.verify()


def _delete_partial_upload(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
