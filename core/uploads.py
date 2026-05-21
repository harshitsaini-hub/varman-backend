import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from core.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES_PER_FILE, MAX_UPLOAD_FILES


async def save_upload_with_limits(upload: UploadFile, destination_path: str) -> None:
    ext = Path(os.path.basename(upload.filename or "fallback.jpg")).suffix.lower().lstrip(".")
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

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


def validate_upload_count(files: list[UploadFile]) -> None:
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_UPLOAD_FILES} files per request")
