import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from celery_worker import process_image
from core.config import STORAGE_DIR
from core.security import require_owned_user_id, require_service_auth
from core.uploads import save_upload_with_limits, validate_upload_count

router = APIRouter()


@router.post("/protect")
async def protect_images(
    user_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    auth_payload: dict = Depends(require_service_auth),
):
    require_owned_user_id(user_id, auth_payload)
    validate_upload_count(files)
    saved_paths = []

    for file in files:
        safe_filename = file.filename if file.filename else "fallback.jpg"
        file_ext = safe_filename.split(".")[-1].lower()
        temp_name = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(STORAGE_DIR, temp_name)

        await save_upload_with_limits(file, temp_path)
        saved_paths.append(temp_path)

        process_image.delay(user_id, temp_path)  # type: ignore[attr-defined]

    return {
        "message": "Images accepted. The Armor is being applied in the background.",
        "files_queued": len(saved_paths),
    }
