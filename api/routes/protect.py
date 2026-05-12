import logging
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from core.config import STORAGE_DIR
from services.image_pipeline import process_image_file

logger = logging.getLogger(__name__)
router = APIRouter()


def _safe_extension(filename: str | None) -> str:
    suffix = Path(os.path.basename(filename or "fallback.jpg")).suffix.lower().lstrip(".")
    return suffix or "jpg"


def process_image_background(user_id: str, file_path: str) -> None:
    try:
        result = process_image_file(user_id=user_id, temp_file_path=file_path)
        logger.info("AMOR background image processing completed: %s", result)
    except Exception:
        logger.exception("AMOR background image processing failed for %s", file_path)


@router.post("/protect")
async def protect_images(
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
):
    saved_paths = []

    for file in files:
        file_ext = _safe_extension(file.filename)
        temp_name = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(STORAGE_DIR, temp_name)

        with open(temp_path, "wb") as output_file:
            output_file.write(await file.read())
        saved_paths.append(temp_path)

        background_tasks.add_task(process_image_background, user_id, temp_path)

    return {
        "message": "Images accepted. The Armor is being applied in the background.",
        "files_processing": len(saved_paths),
    }
