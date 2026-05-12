import os
import uuid
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import STORAGE_DIR
from celery_worker import process_image_task
from services.db_service import init_db, get_db_connection, lookup_phash_global
from services.notification_service import send_radar_alert

app = FastAPI(title="Project AMOR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC MODELS (Strict Data Validation) ---
class RadarPayload(BaseModel):
    suspect_hash: str
    source_url: str
    platform: str

class TakedownPayload(BaseModel):
    user_id: str
    watermark_id: str
    incident_url: str
    threat_type: str  # Must be "general" or "ncii"

# --- LIFECYCLE ---
@app.on_event("startup")
def startup_event():
    init_db()
    print("AMOR Database (pgvector) initialized.")

# --- ENDPOINT 1: THE ARMOR FACTORY ---
@app.post("/protect")
async def protect_images(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    saved_paths = []
    for file in files:
        safe_filename = file.filename if file.filename else "fallback.jpg"
        file_ext = safe_filename.split(".")[-1]
        temp_name = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(STORAGE_DIR, temp_name)

        with open(temp_path, "wb") as f:
            f.write(await file.read())
        saved_paths.append(temp_path)

        # Handoff to Celery
        process_image_task.delay(user_id, temp_path)

    return {
        "message": "Images accepted. The Armor is being applied in the background.",
        "files_queued": len(saved_paths),
    }

# --- ENDPOINT 2: THE SILENT FLARE (Receiver) ---
@app.post("/api/radar/flag")
async def radar_flag(payload: RadarPayload):
    """
    Receives pings from the Chrome Extension and Python Scrapers.
    Queries pgvector natively. If matched, triggers the alert.
    """
    db = get_db_connection()
    try:
        # Using a threshold of 10 for the XOR bitwise comparison
        match = lookup_phash_global(db, payload.suspect_hash, threshold=10)
        
        if match:
            # Trigger the email/alert to the user (currently logs via notification_service)
            send_radar_alert(
                user_id=match.user_id,
                suspect_url=payload.source_url,
                image_url="[Radar Local Image]",
                platform=payload.platform,
                context="Detected via AMOR Radar Network"
            )
            return {"status": "match_found", "action": "user_notified"}
            
        return {"status": "clean"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# --- ENDPOINT 3: THE LEGAL ARSENAL ---
@app.post("/api/takedown/generate")
async def generate_takedown(payload: TakedownPayload):
    """
    Generates the specific legal weapon based on the threat type.
    """
    # Note: We are trusting the payload.user_id here. 
    # TODO: Replace with JWT Auth token extraction later.

    if payload.threat_type == "ncii":
        template = (
            f"URGENT: Safety Violation / Non-Consensual Intimate Imagery.\n"
            f"I am reporting a severe safety violation at the following URL: {payload.incident_url}.\n"
            f"This content is an unauthorized, manipulated deepfake designed to cause harm. "
            f"Please review under your platform's strict NCII and safety guidelines immediately."
        )
        # Direct the user to the specific platform portal rather than trying to send a generic email
        action_url = "https://www.facebook.com/help/contact/ncii_portal" if "facebook" in payload.incident_url or "instagram" in payload.incident_url else "https://support.google.com/websearch/answer/6302812"
        
    elif payload.threat_type == "general":
        template = (
            f"DMCA Takedown Notice / Right of Publicity Violation.\n"
            f"I am the legal copyright owner of the underlying base image used to create the manipulated content at: {payload.incident_url}.\n"
            f"Cryptographic Proof of Ownership (dwtDct Steganography Watermark ID): {payload.watermark_id}\n"
            f"I request the immediate removal of this infringing and manipulated content."
        )
        action_url = None
    else:
        raise HTTPException(status_code=400, detail="Invalid threat_type. Must be 'general' or 'ncii'.")

    return {
        "status": "success",
        "template": template,
        "action_url": action_url
    }