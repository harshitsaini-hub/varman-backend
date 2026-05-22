from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.routes.protect import router as protect_router
from core.config import CORS_ALLOWED_ORIGINS, PHASH_MATCH_THRESHOLD, REGION_PHASH_MATCH_THRESHOLD
from core.security import require_owned_user_id, require_service_auth
from services import detection_service, notification_service
from services.db_service import (
    get_db_connection,
    init_db,
    lookup_phash_candidates_global,
    lookup_phash_global,
)

app = FastAPI(title="Project AMOR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOWED_ORIGINS),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
app.include_router(protect_router)
ServiceAuth = Annotated[dict, Depends(require_service_auth)]

# --- PYDANTIC MODELS (Strict Data Validation) ---
class RadarPayload(BaseModel):
    suspect_hash: str
    source_url: str
    platform: str
    candidate_hashes: list[str] = Field(default_factory=list)


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
# --- ENDPOINT 2: THE SILENT FLARE (Receiver) ---
@app.post("/api/radar/flag")
async def radar_flag(payload: RadarPayload, _auth: ServiceAuth):
    """
    Receives pings from the Chrome Extension and Python Scrapers.
    Queries pgvector natively. If matched, triggers the alert.
    """
    db = get_db_connection()
    try:
        match = lookup_phash_global(
            db,
            payload.suspect_hash,
            threshold=PHASH_MATCH_THRESHOLD,
        )

        if match is None and payload.candidate_hashes:
            candidates = [
                (f"client_candidate_{index}", candidate_hash)
                for index, candidate_hash in enumerate(payload.candidate_hashes)
            ]
            match = lookup_phash_candidates_global(
                db,
                candidates,
                threshold=REGION_PHASH_MATCH_THRESHOLD,
            )
            if match:
                match.detection_method = "region_phash"

        if match:
            alert_queued = notification_service.queue_radar_alert(
                user_id=match.user_id,
                suspect_url=payload.source_url,
                image_url="[Radar Local Image]",
                platform=payload.platform,
                context=(
                    "Detected via AMOR Radar Network; "
                    f"{detection_service.describe_match(match)}"
                ),
            )
            return {
                "status": "match_found",
                "action": "user_notification_queued" if alert_queued else "alert_queue_failed",
            }

        return {"status": "clean"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        db.close()


# --- ENDPOINT 3: THE LEGAL ARSENAL ---
@app.post("/api/takedown/generate")
async def generate_takedown(
    payload: TakedownPayload,
    auth_payload: ServiceAuth,
):
    """
    Generates the specific legal weapon based on the threat type.
    """
    require_owned_user_id(payload.user_id, auth_payload)

    if payload.threat_type == "ncii":
        template = (
            "URGENT: Safety Violation / Non-Consensual Intimate Imagery.\n"
            "I am reporting a severe safety violation at the following URL: "
            f"{payload.incident_url}.\n"
            "This content is an unauthorized, manipulated deepfake designed to cause harm. "
            "Please review under your platform's strict NCII and safety guidelines immediately."
        )
        # Direct the user to the specific platform portal rather than trying to send a generic email
        if "facebook" in payload.incident_url or "instagram" in payload.incident_url:
            action_url = "https://www.facebook.com/help/contact/ncii_portal"
        else:
            action_url = "https://support.google.com/websearch/answer/6302812"

    elif payload.threat_type == "general":
        template = (
            "DMCA Takedown Notice / Right of Publicity Violation.\n"
            "I am the legal copyright owner of the underlying base image used to create the "
            f"manipulated content at: {payload.incident_url}.\n"
            "Cryptographic Proof of Ownership (dwtDct Steganography Watermark ID): "
            f"{payload.watermark_id}\n"
            "I request the immediate removal of this infringing and manipulated content."
        )
        action_url = None
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid threat_type. Must be 'general' or 'ncii'.",
        )

    return {"status": "success", "template": template, "action_url": action_url}
