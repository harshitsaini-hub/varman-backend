import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db

logger = logging.getLogger("varman")


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the Varman API.

    Startup:
      1. Enforce that critical secrets are configured.
      2. Create all database tables (if they don't exist).
      3. Log readiness.

    Shutdown:
      (nothing to tear down for now — PyTorch models are lazily loaded
       and garbage-collected automatically)
    """
    # ── Startup guards ─────────────────────────────────────────────────
    if not settings.jwt_secret or settings.jwt_secret == "change-this-to-a-long-random-string":
        logger.warning(
            "[STARTUP] JWT_SECRET is not set or still at its example value. "
            "Auth will be insecure.  Set a proper value in .env for production."
        )

    # ── Database bootstrap ─────────────────────────────────────────────
    logger.info("[STARTUP] Loaded settings.database_url: %s", settings.database_url)
    logger.info("[STARTUP] Loaded settings.varman_admin_email: %s", settings.varman_admin_email)
    logger.info("[STARTUP] Loaded settings.varman_admin_password length: %d", len(settings.varman_admin_password) if settings.varman_admin_password else 0)
    await init_db()
    logger.info("[STARTUP] Database tables initialised.")
    logger.info("[STARTUP] Storage dir: %s", settings.storage_dir)
    logger.info("[STARTUP] Device: %s", settings.device)
    logger.info("[STARTUP] EoT iterations: %s", settings.eot_iterations)
    logger.info("[STARTUP] Varman API is ready.")

    yield  # ── application runs here ───────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("[SHUTDOWN] Varman API shutting down.")


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    app = FastAPI(
        title="Varman API",
        description="Adversarial image protection against AI deepfakes.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────
    from app.auth.router import router as auth_router
    from app.routes.images import router as images_router
    from app.routes.analytics import router as analytics_router
    app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
    app.include_router(images_router, prefix="/api/images", tags=["Images"])
    app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])

    # ── Health check ───────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "service": "varman"}

    return app


# Module-level app instance — Uvicorn points at ``app.main:app``
app = create_app()
