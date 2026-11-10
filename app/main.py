import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, update

from app.config import settings
from app.database import async_session, init_db
from app.models.protected_image import ProtectedImage

logger = logging.getLogger("varman")


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the Varman API.

    Startup:
      1. Enforce that critical secrets are configured.
      2. Create all database tables (if they don't exist).
      3. Run non-destructive schema migrations (add missing columns if any).
      4. Recover orphaned in-flight jobs from prior crashes/restarts.
      5. Log readiness.
    """
    # ── Startup guards ─────────────────────────────────────────────────
    if not settings.jwt_secret or settings.jwt_secret == "change-this-to-a-long-random-string":
        logger.warning(
            "[STARTUP] JWT_SECRET is not set or still at its example value. "
            "Auth will be insecure. Set a proper value in .env for production."
        )

    # ── Database bootstrap ─────────────────────────────────────────────
    logger.info("[STARTUP] Loaded settings.database_url: %s", settings.database_url)
    logger.info("[STARTUP] Loaded settings.varman_admin_email: %s", settings.varman_admin_email)
    logger.info("[STARTUP] Loaded settings.varman_admin_password length: %d", len(settings.varman_admin_password) if settings.varman_admin_password else 0)
    await init_db()
    logger.info("[STARTUP] Database tables initialised.")

    # ── Schema migration & orphan cleanup ──────────────────────────────
    try:
        from app.protection.gpu_probe import probe_gpu_available
        gpu_ok = probe_gpu_available(force_refresh=True)
        logger.info("[STARTUP] GPU Acceleration Probe Status: %s", gpu_ok)
    except Exception as e:
        logger.warning("[STARTUP] GPU probe notice: %s", e)

    try:
        async with async_session() as db:
            # 1. SQLite schema migration check for Phase 2/3 metadata columns
            try:
                res = await db.execute(text("PRAGMA table_info(protected_images)"))
                cols = [row[1] for row in res.fetchall()]
                if cols:
                    col_defs = [
                        ("mode", "VARCHAR(20) DEFAULT 'fast' NOT NULL"),
                        ("requested_mode", "VARCHAR(20) DEFAULT 'fast' NOT NULL"),
                        ("actual_mode", "VARCHAR(20) DEFAULT 'fast' NOT NULL"),
                        ("fallback_occurred", "BOOLEAN DEFAULT 0 NOT NULL"),
                        ("fallback_reason", "TEXT"),
                        ("fallback_user_accepted", "BOOLEAN DEFAULT 0 NOT NULL"),
                        ("engine_version", "VARCHAR(32) DEFAULT '4.0.0' NOT NULL"),
                        ("processing_duration", "FLOAT"),
                        ("eot_steps_executed", "INTEGER"),
                        ("eps_budget", "FLOAT"),
                        ("alpha_step", "FLOAT"),
                        ("w_fn", "FLOAT"),
                        ("mask_scale", "FLOAT"),
                    ]
                    for col_name, col_sql in col_defs:
                        if col_name not in cols:
                            logger.info("[STARTUP] Adding missing '%s' column to protected_images table...", col_name)
                            await db.execute(text(f"ALTER TABLE protected_images ADD COLUMN {col_name} {col_sql}"))
                            await db.commit()
                            logger.info("[STARTUP] '%s' column successfully added.", col_name)
            except Exception as e:
                logger.warning("[STARTUP] Schema migration check notice: %s", e)

            # 2. Cleanup orphaned in-flight tasks from prior server restarts
            try:
                stmt = (
                    update(ProtectedImage)
                    .where(ProtectedImage.status.in_(["pending", "processing"]))
                    .values(status="failed", error_message="Interrupted by server restart")
                )
                result = await db.execute(stmt)
                if result.rowcount and result.rowcount > 0:
                    await db.commit()
                    logger.info("[STARTUP] Recovered %d orphaned jobs to failed state.", result.rowcount)
            except Exception as e:
                logger.warning("[STARTUP] Orphaned job recovery notice: %s", e)
    except Exception as e:
        logger.warning("[STARTUP] Database startup recovery notice: %s", e)

    logger.info("[STARTUP] Storage dir: %s", settings.storage_dir)
    logger.info("[STARTUP] Device: %s", settings.device)
    logger.info("[STARTUP] Default preset: %s", settings.default_preset)
    logger.info("[STARTUP] Varman 4.0 API is ready.")

    yield  # ── application runs here ───────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("[SHUTDOWN] Varman API shutting down.")


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    app = FastAPI(
        title="Varman API",
        description="Adversarial image protection against AI deepfakes.",
        version="4.0.0",
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

    # ── Exception Handlers ──────────────────────────────────────────────
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi import Request, status

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        parsed_errors = []
        for err in exc.errors():
            loc = err.get("loc", ())
            field = str(loc[-1]) if loc else "request"
            msg = err.get("msg", "Validation error")
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, "):]
            parsed_errors.append({"field": field, "message": msg})

        # Log ONLY field names and error types, never user-submitted values
        log_items = [
            f"{e.get('loc', ())[-1] if e.get('loc') else 'request'}:{e.get('type', 'unknown')}"
            for e in exc.errors()
        ]
        logger.warning("[Validation] Validation error on fields: %s", ", ".join(log_items))

        first_field = parsed_errors[0]["field"] if parsed_errors else "request"
        first_msg = parsed_errors[0]["message"] if parsed_errors else "Invalid request parameters"
        detail_msg = f"{first_field}: {first_msg}"

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": detail_msg,
                "errors": parsed_errors,
            },
        )

    # ── Routers ─────────────────────────────────────────────────────────
    from app.auth.router import router as auth_router
    from app.routes.images import router as images_router
    from app.routes.analytics import router as analytics_router
    app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
    app.include_router(images_router, prefix="/api/images", tags=["Images"])
    app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])

    # ── Root & Health check ────────────────────────────────────────────
    @app.get("/", tags=["System"])
    async def root():
        return {
            "service": "Varman Adversarial Facial Privacy Engine",
            "version": "4.0.0",
            "status": "online",
            "docs_url": "/docs",
        }

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "service": "varman"}

    # ── Capabilities endpoint ──────────────────────────────────────────
    @app.get("/api/capabilities", tags=["System"])
    @app.get("/capabilities", tags=["System"])
    async def get_capabilities():
        from app.protection.gpu_probe import get_cached_gpu_available
        gpu_available = get_cached_gpu_available()
        modes_supported = ["fast", "hard"] if gpu_available else ["fast"]
        presets = {
            "fast": ["Light", "Balanced", "Aggressive"],
            "hard": ["F5"] if gpu_available else [],
        }
        return {
            "service": "Varman Engine",
            "version": "4.0.0",
            "gpu_available": gpu_available,
            "modes_supported": modes_supported,
            "default_mode": "hard" if gpu_available else "fast",
            "presets": presets,
            "max_queue_depth": 5,
            "concurrency_limit": 1,
            "requires_consent_for_cpu_fallback": True,
        }

    return app


# Module-level app instance — Uvicorn points at ``app.main:app``
app = create_app()
