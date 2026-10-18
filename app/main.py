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
      (nothing to tear down for now — ONNX models are lazily loaded
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
    logger.info("[STARTUP] Default preset: %s", settings.default_preset)
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
            "version": "3.0.0",
            "status": "online",
            "docs_url": "/docs",
        }

    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "service": "varman"}

    return app


# Module-level app instance — Uvicorn points at ``app.main:app``
app = create_app()
