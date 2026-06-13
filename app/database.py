from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

# ── Engine ─────────────────────────────────────────────────────────────────
# Connection-pooled async engine.  pool_size is intentionally small because
# Varman is a portfolio/local project, not a 1000-RPS production service.

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# ── Session factory ────────────────────────────────────────────────────────

async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── FastAPI dependency ─────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped DB session and guarantee cleanup."""
    session = async_session()
    try:
        yield session
    finally:
        await session.close()


# ── Startup helper ─────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables that don't exist yet.

    Called once during the FastAPI lifespan startup.  Uses the ``Base``
    metadata from ``app.models`` so every model registered there gets
    its table created automatically.

    If PostgreSQL is unreachable the server still boots — routes that
    need the DB will fail at request time with a clear error, but the
    health-check and frontend proxy keep working.
    """
    import logging

    logger = logging.getLogger("varman")

    try:
        from app.models import Base  # deferred to avoid circular import

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.warning(
            "[STARTUP] Could not connect to PostgreSQL — tables not created. "
            "DB-dependent routes will fail until the database is available. "
            "Error: %s",
            exc,
        )
