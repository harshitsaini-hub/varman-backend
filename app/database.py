from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

is_postgres = settings.database_url.startswith("postgresql")

engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
}

if is_postgres:
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 10

engine = create_async_engine(
    settings.database_url,
    **engine_kwargs
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
    """Create all tables that don't exist yet and seed default user."""
    import logging

    logger = logging.getLogger("varman")

    try:
        from app.models import Base  # deferred to avoid circular import

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        # Seed default operator
        if settings.varman_admin_email and settings.varman_admin_password:
            from app.auth import security
            from app.models.user import User
            from sqlalchemy import select

            async with async_session() as session:
                stmt = select(User).where(User.email == settings.varman_admin_email)
                res = await session.execute(stmt)
                if res.scalar_one_or_none() is None:
                    logger.info("[STARTUP] Seeding default operator %s...", settings.varman_admin_display_name)
                    hashed_pwd = security.get_password_hash(settings.varman_admin_password)
                    default_user = User(
                        email=settings.varman_admin_email,
                        hashed_password=hashed_pwd,
                        display_name=settings.varman_admin_display_name
                    )
                    session.add(default_user)
                    await session.commit()
                    logger.info("[STARTUP] Seeding complete.")
        else:
            logger.info("[STARTUP] Admin credentials not configured in environment. Skipping operator seeding.")
    except Exception as exc:
        logger.exception(
            "[STARTUP] Could not connect to PostgreSQL or initialize DB. "
            "DB-dependent routes will fail until the database is available."
        )
