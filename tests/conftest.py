from typing import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth.security import get_password_hash
from app.database import get_db
from app.main import app
from app.models import Base
from app.models.user import User

# In-memory SQLite for blazing fast, isolated tests
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    """Create all tables once per test session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield
    finally:
        try:
            async with test_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        finally:
            await test_engine.dispose()
            from app.database import engine as app_engine
            await app_engine.dispose()

@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a DB session and aggressively clear data after each test to prevent bleeding."""
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            # Truncate tables to ensure tests never collide (Feedback point #3)
            for table in reversed(Base.metadata.sorted_tables):
                await session.execute(table.delete())
            await session.commit()
            await session.close()

@pytest_asyncio.fixture
async def client(db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Test client with the DB dependency overridden."""
    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def test_user(db: AsyncSession) -> User:
    """Provide a pre-registered active test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("password123"),
        display_name="Test User",
        vault_salt="a" * 64,
        is_active=True
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@pytest_asyncio.fixture
async def test_token(client: AsyncClient, test_user: User) -> str:
    """Provide a valid JWT token for the test user."""
    response = await client.post(
        "/api/auth/login",
        data={"username": "test@example.com", "password": "password123"}
    )
    return response.json()["access_token"]
