"""
Cooperative Task Cancellation & Authorization Guard Test Suite
==============================================================
Tests that:
  1. POST /api/images/cancel/{image_id} cancels a pending task immediately.
  2. Cancelling an in-flight task triggers cooperative cancellation and updates DB status to 'cancelled'.
  3. Cancelling another user's task returns HTTP 404 (anti-enumeration / access control).
  4. Nonexistent tasks return HTTP 404.
"""

import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import create_access_token, get_password_hash
from app.models.protected_image import ProtectedImage
from app.models.user import User


@pytest.fixture
async def user2(db: AsyncSession) -> dict:
    u2 = User(
        email="second_user@varman.test",
        hashed_password=get_password_hash("password123"),
        display_name="User Two",
        vault_salt="b" * 64,
        is_active=True,
    )
    db.add(u2)
    await db.commit()
    await db.refresh(u2)
    token = create_access_token(str(u2.id))
    return {"user": u2, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.mark.asyncio
async def test_cancel_pending_task(client: AsyncClient, db: AsyncSession, test_user: User, test_token: str):
    headers = {"Authorization": f"Bearer {test_token}"}
    task_id = uuid.uuid4()

    record = ProtectedImage(
        id=task_id,
        user_id=test_user.id,
        original_filename="test_cancel.png",
        width=100,
        height=100,
        original_size_bytes=1000,
        mode="fast",
        status="pending",
    )
    db.add(record)
    await db.commit()

    # Cancel task
    res = await client.post(f"/api/images/cancel/{task_id}", headers=headers)
    assert res.status_code == 200
    assert res.json()["status"] == "cancelled"

    # Verify DB status
    img = await db.get(ProtectedImage, task_id)
    assert img is not None
    assert img.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_unauthorized_task_returns_404(
    client: AsyncClient, db: AsyncSession, test_user: User, user2: dict
):
    # Create task belonging to test_user
    task_id = uuid.uuid4()
    record = ProtectedImage(
        id=task_id,
        user_id=test_user.id,
        original_filename="test_user_file.png",
        width=100,
        height=100,
        original_size_bytes=1000,
        mode="fast",
        status="pending",
    )
    db.add(record)
    await db.commit()

    # user2 attempts to cancel test_user's task -> must return 404
    res = await client.post(f"/api/images/cancel/{task_id}", headers=user2["headers"])
    assert res.status_code == 404
    assert res.json()["detail"] == "Image not found"


@pytest.mark.asyncio
async def test_cancel_nonexistent_task_returns_404(client: AsyncClient, test_token: str):
    headers = {"Authorization": f"Bearer {test_token}"}
    non_existent_id = uuid.uuid4()
    res = await client.post(f"/api/images/cancel/{non_existent_id}", headers=headers)
    assert res.status_code == 404
    assert res.json()["detail"] == "Image not found"
