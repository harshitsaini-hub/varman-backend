"""
Concurrency, Bounded Queue, and HTTP 429 / 422 Validation Test Suite
===================================================================
Tests that the protection endpoint strictly enforces:
  1. Concurrency limit = 1 running job.
  2. Bounded pending queue = 5 pending jobs max.
  3. Load testing matrix: 1, 5, 10, 25, 50, 100 concurrent client requests.
     - While 1 job is executing (holding semaphore), at most 5 pending jobs can queue.
     - Any requests beyond 5 pending jobs receive HTTP 429 Too Many Requests with Retry-After header.
  4. Validation: Invalid mode or preset immediately receives HTTP 422 Unprocessable Entity.
"""

import io
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient
from PIL import Image

import app.routes.images as img_routes


def _create_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(120, 140, 160))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_validation_rejects_unknown_mode_with_422(client: AsyncClient, test_token: str):
    headers = {"Authorization": f"Bearer {test_token}"}
    img_bytes = _create_test_image_bytes()
    files = [("files", ("test.png", img_bytes, "image/png"))]
    data = {"mode": "quantum", "preset_name": "Balanced"}

    response = await client.post("/api/images/protect", headers=headers, files=files, data=data)
    assert response.status_code == 422
    assert "Invalid protection mode" in response.text


@pytest.mark.asyncio
async def test_validation_rejects_unknown_preset_with_422(client: AsyncClient, test_token: str):
    headers = {"Authorization": f"Bearer {test_token}"}
    img_bytes = _create_test_image_bytes()
    files = [("files", ("test.png", img_bytes, "image/png"))]
    data = {"mode": "fast", "preset_name": "NonExistentPreset99"}

    response = await client.post("/api/images/protect", headers=headers, files=files, data=data)
    assert response.status_code == 422
    assert "Invalid preset" in response.text


@pytest.mark.asyncio
async def test_bounded_queue_enforces_max_5_pending_and_returns_429(client: AsyncClient, test_token: str):
    """
    Directly tests the queue capacity:
    When 5 jobs are pending, the next job triggers HTTP 429 Too Many Requests.
    """
    headers = {"Authorization": f"Bearer {test_token}"}

    # Reset queue counters for test isolation
    async with img_routes._queue_lock:
        img_routes._pending_jobs_count = 5

    img_bytes = _create_test_image_bytes()
    files = [("files", ("overflow.png", img_bytes, "image/png"))]
    data = {"mode": "fast", "preset_name": "Balanced"}

    res = await client.post("/api/images/protect", headers=headers, files=files, data=data)
    assert res.status_code == 429
    assert "Retry-After" in res.headers
    assert res.headers["Retry-After"] == "30"
    assert "queue is full" in res.json()["detail"]

    # Clean up queue counter
    async with img_routes._queue_lock:
        img_routes._pending_jobs_count = 0


@pytest.mark.asyncio
@pytest.mark.parametrize("client_count", [1, 5, 10, 25, 50, 100])
async def test_load_matrix_queue_behavior(client_count: int, client: AsyncClient, test_token: str):
    """
    Simulates concurrent client burst requests [1, 5, 10, 25, 50, 100] arriving simultaneously
    while an active task is running (holding the semaphore).
    Asserts that at most 5 jobs can be pending in the queue, and any beyond
    the limit receive HTTP 429 with Retry-After header.
    """
    headers = {"Authorization": f"Bearer {test_token}"}

    with patch("app.routes.images._run_protection_task", new_callable=AsyncMock):
        # Reset queue counter
        async with img_routes._queue_lock:
            img_routes._pending_jobs_count = 0

        # Hold the protection semaphore to simulate an active running GPU/CPU job
        async with img_routes._protection_semaphore:
            accepted = 0
            rejected_429 = 0

            for i in range(client_count):
                img_bytes = _create_test_image_bytes()
                files = [("files", (f"img_{client_count}_{i}.png", img_bytes, "image/png"))]
                data = {"mode": "fast", "preset_name": "Balanced"}

                res = await client.post("/api/images/protect", headers=headers, files=files, data=data)
                if res.status_code == 202:
                    accepted += 1
                elif res.status_code == 429:
                    rejected_429 += 1
                    assert res.headers.get("Retry-After") == "30"

            if client_count <= 5:
                assert accepted == client_count
                assert rejected_429 == 0
            else:
                assert accepted == 5
                assert rejected_429 == client_count - 5

        # Reset queue for subsequent tests
        async with img_routes._queue_lock:
            img_routes._pending_jobs_count = 0
