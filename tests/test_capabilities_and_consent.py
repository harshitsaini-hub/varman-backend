import asyncio
import io
import uuid
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.protected_image import ProtectedImage
from app.models.user import User
from app.protection.gpu_probe import set_cached_gpu_available, get_cached_gpu_available


def _create_test_image_bytes() -> io.BytesIO:
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100), color="blue").save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.mark.asyncio
async def test_capabilities_endpoint_both_routes(client: AsyncClient):
    """Test both /api/capabilities and /capabilities return the valid schema."""
    set_cached_gpu_available(False)
    for path in ("/api/capabilities", "/capabilities"):
        res = await client.get(path)
        assert res.status_code == 200, f"Failed on path {path}"
        data = res.json()
        assert data["service"] == "Varman Engine"
        assert data["version"] == "4.0.0"
        assert "gpu_available" in data
        assert "modes_supported" in data
        assert "presets" in data
        assert data["max_queue_depth"] == 5
        assert data["concurrency_limit"] == 1
        assert data["requires_consent_for_cpu_fallback"] is True


@pytest.mark.asyncio
async def test_capabilities_cpu_only_vs_gpu(client: AsyncClient):
    """Verify capabilities correctly reflect cached CUDA probe presence/absence."""
    set_cached_gpu_available(False)
    res = await client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert data["gpu_available"] is False
    assert data["modes_supported"] == ["fast"]
    assert data["default_mode"] == "fast"
    assert data["presets"]["hard"] == []

    set_cached_gpu_available(True)
    res = await client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert data["gpu_available"] is True
    assert data["modes_supported"] == ["fast", "hard"]
    assert data["default_mode"] == "hard"
    assert data["presets"]["hard"] == ["F5"]


@pytest.mark.asyncio
async def test_capabilities_does_not_import_torch_on_request_path(client: AsyncClient):
    """Confirm /capabilities serves requests from cached state without touching torch."""
    set_cached_gpu_available(False)
    with patch("torch.cuda.is_available") as mock_cuda:
        res = await client.get("/api/capabilities")
        assert res.status_code == 200
        assert mock_cuda.call_count == 0, "torch.cuda.is_available was unexpectedly called on the request path!"


@pytest.mark.asyncio
async def test_validation_rejects_unknown_mode_422(client: AsyncClient, test_token: str):
    """Unknown protection mode must return HTTP 422."""
    img_bytes = _create_test_image_bytes()
    res = await client.post(
        "/api/images/protect",
        headers={"Authorization": f"Bearer {test_token}"},
        files={"files": ("test.jpg", img_bytes, "image/jpeg")},
        data={"mode": "quantum", "preset_name": "Balanced"},
    )
    assert res.status_code == 422
    assert "Invalid protection mode" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_validation_rejects_unknown_preset_for_fast_mode_422(client: AsyncClient, test_token: str):
    """Unknown preset on Fast Mode must return HTTP 422."""
    img_bytes = _create_test_image_bytes()
    res = await client.post(
        "/api/images/protect",
        headers={"Authorization": f"Bearer {test_token}"},
        files={"files": ("test.jpg", img_bytes, "image/jpeg")},
        data={"mode": "fast", "preset_name": "UltraExtreme"},
    )
    assert res.status_code == 422
    assert "Invalid preset 'UltraExtreme' for Fast Mode" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_validation_rejects_unknown_preset_for_hard_mode_422(client: AsyncClient, test_token: str):
    """Unknown preset on Hard Mode must return HTTP 422."""
    img_bytes = _create_test_image_bytes()
    set_cached_gpu_available(True)
    res = await client.post(
        "/api/images/protect",
        headers={"Authorization": f"Bearer {test_token}"},
        files={"files": ("test.jpg", img_bytes, "image/jpeg")},
        data={"mode": "hard", "preset_name": "Balanced"},
    )
    assert res.status_code == 422
    assert "Invalid preset 'Balanced' for Hard Mode" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_hard_mode_on_cpu_server_without_consent_returns_409(client: AsyncClient, test_token: str):
    """Requesting Hard Mode on a CPU-only server without consent must return HTTP 409 Conflict."""
    img_bytes = _create_test_image_bytes()
    set_cached_gpu_available(False)
    res = await client.post(
        "/api/images/protect",
        headers={"Authorization": f"Bearer {test_token}"},
        files={"files": ("test.jpg", img_bytes, "image/jpeg")},
        data={"mode": "hard", "preset_name": "F5", "allow_cpu_fallback": "false"},
    )
    assert res.status_code == 409
    detail = res.json().get("detail", "")
    assert "Hard Mode requires a CUDA GPU" in detail
    assert "Explicit consent (allow_cpu_fallback=true) required" in detail


@pytest.mark.asyncio
async def test_hard_mode_on_cpu_server_with_consent_falls_back_to_fast(
    client: AsyncClient, test_token: str, tmp_path
):
    """Requesting Hard Mode on a CPU-only server WITH consent proceeds with Fast Mode and records fallback metadata."""
    img_bytes = _create_test_image_bytes()
    set_cached_gpu_available(False)

    with patch("app.routes.images.settings.storage_dir", str(tmp_path)):
        res = await client.post(
            "/api/images/protect",
            headers={"Authorization": f"Bearer {test_token}"},
            files={"files": ("test_consent.jpg", img_bytes, "image/jpeg")},
            data={"mode": "hard", "preset_name": "F5", "allow_cpu_fallback": "true"},
        )

        assert res.status_code == 202
        data = res.json()
        assert len(data) == 1
        record = data[0]
        assert record["status"] == "pending"
        assert record["requested_mode"] == "hard"
        assert record["actual_mode"] == "fast"
        assert record["mode"] == "fast"
        assert record["fallback_occurred"] is True
        assert record["fallback_user_accepted"] is True
        assert record["engine_version"] == "4.0.0"
        assert "Server is CPU-only" in record["fallback_reason"]


@pytest.mark.asyncio
async def test_status_endpoint_returns_all_metadata_fields(
    client: AsyncClient, db: AsyncSession, test_user: User, test_token: str
):
    """Status polling returns complete metadata fields for completed Hard Mode task."""
    headers = {"Authorization": f"Bearer {test_token}"}
    task_id = uuid.uuid4()

    record = ProtectedImage(
        id=task_id,
        user_id=test_user.id,
        original_filename="portrait.jpg",
        width=512,
        height=512,
        original_size_bytes=102400,
        mode="hard",
        requested_mode="hard",
        actual_mode="hard",
        fallback_occurred=False,
        fallback_reason=None,
        fallback_user_accepted=False,
        engine_version="4.0.0",
        status="completed",
        ssim_score=0.9733,
        face_ssim_score=0.9010,
        psnr_score=34.2,
        eot_steps_executed=25,
        eps_budget=8.0 / 255.0,
        alpha_step=(8.0 / 255.0) / 7.0,
        w_fn=1.0,
        mask_scale=1.15,
        processing_time_ms=1950,
        processing_duration=1.95,
    )
    db.add(record)
    await db.commit()

    res = await client.get(f"/api/images/status/{task_id}", headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert data["id"] == str(task_id)
    assert data["status"] == "completed"
    assert data["mode"] == "hard"
    assert data["requested_mode"] == "hard"
    assert data["actual_mode"] == "hard"
    assert data["fallback_occurred"] is False
    assert data["fallback_reason"] is None
    assert data["fallback_user_accepted"] is False
    assert data["engine_version"] == "4.0.0"
    assert data["ssim_score"] == 0.9733
    assert data["face_ssim_score"] == 0.9010
    assert data["eot_steps_executed"] == 25
    assert abs(data["eps_budget"] - (8.0 / 255.0)) < 1e-5
    assert abs(data["alpha_step"] - ((8.0 / 255.0) / 7.0)) < 1e-5
    assert data["w_fn"] == 1.0
    assert data["mask_scale"] == 1.15
    assert data["processing_time_ms"] == 1950
    assert data["processing_duration"] == 1.95
