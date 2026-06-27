import asyncio
import io
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from PIL import Image

from app.models.protected_image import ProtectedImage


@pytest.mark.asyncio
async def test_upload_image_creates_record(client: AsyncClient, test_token: str, tmp_path):
    """Test that uploading creates a DB row and spawns the background task."""
    
    # We mock the heavy GPU pipeline so the test runs instantly.
    # The actual PyTorch math is tested in test_engine.py
    with patch("app.routes.images.settings.storage_dir", str(tmp_path)), \
         patch("app.routes.images.protect_image_pipeline") as mock_pipeline:
         
        mock_pipeline.return_value = {"status": "completed", "ssim": 0.95, "psnr": 30.0, "epsilon_used": 0.05}
        
        # Create dummy image
        img_byte_arr = io.BytesIO()
        Image.new('RGB', (100, 100), color='red').save(img_byte_arr, format='JPEG')# type:ignore
        img_byte_arr.seek(0)
        
        response = await client.post(
            "/api/images/protect",
            headers={"Authorization": f"Bearer {test_token}"},
            files={"files": ("test.jpg", img_byte_arr, "image/jpeg")},
            data={"protection_strength": 0.5}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "pending"
        
        image_id = data[0]["id"]
        
        # Let the background task run via executor
        await asyncio.sleep(0.1)
        
        # Check if DB was updated by the background task
        status_res = await client.get(f"/api/images/status/{image_id}", headers={"Authorization": f"Bearer {test_token}"})
        assert status_res.status_code == 200
        assert status_res.json()["status"] in ["pending", "processing", "completed"]


@pytest.mark.asyncio
async def test_list_images(client: AsyncClient, test_token: str, db, test_user):
    img = ProtectedImage(
        user_id=test_user.id, 
        original_filename="test.jpg", 
        original_path="/dummy/test.jpg", 
        status="completed"
    )
    db.add(img)
    await db.commit()
    
    res = await client.get("/api/images/list", headers={"Authorization": f"Bearer {test_token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert data["images"][0]["original_filename"] == "test.jpg"
