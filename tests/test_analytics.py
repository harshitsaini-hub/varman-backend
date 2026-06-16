import pytest
from httpx import AsyncClient

from app.models.protected_image import ProtectedImage

@pytest.mark.asyncio
async def test_analytics_summary_aggregates_correctly(client: AsyncClient, test_token: str, db, test_user):
    """
    Since we strictly truncate tables between tests (Feedback #3),
    these mock rows will never collide with test_images.py.
    """
    img1 = ProtectedImage(user_id=test_user.id, original_filename="1.jpg", original_path="/d", status="completed", ssim_score=0.90, psnr_score=30.0, processing_time_ms=100)
    img2 = ProtectedImage(user_id=test_user.id, original_filename="2.jpg", original_path="/d", status="completed", ssim_score=0.95, psnr_score=32.0, processing_time_ms=200)
    img3 = ProtectedImage(user_id=test_user.id, original_filename="3.jpg", original_path="/d", status="failed", processing_time_ms=50)
    img4 = ProtectedImage(user_id=test_user.id, original_filename="4.jpg", original_path="/d", status="pending")
    
    db.add_all([img1, img2, img3, img4])
    await db.commit()
    
    res = await client.get("/api/analytics/summary", headers={"Authorization": f"Bearer {test_token}"})
    assert res.status_code == 200
    data = res.json()
    
    assert data["total_images"] == 4
    assert data["completed_images"] == 2
    assert data["failed_images"] == 1
    assert data["pending_images"] == 1
    
    # Average of 0.90 and 0.95 = 0.925
    assert data["avg_ssim"] == 0.925
    # Total time of COMPLETED images (100 + 200 = 300)
    assert data["total_processing_time_ms"] == 300
