from io import BytesIO

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def _payload_file(name: str = "x.jpg", content: bytes = b"\xff\xd8\xff"):
    return (name, BytesIO(content), "image/jpeg")


def test_protect_requires_auth():
    response = client.post(
        "/protect",
        data={"user_id": "u1"},
        files={"files": _payload_file()},
    )
    assert response.status_code == 401


def test_protect_rejects_too_many_files(monkeypatch):
    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr("core.config.MAX_UPLOAD_FILES", 1)
    monkeypatch.setattr("core.uploads.MAX_UPLOAD_FILES", 1)
    response = client.post(
        "/protect",
        headers={"X-API-Key": "testkey"},
        data={"user_id": "u1"},
        files=[("files", _payload_file("1.jpg")), ("files", _payload_file("2.jpg"))],
    )
    assert response.status_code == 400


def test_takedown_requires_auth():
    response = client.post(
        "/api/takedown/generate",
        json={
            "user_id": "u1",
            "watermark_id": "w",
            "incident_url": "https://example.com",
            "threat_type": "general",
        },
    )
    assert response.status_code == 401
