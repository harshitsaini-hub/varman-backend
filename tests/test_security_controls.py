from io import BytesIO

import jwt
from fastapi.testclient import TestClient
from PIL import Image

import main

client = TestClient(main.app)


def _payload_file(name: str = "x.jpg", content: bytes = b"\xff\xd8\xff"):
    return (name, BytesIO(content), "image/jpeg")


def _valid_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jwt_header(payload: dict) -> dict[str, str]:
    token = jwt.encode(payload, "secret", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


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


def test_takedown_allows_matching_jwt_subject(monkeypatch):
    monkeypatch.setattr("core.security.API_KEY", "")
    monkeypatch.setattr("core.security.JWT_SECRET", "secret")
    monkeypatch.setattr("core.security.JWT_ALGORITHM", "HS256")
    response = client.post(
        "/api/takedown/generate",
        headers=_jwt_header({"sub": "u1"}),
        json={
            "user_id": "u1",
            "watermark_id": "w",
            "incident_url": "https://example.com",
            "threat_type": "general",
        },
    )
    assert response.status_code == 200


def test_takedown_rejects_jwt_for_different_user(monkeypatch):
    monkeypatch.setattr("core.security.API_KEY", "")
    monkeypatch.setattr("core.security.JWT_SECRET", "secret")
    monkeypatch.setattr("core.security.JWT_ALGORITHM", "HS256")
    response = client.post(
        "/api/takedown/generate",
        headers=_jwt_header({"sub": "attacker"}),
        json={
            "user_id": "u1",
            "watermark_id": "w",
            "incident_url": "https://example.com",
            "threat_type": "general",
        },
    )
    assert response.status_code == 403


def test_takedown_rejects_jwt_without_subject(monkeypatch):
    monkeypatch.setattr("core.security.API_KEY", "")
    monkeypatch.setattr("core.security.JWT_SECRET", "secret")
    monkeypatch.setattr("core.security.JWT_ALGORITHM", "HS256")
    response = client.post(
        "/api/takedown/generate",
        headers=_jwt_header({}),
        json={
            "user_id": "u1",
            "watermark_id": "w",
            "incident_url": "https://example.com",
            "threat_type": "general",
        },
    )
    assert response.status_code == 403


def test_protect_rejects_invalid_image_and_removes_saved_file(monkeypatch, tmp_path):
    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr("api.routes.protect.STORAGE_DIR", str(tmp_path))
    response = client.post(
        "/protect",
        headers={"X-API-Key": "testkey"},
        data={"user_id": "u1"},
        files={"files": _payload_file("bad.jpg", b"not really an image")},
    )
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_protect_rejects_oversized_file_and_removes_saved_file(monkeypatch, tmp_path):
    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr("core.uploads.MAX_UPLOAD_BYTES_PER_FILE", 2)
    monkeypatch.setattr("api.routes.protect.STORAGE_DIR", str(tmp_path))
    response = client.post(
        "/protect",
        headers={"X-API-Key": "testkey"},
        data={"user_id": "u1"},
        files={"files": _payload_file("big.png", _valid_png_bytes())},
    )
    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_protect_accepts_valid_image_and_enqueues(monkeypatch, tmp_path):
    queued = []

    def fake_delay(user_id: str, path: str) -> None:
        queued.append((user_id, path))

    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr("api.routes.protect.STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr("api.routes.protect.process_image.delay", fake_delay)
    response = client.post(
        "/protect",
        headers={"X-API-Key": "testkey"},
        data={"user_id": "u1"},
        files={"files": _payload_file("ok.png", _valid_png_bytes())},
    )
    assert response.status_code == 200
    assert response.json()["files_queued"] == 1
    assert queued[0][0] == "u1"
    assert queued[0][1].startswith(str(tmp_path))
