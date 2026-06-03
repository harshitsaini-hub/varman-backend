from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

import main
from services import db_service, image_pipeline
from services.db_service import PhashMatch
from utils.hashing import PhashCandidate

client = TestClient(main.app)


class FakeDb:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.cursor_instance = FakeCursor()

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self):
        self.executemany_sql = None
        self.executemany_rows = None
        self.closed = False

    def executemany(self, sql, rows):
        self.executemany_sql = sql
        self.executemany_rows = rows

    def close(self):
        self.closed = True


def test_successful_pipeline_persists_primary_and_candidate_phashes(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (96, 96), color=(240, 240, 240))
    for x in range(24, 72):
        for y in range(24, 72):
            image.putpixel((x, y), (20, 80, 160))
    image.save(image_path)

    db = FakeDb()
    primary_rows = []
    hash_batches = []

    def passed_armor(image_array, watermark_id):
        return image_array, {
            "passed": True,
            "watermark_id": watermark_id,
            "recovered_prefix": watermark_id[:8],
            "compression_quality_tested": 75,
        }

    def save_primary(**kwargs):
        assert kwargs["commit"] is False
        primary_rows.append(kwargs)
        return 42

    def save_hashes(db_arg, protected_image_id, phash_entries, *, commit):
        assert db_arg is db
        assert protected_image_id == 42
        assert commit is False
        hash_batches.append(list(phash_entries))

    monkeypatch.setattr(image_pipeline, "TEMP_STORAGE_PATH", str(tmp_path))
    monkeypatch.setattr(image_pipeline.amor_service, "armor_image", passed_armor)
    monkeypatch.setattr(image_pipeline, "extract_face_encoding", lambda image_array: None)
    monkeypatch.setattr(image_pipeline.db_service, "save_protected_image", save_primary)
    monkeypatch.setattr(image_pipeline.db_service, "save_protected_image_hashes", save_hashes)

    result = image_pipeline.process_image_file(
        user_id="u1",
        temp_file_path=str(image_path),
        db=db,
        delete_original=False,
    )

    assert result["status"] == "success"
    assert result["phash"] == primary_rows[0]["phash"]
    assert primary_rows[0]["user_id"] == "u1"
    assert hash_batches[0][0].hash_kind == "full"
    assert len(hash_batches[0]) > 1
    assert db.commits == 1
    assert db.rollbacks == 0
    assert not db.closed


def test_save_protected_image_hashes_deduplicates_and_commits():
    db = FakeDb()
    entries = [
        PhashCandidate(hash_kind="full", phash="abc"),
        PhashCandidate(hash_kind="full", phash="abc"),
        PhashCandidate(hash_kind="center_92", phash="def"),
        SimpleNamespace(hash_kind="", phash="ignored"),
        SimpleNamespace(hash_kind="center_84", phash=None),
    ]

    db_service.save_protected_image_hashes(db, 42, entries)

    assert db.cursor_instance.executemany_rows == [
        (42, "full", "abc"),
        (42, "center_92", "def"),
    ]
    assert "ON CONFLICT DO NOTHING" in db.cursor_instance.executemany_sql
    assert db.cursor_instance.closed is True
    assert db.commits == 1
    assert db.rollbacks == 0


def test_radar_flag_returns_clean_for_extension_hash_miss(monkeypatch):
    db = FakeDb()

    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr(main, "get_db_connection", lambda: db)
    monkeypatch.setattr(main, "lookup_phash_global", lambda db_arg, phash, threshold: None)

    response = client.post(
        "/api/radar/flag",
        headers={"X-API-Key": "testkey"},
        json={
            "suspect_hash": "0" * 16,
            "source_url": "chrome-extension://scan",
            "platform": "chrome_extension",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "clean"}
    assert db.closed is True


def test_radar_flag_rejects_malformed_hash_before_database(monkeypatch):
    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr(
        main,
        "get_db_connection",
        lambda: (_ for _ in ()).throw(AssertionError("database should not be touched")),
    )

    response = client.post(
        "/api/radar/flag",
        headers={"X-API-Key": "testkey"},
        json={
            "suspect_hash": "not-a-phash",
            "source_url": "chrome-extension://scan",
            "platform": "chrome_extension",
        },
    )

    assert response.status_code == 422


def test_radar_preflight_allows_chrome_extension_origin():
    response = client.options(
        "/api/radar/flag",
        headers={
            "Origin": "chrome-extension://abcdef",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key, Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "chrome-extension://abcdef"


def test_radar_flag_uses_extension_candidate_hashes_and_queues_alert(monkeypatch):
    db = FakeDb()
    queued_alerts = []

    def hit_candidate(db_arg, candidates, threshold):
        assert db_arg is db
        assert candidates == [("client_candidate_0", "1" * 16)]
        return PhashMatch(
            user_id="u1",
            phash="stored",
            watermark_id="wm1",
            confidence_score=0.9,
            distance=3,
            suspect_hash_kind="client_candidate_0",
            matched_hash_kind="center_92",
        )

    def queue_alert(**kwargs):
        queued_alerts.append(kwargs)
        return True

    monkeypatch.setattr("core.security.API_KEY", "testkey")
    monkeypatch.setattr(main, "get_db_connection", lambda: db)
    monkeypatch.setattr(main, "lookup_phash_global", lambda db_arg, phash, threshold: None)
    monkeypatch.setattr(main, "lookup_phash_candidates_global", hit_candidate)
    monkeypatch.setattr(main.notification_service, "queue_radar_alert", queue_alert)

    response = client.post(
        "/api/radar/flag",
        headers={"X-API-Key": "testkey"},
        json={
            "suspect_hash": "0" * 16,
            "candidate_hashes": ["1" * 16],
            "source_url": "chrome-extension://scan",
            "platform": "chrome_extension",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "match_found",
        "action": "user_notification_queued",
    }
    assert queued_alerts[0]["user_id"] == "u1"
    assert queued_alerts[0]["platform"] == "chrome_extension"
    assert "region pHash match" in queued_alerts[0]["context"]
    assert db.closed is True
