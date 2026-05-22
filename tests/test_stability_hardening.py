from pathlib import Path

from PIL import Image

from services import bloom_service, image_pipeline


def test_bloom_filter_accepts_streaming_phash_iterable():
    consumed = []

    def phashes():
        for index in range(3):
            phash = f"{index:016x}"
            consumed.append(phash)
            yield phash

    bloom = bloom_service.build_global_bloom_filter_from_iterable(phashes(), capacity=3)

    assert consumed == ["0000000000000000", "0000000000000001", "0000000000000002"]
    assert bloom["capacity"] >= 10000
    assert bloom["num_bits"] > 0


def test_failed_armor_validation_does_not_persist_or_deliver(monkeypatch, tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (32, 32), color=(20, 80, 160)).save(image_path)

    def failed_armor(image_array, watermark_id):
        return image_array, {
            "passed": False,
            "watermark_id": watermark_id,
            "recovered_prefix": None,
            "compression_quality_tested": 75,
        }

    def fail_if_persisted(*args, **kwargs):
        raise AssertionError("failed armor must not be persisted")

    monkeypatch.setattr(image_pipeline.amor_service, "armor_image", failed_armor)
    monkeypatch.setattr(image_pipeline.db_service, "save_protected_image", fail_if_persisted)
    monkeypatch.setattr(image_pipeline, "extract_face_encoding", lambda image_array: None)

    result = image_pipeline.process_image_file(
        user_id="u1",
        temp_file_path=str(image_path),
        db=object(),
        delete_original=False,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "armor_validation_failed"
    assert not list(Path(tmp_path).glob("armored_*"))
