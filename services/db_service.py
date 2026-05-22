from dataclasses import dataclass

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from config import (
    FACE_MATCH_DISTANCE_THRESHOLD,
    PHASH_MATCH_THRESHOLD,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    REGION_PHASH_MATCH_THRESHOLD,
)
from models.protected_image import (
    CREATE_PROTECTED_IMAGE_HASHES_IMAGE_ID_INDEX_SQL,
    CREATE_PROTECTED_IMAGE_HASHES_PHASH_INDEX_SQL,
    CREATE_PROTECTED_IMAGE_HASHES_TABLE_SQL,
    CREATE_PROTECTED_IMAGES_FACE_HNSW_INDEX_SQL,
    CREATE_PROTECTED_IMAGES_PHASH_INDEX_SQL,
    CREATE_PROTECTED_IMAGES_TABLE_SQL,
)

PHASH_UNION_SQL = """
SELECT phash FROM protected_images
UNION
SELECT phash FROM protected_image_hashes
"""

DB_PARAMS = {
    "dbname": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
}


# ── PhashMatch — proper dataclass so Pylance can see its attributes ────────


@dataclass
class PhashMatch:
    user_id: str
    phash: str
    watermark_id: str
    confidence_score: float
    armored_image_path: str | None = None
    detection_method: str = "phash"
    distance: float | None = None
    suspect_hash_kind: str | None = None
    matched_hash_kind: str | None = None


# ── Connection ─────────────────────────────────────────────────────────────


def get_db_connection():
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(CREATE_PROTECTED_IMAGES_TABLE_SQL)
    cursor.execute(CREATE_PROTECTED_IMAGE_HASHES_TABLE_SQL)
    cursor.execute(CREATE_PROTECTED_IMAGES_PHASH_INDEX_SQL)
    cursor.execute(CREATE_PROTECTED_IMAGE_HASHES_PHASH_INDEX_SQL)
    cursor.execute(CREATE_PROTECTED_IMAGE_HASHES_IMAGE_ID_INDEX_SQL)
    cursor.execute(CREATE_PROTECTED_IMAGES_FACE_HNSW_INDEX_SQL)
    conn.commit()
    cursor.close()
    conn.close()
    print("[DATABASE] pgvector initialized cleanly. Unified table ready.")


# ── Helpers ────────────────────────────────────────────────────────────────


def _coerce_face_encoding(face_encoding: np.ndarray | list[float] | None):
    if face_encoding is None:
        return None
    if isinstance(face_encoding, np.ndarray):
        return face_encoding.tolist()
    return list(face_encoding)


def _hamming_distance(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


# ── Write ──────────────────────────────────────────────────────────────────


def save_image_metadata(
    user_id: str,
    phash: str,
    watermark_id: str,
    face_encoding: np.ndarray | list[float] | None,
    armored_image_path: str | None = None,
    validation_passed: bool = True,
    compression_quality_tested: int | None = None,
):
    conn = get_db_connection()
    try:
        return save_protected_image(
            conn,
            user_id=user_id,
            phash=phash,
            watermark_id=watermark_id,
            face_vector=face_encoding,
            armored_image_path=armored_image_path,
            validation_passed=validation_passed,
            compression_quality_tested=compression_quality_tested,
        )
    finally:
        conn.close()


def save_protected_image(
    db,
    user_id: str,
    phash: str,
    watermark_id: str,
    face_vector: np.ndarray | list[float] | None = None,
    armored_image_path: str | None = None,
    validation_passed: bool = True,
    compression_quality_tested: int | None = None,
    commit: bool = True,
):
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO protected_images (
                user_id, phash, watermark_id, face_encoding,
                armored_image_path, validation_passed, compression_quality_tested
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                str(user_id),
                phash,
                watermark_id,
                _coerce_face_encoding(face_vector),
                armored_image_path,
                validation_passed,
                compression_quality_tested,
            ),
        )
        result = cursor.fetchone()
        if result is None:
            if commit:
                db.rollback()
            raise ValueError(f"Failed to insert image metadata for user: {user_id}")
        if commit:
            db.commit()
        new_id = result[0]
        print(f"[DATABASE] Secured image data for user: {user_id} (Row ID: {new_id})")
        return new_id
    except Exception:
        if commit:
            db.rollback()
        raise
    finally:
        cursor.close()


def save_protected_image_hashes(
    db,
    protected_image_id: int,
    phash_entries,
    commit: bool = True,
) -> None:
    rows = []
    seen: set[tuple[str, str]] = set()
    for entry in phash_entries:
        hash_kind = getattr(entry, "hash_kind", None)
        phash = getattr(entry, "phash", None)
        if not hash_kind or not phash:
            continue
        key = (str(hash_kind), str(phash))
        if key in seen:
            continue
        rows.append((protected_image_id, str(hash_kind), str(phash)))
        seen.add(key)

    if not rows:
        return

    cursor = db.cursor()
    try:
        cursor.executemany(
            """
            INSERT INTO protected_image_hashes (protected_image_id, hash_kind, phash)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            rows,
        )
        if commit:
            db.commit()
    except Exception:
        if commit:
            db.rollback()
        raise
    finally:
        cursor.close()


# ── Read ───────────────────────────────────────────────────────────────────


def get_all_phashes(db) -> list[str]:
    cursor = db.cursor()
    try:
        cursor.execute(PHASH_UNION_SQL)
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()


def count_all_phashes(db) -> int:
    cursor = db.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM ({PHASH_UNION_SQL}) AS all_phashes;")
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    finally:
        cursor.close()


def iter_all_phashes(db, batch_size: int = 1000):
    cursor = db.cursor(name="amor_phash_stream")
    cursor.itersize = batch_size
    try:
        cursor.execute(PHASH_UNION_SQL)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield row[0]
    finally:
        cursor.close()


def lookup_phash_global(
    db,
    phash: str,
    threshold: int = PHASH_MATCH_THRESHOLD,
) -> PhashMatch | None:
    return lookup_phash_candidates_global(db, [("full", phash)], threshold=threshold)


def lookup_phash_candidates_global(
    db,
    phash_candidates: list[tuple[str, str]],
    threshold: int = REGION_PHASH_MATCH_THRESHOLD,
) -> PhashMatch | None:
    if not phash_candidates:
        return None

    best_match: PhashMatch | None = None
    for suspect_hash_kind, suspect_phash in phash_candidates:
        match = _lookup_single_phash_candidate(
            db,
            suspect_phash=str(suspect_phash),
            suspect_hash_kind=str(suspect_hash_kind),
            threshold=threshold,
        )
        if match is None:
            continue
        if best_match is None or (match.distance or 65) < (best_match.distance or 65):
            best_match = match
    return best_match


def _lookup_single_phash_candidate(
    db,
    *,
    suspect_phash: str,
    suspect_hash_kind: str,
    threshold: int,
) -> PhashMatch | None:
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            WITH stored_hashes AS (
                SELECT
                    id AS protected_image_id,
                    user_id,
                    phash AS original_phash,
                    watermark_id,
                    armored_image_path,
                    'full' AS hash_kind,
                    phash AS match_phash
                FROM protected_images
                UNION ALL
                SELECT
                    pi.id AS protected_image_id,
                    pi.user_id,
                    pi.phash AS original_phash,
                    pi.watermark_id,
                    pi.armored_image_path,
                    pih.hash_kind,
                    pih.phash AS match_phash
                FROM protected_image_hashes pih
                JOIN protected_images pi ON pi.id = pih.protected_image_id
            )
            SELECT
                user_id,
                original_phash,
                watermark_id,
                armored_image_path,
                hash_kind,
                bit_count((('x' || match_phash)::bit(64)) # (('x' || %s)::bit(64))) AS distance
            FROM stored_hashes
            WHERE bit_count((('x' || match_phash)::bit(64)) # (('x' || %s)::bit(64))) <= %s
            ORDER BY distance ASC
            LIMIT 1;
            """,
            (suspect_phash, suspect_phash, threshold),
        )
        row = cursor.fetchone()
        if row:
            (
                user_id,
                stored_phash,
                watermark_id,
                armored_image_path,
                matched_hash_kind,
                distance,
            ) = row
            return PhashMatch(
                user_id=str(user_id),
                phash=stored_phash,
                watermark_id=watermark_id,
                confidence_score=max(0.0, 1.0 - float(distance) / 64.0),
                armored_image_path=armored_image_path,
                detection_method="phash",
                distance=float(distance),
                suspect_hash_kind=suspect_hash_kind,
                matched_hash_kind=matched_hash_kind,
            )
        return None
    finally:
        cursor.close()


def lookup_face_global(
    db,
    face_encoding: np.ndarray | list[float],
    threshold: float = FACE_MATCH_DISTANCE_THRESHOLD,
) -> PhashMatch | None:
    vector = _coerce_face_encoding(face_encoding)
    if vector is None:
        return None

    cursor = db.cursor()
    try:
        cursor.execute(
            """
            SELECT
                user_id,
                phash,
                watermark_id,
                armored_image_path,
                face_encoding <-> %s AS distance
            FROM protected_images
            WHERE face_encoding IS NOT NULL
              AND face_encoding <-> %s <= %s
            ORDER BY face_encoding <-> %s ASC
            LIMIT 1;
            """,
            (vector, vector, threshold, vector),
        )
        row = cursor.fetchone()
        if row:
            user_id, stored_phash, watermark_id, armored_image_path, distance = row
            return PhashMatch(
                user_id=str(user_id),
                phash=stored_phash,
                watermark_id=watermark_id,
                confidence_score=max(0.0, 1.0 - float(distance) / threshold),
                armored_image_path=armored_image_path,
                detection_method="face",
                distance=float(distance),
            )
        return None
    finally:
        cursor.close()
