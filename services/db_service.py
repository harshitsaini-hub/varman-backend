from dataclasses import dataclass

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)

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


# ── Connection ─────────────────────────────────────────────────────────────


def get_db_connection():
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS protected_images (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            phash TEXT NOT NULL,
            watermark_id TEXT NOT NULL,
            face_encoding vector(128),
            armored_image_path TEXT,
            validation_passed BOOLEAN NOT NULL DEFAULT FALSE,
            compression_quality_tested INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
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
            db.rollback()
            raise ValueError(f"Failed to insert image metadata for user: {user_id}")
        db.commit()
        new_id = result[0]
        print(f"[DATABASE] Secured image data for user: {user_id} (Row ID: {new_id})")
        return new_id
    finally:
        cursor.close()


# ── Read ───────────────────────────────────────────────────────────────────


def get_all_phashes(db) -> list[str]:
    cursor = db.cursor()
    try:
        cursor.execute("SELECT phash FROM protected_images;")
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()


def lookup_phash_global(db, phash: str, threshold: int = 10) -> PhashMatch | None:
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            SELECT user_id, phash, watermark_id
            FROM protected_images
            WHERE bit_count((('x' || phash)::bit(64)) # (('x' || %s)::bit(64))) <= %s
            ORDER BY bit_count((('x' || phash)::bit(64)) # (('x' || %s)::bit(64))) ASC
            LIMIT 1;
            """,
            (phash, threshold, phash),
        )
        row = cursor.fetchone()
        if row:
            user_id, stored_phash, watermark_id = row
            return PhashMatch(
                user_id=str(user_id),
                phash=stored_phash,
                watermark_id=watermark_id,
                confidence_score=0.0,
                armored_image_path=None,
            )
        return None
    finally:
        cursor.close()
