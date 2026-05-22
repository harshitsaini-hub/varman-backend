CREATE_PROTECTED_IMAGES_TABLE_SQL = """
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

CREATE_PROTECTED_IMAGE_HASHES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS protected_image_hashes (
    id SERIAL PRIMARY KEY,
    protected_image_id INTEGER NOT NULL REFERENCES protected_images(id) ON DELETE CASCADE,
    hash_kind TEXT NOT NULL,
    phash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (protected_image_id, hash_kind, phash)
);
"""

CREATE_PROTECTED_IMAGES_PHASH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS protected_images_phash_idx
ON protected_images (phash);
"""

CREATE_PROTECTED_IMAGE_HASHES_PHASH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS protected_image_hashes_phash_idx
ON protected_image_hashes (phash);
"""

CREATE_PROTECTED_IMAGE_HASHES_IMAGE_ID_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS protected_image_hashes_image_id_idx
ON protected_image_hashes (protected_image_id);
"""

CREATE_PROTECTED_IMAGES_FACE_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS protected_images_face_hnsw_idx
ON protected_images USING hnsw (face_encoding vector_l2_ops)
WHERE face_encoding IS NOT NULL;
"""
