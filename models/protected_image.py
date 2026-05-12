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
