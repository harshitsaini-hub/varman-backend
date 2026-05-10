import sqlite3
import faiss
import numpy as np
import os
from config import SQLITE_DB_PATH, FAISS_INDEX_PATH

def init_db():

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS protected_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            phash TEXT NOT NULL,
            watermark_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

    if not os.path.exists(FAISS_INDEX_PATH):

        index = faiss.IndexFlatL2(128) 
        index = faiss.IndexIDMap(index)
        faiss.write_index(index, FAISS_INDEX_PATH)

def save_image_metadata(user_id: str, phash: str, watermark_id: str, face_encoding: np.ndarray):

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO protected_images (user_id, phash, watermark_id)
        VALUES (?, ?, ?)
    """, (user_id, phash, watermark_id))
    sqlite_id = cursor.lastrowid
    conn.commit()
    conn.close()

    index = faiss.read_index(FAISS_INDEX_PATH)
    
    vector = np.array([face_encoding]).astype('float32')
    ids = np.array([sqlite_id]).astype('int64')
    
    index.add_with_ids(vector, ids)
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"Secured image data for user: {user_id} (ID: {sqlite_id})")