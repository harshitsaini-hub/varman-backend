import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

DB_PARAMS = {
    "dbname": "amor_db",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():

    conn = psycopg2.connect(
        dbname=DB_PARAMS["dbname"],
        user=DB_PARAMS["user"],
        password=DB_PARAMS["password"],
        host=DB_PARAMS["host"],
        port=DB_PARAMS["port"]
    )
    
    register_vector(conn)
    return conn

def init_db():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS protected_images (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            phash TEXT NOT NULL,
            watermark_id TEXT NOT NULL,
            face_encoding vector(128),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("[DATABASE] pgvector initialized cleanly. Unified table ready.")

def save_image_metadata(user_id: str, phash: str, watermark_id: str, face_encoding: np.ndarray):

    conn = get_db_connection()
    cursor = conn.cursor()

    encoding_list = face_encoding.tolist()

    cursor.execute("""
        INSERT INTO protected_images (user_id, phash, watermark_id, face_encoding)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """, (user_id, phash, watermark_id, encoding_list))

    result = cursor.fetchone()
    if result is None:
        conn.rollback()
        conn.close()
        cursor.close()
        raise ValueError(f"Failed to insert image metadata for user: {user_id}")

    new_id = result[0]

    conn.commit()
    cursor.close()
    conn.close()

    print(f"[DATABASE] Secured image data for user: {user_id} (Row ID: {new_id})")