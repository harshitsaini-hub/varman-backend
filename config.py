import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DB_DIR = os.path.join(BASE_DIR, "database")

for folder in [STORAGE_DIR, DB_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

SQLITE_DB_PATH = os.path.join(DB_DIR, "amor_metadata.db")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "amor_vectors.index")

NOISE_EPSILON = 0.02
WATERMARK_BIT_LENGTH = 32