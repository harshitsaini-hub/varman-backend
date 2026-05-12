import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STORAGE_DIR = os.getenv("STORAGE_DIR", os.path.join(BASE_DIR, "storage"))
DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "database"))
TEMP_STORAGE_PATH = os.getenv("TEMP_STORAGE_PATH", STORAGE_DIR)

for folder in {STORAGE_DIR, DB_DIR, TEMP_STORAGE_PATH}:
    os.makedirs(folder, exist_ok=True)

SQLITE_DB_PATH = os.path.join(DB_DIR, "amor_metadata.db")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "amor_vectors.index")

POSTGRES_DB = os.getenv("POSTGRES_DB", "amor_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

NOISE_EPSILON = float(os.getenv("NOISE_EPSILON", "0.04"))
WATERMARK_BIT_LENGTH = int(os.getenv("WATERMARK_BIT_LENGTH", "32"))
ARMOR_VALIDATION_MIN_QUALITY = int(os.getenv("ARMOR_VALIDATION_MIN_QUALITY", "75"))

TELEGRAM_DANGER_CHANNELS = tuple(
    channel.strip()
    for channel in os.getenv("TELEGRAM_DANGER_CHANNELS", "").split(",")
    if channel.strip()
)
