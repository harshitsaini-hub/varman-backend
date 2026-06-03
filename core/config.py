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

API_KEY = os.getenv("AMOR_API_KEY", "")
JWT_SECRET = os.getenv("AMOR_JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("AMOR_JWT_ALGORITHM", "HS256")
CORS_ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
)
CORS_ALLOWED_ORIGIN_REGEX = os.getenv(
    "CORS_ALLOWED_ORIGIN_REGEX",
    r"^chrome-extension://.*$",
).strip()

MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_BYTES_PER_FILE = int(os.getenv("MAX_UPLOAD_BYTES_PER_FILE", str(10 * 1024 * 1024)))
ALLOWED_IMAGE_EXTENSIONS = tuple(
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_IMAGE_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
    if ext.strip()
)

NOISE_EPSILON = float(os.getenv("NOISE_EPSILON", "0.04"))
WATERMARK_BIT_LENGTH = int(os.getenv("WATERMARK_BIT_LENGTH", "32"))
ARMOR_VALIDATION_MIN_QUALITY = int(os.getenv("ARMOR_VALIDATION_MIN_QUALITY", "75"))

PHASH_MATCH_THRESHOLD = int(os.getenv("PHASH_MATCH_THRESHOLD", "10"))
REGION_PHASH_MATCH_THRESHOLD = int(os.getenv("REGION_PHASH_MATCH_THRESHOLD", "12"))
FACE_MATCH_DISTANCE_THRESHOLD = float(os.getenv("FACE_MATCH_DISTANCE_THRESHOLD", "0.55"))

TELEGRAM_DANGER_CHANNELS = tuple(
    channel.strip()
    for channel in os.getenv("TELEGRAM_DANGER_CHANNELS", "").split(",")
    if channel.strip()
)
