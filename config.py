import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/amor_db")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6373/0")

NOISE_EPSILON = 0.02
WATERMARK_BIT_LENGTH = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)