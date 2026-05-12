import hashlib
import datetime
from pybloom_live import BloomFilter
import base64
import pickle

def get_daily_salt() -> str:
    """
    Salt rotates every 24 hours.
    Same day = same salt = consistent filter for that day's extension checks.
    Next day = different salt = yesterday's filter bits are useless.
    """
    today = datetime.date.today().isoformat()  # e.g., "2025-06-15"
    return hashlib.sha256(today.encode()).hexdigest()[:16]

def salt_phash(phash: str, salt: str) -> str:
    """One-way transform: salt + pHash → salted token. Not reversible."""
    return hashlib.sha256(f"{salt}{phash}".encode()).hexdigest()

def build_global_bloom_filter(all_phashes: list[str]) -> str:
    """
    Called once per day by a scheduled Celery beat task.
    Builds a fresh filter with today's salt applied to every hash.
    Returns base64-encoded filter for the CDN/cache.
    """
    salt = get_daily_salt()
    bloom = BloomFilter(capacity=max(len(all_phashes), 10000), error_rate=0.001)
    
    for phash in all_phashes:
        salted = salt_phash(phash, salt)
        bloom.add(salted)
    
    serialized = pickle.dumps(bloom)
    return base64.b64encode(serialized).decode('utf-8')

def check_phash_in_bloom(phash: str, bloom_b64: str) -> bool:
    """Used by the backend to verify a suspect hash from the extension."""
    salt = get_daily_salt()
    salted = salt_phash(phash, salt)
    bloom_bytes = base64.b64decode(bloom_b64.encode('utf-8'))
    bloom = pickle.loads(bloom_bytes)
    return salted in bloom