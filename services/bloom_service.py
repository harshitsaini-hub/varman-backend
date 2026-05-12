import base64
import datetime
import hashlib
import math

import mmh3
from bitarray import bitarray


def get_daily_salt() -> str:
    today = datetime.date.today().isoformat()
    return hashlib.sha256(today.encode()).hexdigest()[:16]


def salt_phash(phash: str, salt: str) -> str:

    return hashlib.sha256(f"{salt}{phash}".encode()).hexdigest()


class CrossPlatformBloomFilter:
    def __init__(self, capacity: int, error_rate: float):
        self.capacity = max(capacity, 10000)
        self.error_rate = error_rate
        # Calculate optimal size (m) and hash count (k)
        self.num_bits = int(-self.capacity * math.log(self.error_rate) / (math.log(2) ** 2))
        self.num_hashes = int(self.num_bits / self.capacity * math.log(2))

        self.bit_array = bitarray(self.num_bits)
        self.bit_array.setall(0)

    def add(self, item: str):
        for i in range(self.num_hashes):
            # mmh3 uses 'i' as a seed, giving us 'k' different hash functions safely
            digest = mmh3.hash(item, i, signed=False) % self.num_bits
            self.bit_array[digest] = 1

    def export_for_client(self) -> dict:
        """
        Exports the raw bytes (NO PICKLE) and the parameters JS needs to rebuild it.
        """
        raw_bytes = self.bit_array.tobytes()
        b64_string = base64.b64encode(raw_bytes).decode("utf-8")

        return {
            "bloom_b64": b64_string,
            "num_bits": self.num_bits,
            "num_hashes": self.num_hashes,
            "capacity": self.capacity,
        }


def build_global_bloom_filter(all_phashes: list[str]) -> dict:
    """
    Called daily to build the cross-platform filter for the Chrome Extension.
    """
    salt = get_daily_salt()
    bloom = CrossPlatformBloomFilter(capacity=len(all_phashes), error_rate=0.001)

    for phash in all_phashes:
        salted = salt_phash(phash, salt)
        bloom.add(salted)

    return bloom.export_for_client()


def check_phash_in_bloom(phash: str, bloom_data: dict) -> bool:
    """Used strictly for internal backend validation if needed."""
    salt = get_daily_salt()
    salted = salt_phash(phash, salt)

    # Reconstruct from raw bytes
    raw_bytes = base64.b64decode(bloom_data["bloom_b64"].encode("utf-8"))
    ba = bitarray()
    ba.frombytes(raw_bytes)
    # Trim excess bits added during byte conversion
    ba = ba[: bloom_data["num_bits"]]

    num_bits = bloom_data["num_bits"]
    num_hashes = bloom_data["num_hashes"]

    for i in range(num_hashes):
        digest = mmh3.hash(salted, i, signed=False) % num_bits
        if not ba[digest]:
            return False
    return True
