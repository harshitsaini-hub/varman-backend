# AMOR Backend Architecture

This repository is organized around a small FastAPI application plus background
workers that apply image protection, persist metadata, and watch external
platforms for pHash matches.

## Top-level layout

```text
amor-backend/
├── api/                         # FastAPI HTTP route modules
│   └── routes/
│       └── protect.py           # /protect upload endpoint and API background hook
├── core/                        # Core runtime configuration and shared settings
│   └── config.py                # Env-driven storage, Postgres, Redis, armor, Telegram config
├── models/                      # Database model/schema declarations
│   └── protected_image.py       # protected_images table DDL
├── services/                    # Business logic, AI/image pipeline, persistence helpers
│   ├── amor_service.py          # Frequency-domain noise + watermark pipeline
│   ├── bloom_service.py         # Daily salted Bloom filter generation/checking
│   ├── db_service.py            # PostgreSQL/pgvector connection and query helpers
│   ├── image_pipeline.py        # Shared API/Celery image protection workflow
│   ├── noise_service.py         # Standalone DCT noise helper
│   ├── notification_service.py  # Ops/radar notification adapters
│   └── scrapers/                # External platform monitoring workers
│       ├── danger_zone_coordinator.py
│       ├── fourchan_scraper.py
│       ├── reddit_scraper.py
│       └── telegram_scraper.py
├── utils/                       # Reusable helpers and validation utilities
│   ├── armor_validator.py       # JPEG compression/watermark validation helper
│   ├── face.py                  # Optional face_recognition wrapper
│   └── hashing.py               # pHash helpers for arrays, bytes, and URLs
├── tests/                       # Pytest-based verification
│   └── test_project_integrity.py
├── storage/                     # Runtime temporary/armored image output (ignored by Docker)
├── database/                    # Runtime local DB/index files if used (ignored by Docker)
├── celery_worker.py             # Celery app, scheduled Bloom task, image task entrypoint
├── config.py                    # Backward-compatible shim that re-exports core.config
├── db.py                        # Backward-compatible DB session helper for workers
├── main.py                      # FastAPI app factory surface and router registration
├── docker-compose.yml           # Postgres/pgvector + Redis support services
├── Dockerfile                   # API container image
└── requirements.txt             # Python dependencies
```

## Runtime flow

1. `main.py` creates the FastAPI app, installs CORS middleware, and registers
   `api.routes.protect.router`.
2. `POST /protect` in `api/routes/protect.py` stores uploads in `STORAGE_DIR`
   and schedules `process_image_background`.
3. The API background task and `celery_worker.process_image` both call
   `services.image_pipeline.process_image_file`, which is the single source of
   truth for image processing.
4. `process_image_file` loads the image, computes a pHash, optionally extracts
   a face vector, applies armor through `services.amor_service`, validates the
   watermark, writes the armored image, and persists metadata through
   `services.db_service`.
5. The Celery path queues `rebuild_global_bloom`, which reads all pHashes from
   PostgreSQL through a streaming cursor, builds a salted Bloom filter with
   `services.bloom_service`, and caches it in Redis.
6. Scrapers under `services/scrapers/` send external media through
   `services.detection_service`, which tries pHash, region pHash, and face-vector
   matching before queueing alerts through Celery.
7. Scraper watcher events use per-event DB sessions when launched with a DB
   factory, avoiding shared PostgreSQL connections across long-lived threads.

## Screenshot and manipulation detection

AMOR treats the watermark as proof, not as the only detector. When an armored
image is protected, the backend now stores whole-image and center-crop pHash
variants in `protected_image_hashes`, alongside the original pHash and optional
face vector in `protected_images`.

When a scraper sees a suspect image, `services.detection_service` runs a staged
match:

1. Whole-image pHash match for reposts and light edits.
2. Region pHash match for screenshots, crops, added borders, captions, and UI
   pixels around the stolen image.
3. Face-vector match for heavier edits where the person is still recognizable
   but the image hash has drifted too far.

This does not make AMOR impossible to evade. It raises the attacker cost: simple
screenshots should still be caught by hash variants, and face-preserving edits
can still alert through pgvector even when the watermark or pHash is damaged.

The `/api/radar/flag` endpoint also accepts optional `candidate_hashes` from the
Chrome extension. That lets the client compute crop hashes locally and send only
hashes to the backend, preserving the silent-flare privacy model.

## Configuration

All new code imports settings from `core.config`. The root `config.py` remains
as a compatibility shim for older scripts and deployments. Important environment
variables include:

- `STORAGE_DIR`, `DB_DIR`, `TEMP_STORAGE_PATH`
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `REDIS_URL`
- `NOISE_EPSILON`, `WATERMARK_BIT_LENGTH`, `ARMOR_VALIDATION_MIN_QUALITY`
- `PHASH_MATCH_THRESHOLD`, `REGION_PHASH_MATCH_THRESHOLD`, `FACE_MATCH_DISTANCE_THRESHOLD`
- `TELEGRAM_DANGER_CHANNELS` as a comma-separated list of channel usernames/IDs

## Deployment shape

`docker-compose.yml` defines the supporting services used by the backend:

- `db`: PostgreSQL with pgvector.
- `redis`: Celery broker/result backend and Bloom filter cache.
- `worker`: Celery worker for image processing, Bloom rebuilds, and queued alerts.
- `beat`: Celery beat scheduler for the daily Bloom rebuild.

The API itself is still launched by the existing `Dockerfile`/Uvicorn command path.
For production, scraper watchers should run outside the API container with
platform-specific rate limits and backoff.
