# Varman Backend — Architecture

## Current Architecture (v2 — Semantic Disruption Engine)

```
Varman-backend/
├── app/                              # Core application package
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory, CORS, router registration
│   ├── config.py                     # Pydantic settings — reads .env, single source of truth
│   ├── database.py                   # SQLAlchemy async engine + session factory (SQLite)
│   ├── uploads.py                    # Upload validation — file size, extension, dimensions
│   │
│   ├── auth/                         # Authentication module
│   │   ├── __init__.py
│   │   ├── router.py                 # /auth/register, /auth/login, /auth/refresh endpoints
│   │   ├── schemas.py                # Pydantic request/response models for auth
│   │   └── security.py               # JWT creation, password hashing, get_current_user dep
│   │
│   ├── models/                       # SQLAlchemy ORM models
│   │   ├── __init__.py               # Declarative Base
│   │   ├── user.py                   # User table — email, hashed password, role
│   │   └── protected_image.py        # ProtectedImage table — status, SSIM, PSNR, epsilon
│   │
│   ├── protection/                   # The adversarial engine
│   │   ├── __init__.py
│   │   ├── engine.py                 # CLIP PGD attack — 100-iter optimisation, lossless PNG output
│   │   └── quality.py                # SSIM + PSNR computation via scikit-image
│   │
│   └── routes/                       # API endpoints
│       ├── __init__.py
│       ├── images.py                 # /protect, /status, /download, /list, /delete endpoints
│       ├── analytics.py              # /analytics — aggregated stats for the dashboard
│       └── schemas.py                # Pydantic response models for images + analytics
│
├── benchmarks/
│   └── varman_benchmark.py           # CLI tool — measures CLIP cosine, SSIM, LPIPS between two images
│
├── tests/
│   ├── conftest.py                   # Pytest fixtures — in-memory DB, test client, auth helpers
│   ├── test_auth.py                  # Auth endpoint tests
│   ├── test_images.py                # Image upload/download/delete tests
│   ├── test_analytics.py             # Analytics endpoint tests
│   ├── test_engine.py                # Engine invariant tests — epsilon bound, SSIM, CLIP cosine
│   └── legacy/                       # Preserved tests from AMOR era (not runnable)
│
├── experiments/                      # Archived experiment notes from development
│   ├── baseline_clip_resnet.md       # Early CLIP+ResNet surrogate results
│   └── facenet_whitebox.md           # FaceNet white-box attack results
│
├── docs/
│   ├── ARCHITECTURE.md               # This file
│   ├── TESTING_NOTES.md              # Historical testing methodology
│   └── varman_v1_postmortem.md       # Full retrospective — trilemma, lessons, pivot rationale
│
├── .env                              # Runtime config (epsilon, iterations, device, DB URL)
├── .env.example                      # Template for .env
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Ruff/linter config
├── pytest.ini                        # Pytest config
└── varman.db                         # SQLite database (runtime, gitignored)
```

## Request Flow

```
User uploads image via frontend
        │
        ▼
POST /images/protect  (FastAPI)
        │
        ├── Validates file (size, extension)
        ├── Saves original to storage/{user_id}/
        ├── Creates ProtectedImage row (status=pending)
        └── Fires asyncio.create_task(_run_protection_task)
                │
                ▼
        _run_protection_task()
                │
                ├── Acquires GPU semaphore (max 1 concurrent job)
                ├── Sets status=processing in DB
                └── Runs protect_image_pipeline() in ThreadPoolExecutor
                        │
                        ▼
                engine.py: protect_image_pipeline()
                        │
                        ├── Loads image at native resolution (no downscaling)
                        ├── Loads OpenCLIP ViT-B/32
                        ├── Extracts original CLIP embedding
                        ├── PGD loop (100 iterations):
                        │     ├── Forward pass through CLIP
                        │     ├── Cosine similarity loss
                        │     ├── Signed gradient step (α = 2.5ε/N)
                        │     └── L∞ clamp to epsilon
                        ├── Saves as lossless PNG
                        └── Returns {ssim, psnr, clip_cosine, epsilon, status}
                                │
                                ▼
                        DB updated with metrics (status=completed)
                                │
                                ▼
                GET /images/download/{id}  →  FileResponse (image/png)
```

## Key Design Decisions

| Decision | Rationale |
| :--- | :--- |
| **OpenCLIP only** | Single surrogate isolates variables. Ensemble (CLIP+SigLIP) is a v2 upgrade. |
| **No DiffJPEG** | Compression survival was dropped to maximise invisibility at low epsilon. |
| **No face masking** | MLLMs interpret full scenes. Perturbation applied globally, not just to faces. |
| **Lossless PNG output** | Preserves the delicate adversarial gradients for direct upload to target models. |
| **Dynamic alpha** | `α = 2.5ε/N` ensures proper convergence regardless of epsilon/iteration config. |
| **SQLite** | Lightweight, zero-config. Sufficient for single-user / dev / final year project. |

---
---

## Previous Architecture (v1 — AMOR / Face Protection Era)

> The following documents the original AMOR-based architecture that Varman was
> built on top of. This code has been fully removed, but the architecture is
> preserved here for historical reference and the post-mortem.

```
amor-backend/
├── api/
│   └── routes/
│       └── protect.py           # /protect upload endpoint
├── core/
│   └── config.py                # Env-driven Postgres, Redis, armor config
├── models/
│   └── protected_image.py       # protected_images table DDL
├── services/
│   ├── amor_service.py          # Frequency-domain noise + DWT-DCT watermark
│   ├── bloom_service.py         # Daily salted Bloom filter for pHash matching
│   ├── db_service.py            # PostgreSQL/pgvector connection helpers
│   ├── image_pipeline.py        # Orchestration: pHash → face vector → armor → validate
│   ├── noise_service.py         # Standalone DCT noise helper
│   ├── notification_service.py  # Ops/radar notification adapters
│   └── scrapers/                # External platform monitoring workers
│       ├── danger_zone_coordinator.py
│       ├── fourchan_scraper.py
│       ├── reddit_scraper.py
│       └── telegram_scraper.py
├── utils/
│   ├── armor_validator.py       # JPEG compression/watermark validation
│   ├── face.py                  # face_recognition wrapper
│   └── hashing.py               # pHash helpers
├── celery_worker.py             # Celery app, scheduled Bloom task
├── docker-compose.yml           # Postgres/pgvector + Redis
├── Dockerfile
└── requirements.txt
```

### What Varman v1 Tried and What We Learned

**The Trilemma:** Varman v1 attempted to simultaneously achieve invisible perturbation,
strong adversarial protection, and survival through social media JPEG compression.
This proved to be an irreconcilable constraint under our hardware limits (4GB VRAM).

**Technologies removed during the pivot:**
- `mediapipe` — MTCNN face detection and bounding box masking
- `insightface` / `onnxruntime` — ArcFace IResNet50 surrogate model
- `facenet-pytorch` — FaceNet VGGFace2 surrogate model
- `blind-watermark` — DWT-DCT blind watermarking
- `DiffJPEG` — Differentiable JPEG simulation inside the PGD loop

**Why they were removed:** See [varman_v1_postmortem.md](varman_v1_postmortem.md)
for the full retrospective covering the trilemma, hardware constraints, and the
scientific rationale for the pivot to semantic disruption.
