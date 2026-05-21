# Testing Notes

Date: 2026-05-21

Environment observed:

- Working directory: `D:\JavaProjects\AMOR\amor-backend`
- Python: `3.11.9`
- Docker CLI: `29.4.2`
- Docker Compose: `v5.1.3`
- Docker daemon: not running during this pass

## Test Log

| Check | Result | Notes |
| --- | --- | --- |
| `python --version` | Passed | Confirmed Python `3.11.9`, matching `pyproject.toml`. |
| `python -m pytest -q` before venv setup | Failed | Global Python had no `pytest`. |
| `python -m ruff check .` before venv setup | Failed | Global Python had no `ruff`. |
| `python -m venv .venv` | Passed | Created local ignored virtualenv. |
| `.venv\Scripts\python.exe -m pip install -r requirements.txt` | Passed | Full dependency install completed. |
| `.venv\Scripts\python.exe -m pytest -q` before dependency fix | Failed | `httpx` was missing, so `fastapi.testclient.TestClient` could not import. |
| `.venv\Scripts\python.exe -m ruff check .` before fixes | Failed | Import ordering, unused imports, and FastAPI dependency-default lint issues. |
| `.venv\Scripts\python.exe -m pip check` before fixes | Passed | No installed package conflicts. |
| `.venv\Scripts\python.exe -m compileall -q .` | Failed | Only failed inside `.venv` because Torch ships a Python 3.12 syntax test file; not an app source failure. |
| `.venv\Scripts\python.exe -m compileall -q api core services utils tests main.py celery_worker.py config.py db.py` | Passed | App source files compile. |
| `docker compose config` without env vars | Failed | Expected: Compose requires `POSTGRES_PASSWORD` and `AMOR_API_KEY`. |
| `docker compose config` with temporary env vars | Passed | Compose interpolation and YAML are valid. |
| Celery app import smoke | Passed | `celery_worker.app` and `celery_worker.celery_app` both import. |
| `.venv\Scripts\celery.exe -A celery_worker.app report` | Passed | Celery can load the worker app. |
| `.venv\Scripts\python.exe -c "import main; print(main.app.title)"` | Passed | FastAPI app imports and reports `Project AMOR API`. |
| `.venv\Scripts\python.exe -m pip install -r requirements.txt` after dependency fix | Passed | Added dependencies resolve from requirements. |
| `.venv\Scripts\python.exe -m pytest -q` after fixes | Passed | `12 passed`, with two FastAPI `on_event` deprecation warnings. |
| `.venv\Scripts\python.exe -m ruff check .` after fixes | Passed | Lint is clean. |
| `.venv\Scripts\python.exe -m pip check` after fixes | Passed | No broken requirements found. |
| `docker compose config --quiet` with temporary env vars | Passed | Compose config validates silently. |
| `docker info` | Failed | Docker Desktop engine was not running: Docker could not connect to `dockerDesktopLinuxEngine`. |

## Flaws Found And Addressed

- `requirements.txt` missed `httpx`, which blocked FastAPI tests.
- `requirements.txt` missed `PyJWT`, even though `core.security` imports `jwt`.
- Docker worker command pointed to `celery_worker.celery_app`, while the worker originally only exposed `app`.
- JWT ownership checks allowed tokens with no `sub` claim to pass user ownership validation.
- Uploads trusted file extension only and could enqueue invalid image bytes.
- Failed oversized/invalid uploads could leave partial files on disk.
- Celery worker duplicated image-processing logic instead of using `services.image_pipeline.process_image_file`.
- Lint failures existed in imports, unused imports, and FastAPI dependency declarations.

## Remaining Risks

- Full Docker build/container startup was not executed because the Docker daemon was unavailable.
- Live Postgres/Redis integration behavior was not tested in running containers.
- FastAPI emits `on_event` deprecation warnings; migrate startup to a lifespan handler later.
- `redis:alpine` is a mutable tag in Compose. Pinning Redis to a version or digest would improve reproducibility.
- Radar pHash input still relies on database casting for validation; malformed hashes should ideally be rejected by request validation before DB access.
