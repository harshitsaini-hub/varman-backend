# Docker Usage

AMOR already has a `Dockerfile` and `docker-compose.yml` for the API, Celery worker,
Celery beat scheduler, Postgres with pgvector, and Redis.

## 1. Start Docker Desktop

On Windows, Docker Desktop must be running before `docker compose up`, `docker ps`, or
`docker build` can talk to the Docker engine.

## 2. Create your local environment file

Copy `.env.example` to `.env`, then replace the placeholder secrets:

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum required values:

```dotenv
POSTGRES_PASSWORD=your-strong-db-password
AMOR_API_KEY=your-private-api-key
CORS_ALLOWED_ORIGINS=http://localhost:3000
# Optional for AMOR Radar Chrome extension requests:
CORS_ALLOWED_ORIGIN_REGEX=^chrome-extension://.*$
```

The Compose file intentionally refuses to start without `POSTGRES_PASSWORD` and
`AMOR_API_KEY`.

## 3. Validate Compose config

```powershell
docker compose config
```

This catches missing env vars and YAML mistakes before building containers.

## 4. Build and run AMOR

```powershell
docker compose up --build
```

API: `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs`

The API, worker, and beat scheduler all wait for healthy Postgres and Redis before starting.

## 5. Useful commands

```powershell
docker compose ps
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f beat
docker compose up -d db redis
docker compose down
```

To remove Postgres and storage volumes as well:

```powershell
docker compose down -v
```

Only use `down -v` when you are okay deleting local AMOR database/storage data.

## 6. Auth when calling the API

For service/API-key calls, include:

```http
X-API-Key: your-private-api-key
```

For JWT calls, set `AMOR_JWT_SECRET` in `.env` and send:

```http
Authorization: Bearer <token-with-sub>
```

JWT-protected user operations require the token `sub` to match the requested `user_id`.
