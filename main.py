from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.protect import router as protect_router
from services.db_service import init_db

app = FastAPI(title="Project AMOR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(protect_router)


@app.on_event("startup")
def startup_event():
    init_db()
    print("AMOR Database (PostgreSQL + pgvector) initialized.")
