import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before importing modules that read configuration at import time
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.routes.api import router as api_router  # noqa: E402
from backend.routes.auth import router as auth_router  # noqa: E402
from backend.models.models import Base  # noqa: E402
from backend.models.db import engine  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB tables exist
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="AI Medical Assistant", lifespan=lifespan)

# Comma-separated list of allowed origins, e.g. "http://localhost:3000,https://mediq.example.com".
# The React app talks to the API through a same-origin proxy, so this only matters for other clients.
cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api")
app.include_router(auth_router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Medical Assistant backend"}
