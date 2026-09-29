import logging
import os
import threading
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
from backend.services import llm  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _warm_models():
    """Load the chat models so the first user message doesn't wait for them.

    The emergency model goes first, and is primed with the prompt an emergency actually sends
    rather than a bare "hi": a cold model costs about 14 seconds before it writes anything and an
    unread prompt several more, and those are the seconds a message about chest pain must never
    pay. Both models are loaded, because chat and emergencies use different ones and a swap
    between them costs as much as a cold start.
    """
    for model in dict.fromkeys([llm.EMERGENCY_MODEL, llm.FAST_MODEL]):
        primer = llm.emergency_primer() if model == llm.EMERGENCY_MODEL else None
        try:
            llm.warmup(model, messages=primer)
            logger.info("Warmed up chat model %s", model)
        except Exception as e:  # noqa: BLE001 - warmup is best effort, the app works without it
            logger.warning("Could not warm up the chat model %s: %s", model, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB tables exist
    Base.metadata.create_all(bind=engine)
    # In a background thread so startup isn't blocked while the weights load
    if os.environ.get("OLLAMA_WARMUP", "1") != "0":
        threading.Thread(target=_warm_models, daemon=True).start()
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
