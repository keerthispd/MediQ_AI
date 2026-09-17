import itertools
import json
import os
import tempfile

# Configure the app before `backend` is imported: use a throwaway database, and point the model client
# at a closed port so no test can reach a real Ollama server by accident.
_tmp_dir = tempfile.mkdtemp(prefix="mediq-tests-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp_dir, "test.db")
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:9"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from backend.services import llm  # noqa: E402

PASSWORD = "correct-horse-battery"
_user_ids = itertools.count()


@pytest.fixture
def client():
    # Using the client as a context manager runs the lifespan handler, which creates the tables
    with TestClient(app) as c:
        yield c


def register(client, username=None):
    """Create a user and return (username, auth headers)."""
    username = username or f"user{next(_user_ids)}"
    response = client.post("/api/auth/register", json={"username": username, "password": PASSWORD})
    assert response.status_code == 201, response.text
    return username, {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture
def auth(client):
    return register(client)[1]


class FakeLLM:
    """Stands in for the Ollama client and records what it was asked."""

    def __init__(self):
        self.chunks = ["Model ", "reply."]
        self.error = None
        self.transcript = "Hemoglobin 10.2 g/dL (reference 13.5-17.5)"
        self.chats = []
        self.transcribed = []

    def stream_chat(self, messages, model=None, max_tokens=None, keep_alive=None):
        self.chats.append(messages)
        yield from self.chunks
        if self.error:
            raise llm.LLMError(self.error)

    def transcribe_images(self, images):
        self.transcribed.append(images)
        return self.transcript


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "stream_chat", fake.stream_chat)
    monkeypatch.setattr(llm, "transcribe_images", fake.transcribe_images)
    return fake


def stream_events(response):
    """Parse a streamed chat/upload response into its list of events."""
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line]


def reply_text(events):
    return "".join(e["text"] for e in events if e["type"] == "delta")


def get_history(client, headers):
    response = client.get("/api/history", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["items"]
