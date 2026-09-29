"""The hosted fallback must never answer without the user agreeing to it first."""
import pytest

from backend.services import cloud
from conftest import get_history, reply_text, stream_events

DESCRIBE = {"configured": True, "provider": "anthropic", "model": "claude-opus-5", "label": "Anthropic Claude"}


@pytest.fixture
def offline_local(fake_llm):
    """A local model that fails before producing any text."""
    fake_llm.chunks = []
    fake_llm.error = "The local AI model isn't running."
    return fake_llm


@pytest.fixture
def fake_cloud(monkeypatch):
    """A configured hosted provider that records whether it was actually called."""
    calls = []

    def stream_chat(messages, max_tokens=None, effort=None):
        calls.append(messages)
        yield "Hosted reply."

    monkeypatch.setattr(cloud, "configured", lambda: True)
    monkeypatch.setattr(cloud, "describe", lambda: dict(DESCRIBE))
    monkeypatch.setattr(cloud, "stream_chat", stream_chat)
    return calls


def test_asks_before_sending_anything_off_the_machine(client, auth, offline_local, fake_cloud):
    events = stream_events(client.post("/api/chat", data={"message": "I have a headache"}, headers=auth))

    [consent] = [e for e in events if e["type"] == "consent_required"]
    assert consent["label"] == "Anthropic Claude"
    assert "isn't running" in consent["reason"]
    # The whole point: the question has not been sent anywhere yet
    assert fake_cloud == []


def test_nothing_is_saved_while_consent_is_pending(client, auth, offline_local, fake_cloud):
    stream_events(client.post("/api/chat", data={"message": "I have a headache"}, headers=auth))
    assert get_history(client, auth) == []


def test_uses_the_hosted_model_once_the_user_agrees(client, auth, offline_local, fake_cloud):
    events = stream_events(
        client.post("/api/chat", data={"message": "I have a headache", "allow_cloud": "true"}, headers=auth)
    )

    assert reply_text(events) == "Hosted reply."
    assert [e for e in events if e["type"] == "provider"]
    assert len(fake_cloud) == 1
    # The hosted model gets the same conversation the local one would have had
    assert fake_cloud[0][-1] == {"role": "user", "content": "I have a headache"}
    assert get_history(client, auth)[0]["assistant_reply"] == "Hosted reply."


def test_without_a_configured_provider_the_local_error_is_shown(client, auth, offline_local, monkeypatch):
    monkeypatch.setattr(cloud, "configured", lambda: False)
    events = stream_events(client.post("/api/chat", data={"message": "I have a headache"}, headers=auth))

    assert [e for e in events if e["type"] == "consent_required"] == []
    assert "isn't running" in reply_text(events)


def test_a_half_written_local_reply_is_not_restarted_in_the_cloud(client, auth, fake_llm, fake_cloud):
    # Text already on screen: finish with the error rather than silently re-answering elsewhere
    fake_llm.chunks = ["Partial answer"]
    fake_llm.error = "The AI model stopped with an error: out of memory"

    events = stream_events(client.post("/api/chat", data={"message": "I have a headache"}, headers=auth))

    assert [e for e in events if e["type"] == "consent_required"] == []
    assert "Partial answer" in reply_text(events)
    assert fake_cloud == []


def test_consent_is_per_request_not_remembered_by_the_server(client, auth, offline_local, fake_cloud):
    stream_events(client.post("/api/chat", data={"message": "first", "allow_cloud": "true"}, headers=auth))
    assert len(fake_cloud) == 1

    # A later request without the flag must ask again
    events = stream_events(client.post("/api/chat", data={"message": "second"}, headers=auth))
    assert [e for e in events if e["type"] == "consent_required"]
    assert len(fake_cloud) == 1
