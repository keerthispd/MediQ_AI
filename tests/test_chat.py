import json

import pytest

from backend.routes.api import EMPTY_REPLY
from conftest import get_history, reply_text, stream_events


def test_chat_streams_model_reply_and_saves_history(client, auth, fake_llm):
    events = stream_events(client.post("/api/chat", data={"message": "I have a mild headache"}, headers=auth))
    assert events[0] == {"type": "meta", "redflag": False}
    assert reply_text(events) == "Model reply."

    messages = fake_llm.chats[0]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "I have a mild headache"}

    items = get_history(client, auth)
    assert len(items) == 1
    assert items[0]["assistant_reply"] == "Model reply."
    # Timestamps are labelled as UTC so the browser shows the right local time
    assert items[0]["created_at"].endswith("+00:00")


def test_chat_sends_conversation_history(client, auth, fake_llm):
    history = [
        {"role": "user", "text": "I have a sore throat"},
        {"role": "assistant", "text": "How long have you had it?"},
        {"role": "system", "text": "ignored: unknown role"},
    ]
    client.post("/api/chat", data={"message": "Two days", "history": json.dumps(history)}, headers=auth)
    assert [(m["role"], m["content"]) for m in fake_llm.chats[0][1:]] == [
        ("user", "I have a sore throat"),
        ("assistant", "How long have you had it?"),
        ("user", "Two days"),
    ]


@pytest.mark.parametrize("data", [
    {"message": "hi", "history": "not json"},
    {"message": "   "},
    {"message": "a" * 4001},
])
def test_chat_rejects_bad_input(client, auth, fake_llm, data):
    assert client.post("/api/chat", data=data, headers=auth).status_code == 400
    assert fake_llm.chats == []


@pytest.mark.parametrize("message", ["I want to self harm", "I feel suicidal", "I keep thinking about suicide"])
def test_self_harm_messages_get_crisis_reply(client, auth, fake_llm, message):
    events = stream_events(client.post("/api/chat", data={"message": message}, headers=auth))
    assert events[0] == {"type": "meta", "blocked": True}
    assert "helpline" in reply_text(events)
    assert fake_llm.chats == []
    assert "helpline" in get_history(client, auth)[0]["assistant_reply"]


def test_redflag_shows_warning_then_model_answer(client, auth, fake_llm):
    events = stream_events(client.post("/api/chat", data={"message": "I have sudden chest pains"}, headers=auth))
    assert events[0] == {"type": "meta", "redflag": True}

    text = reply_text(events)
    assert text.startswith("⚠️") and "**chest pain**" in text
    assert text.endswith("\n\nModel reply.")
    # The model is told an emergency warning was already shown
    assert "emergency symptoms (chest pain)" in fake_llm.chats[0][0]["content"]

    item = get_history(client, auth)[0]
    assert item["redflag"] is True
    assert item["redflag_details"] == "chest pain"
    assert item["assistant_reply"] == text


def test_redflag_ignores_words_that_merely_contain_a_phrase(client, auth, fake_llm):
    message = "My shoulder hurts after backstroke practice"
    events = stream_events(client.post("/api/chat", data={"message": message}, headers=auth))
    assert events[0] == {"type": "meta", "redflag": False}


def test_model_failure_is_explained_in_the_reply(client, auth, fake_llm):
    fake_llm.chunks = []
    fake_llm.error = "The local AI model isn't running."
    events = stream_events(client.post("/api/chat", data={"message": "hello"}, headers=auth))
    assert reply_text(events) == "The local AI model isn't running."

    # A failure part-way through keeps what was already written
    fake_llm.chunks = ["Partial answer"]
    events = stream_events(client.post("/api/chat", data={"message": "hello again"}, headers=auth))
    assert reply_text(events) == "Partial answer\n\nThe local AI model isn't running."
    assert get_history(client, auth)[0]["assistant_reply"] == "Partial answer\n\nThe local AI model isn't running."


@pytest.mark.parametrize("chunks", [[], ["\n", "  "]])
def test_empty_model_reply_gets_a_fallback_message(client, auth, fake_llm, chunks):
    fake_llm.chunks = chunks
    events = stream_events(client.post("/api/chat", data={"message": "hello"}, headers=auth))
    assert reply_text(events) == EMPTY_REPLY


def test_status_reports_when_ollama_is_unreachable(client):
    status = client.get("/api/status").json()
    assert status["ollama"] is False
    assert status["model_ready"] is False
