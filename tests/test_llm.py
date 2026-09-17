import json

import httpx
import pytest

from backend.services import llm


def use_fake_ollama(monkeypatch, handler):
    """Route the Ollama client through `handler` instead of the network."""
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(
        llm, "_client", lambda timeout=None: httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    )
    return requests


def test_stream_chat_yields_text_as_it_arrives(monkeypatch):
    lines = [
        {"message": {"role": "assistant", "content": "Drink "}, "done": False},
        {"message": {"role": "assistant", "content": "water."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    body = "\n".join(json.dumps(line) for line in lines)
    requests = use_fake_ollama(monkeypatch, lambda request: httpx.Response(200, text=body))

    assert list(llm.stream_chat([{"role": "user", "content": "hi"}], max_tokens=50)) == ["Drink ", "water."]

    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/api/chat"
    assert payload["model"] == llm.CHAT_MODEL
    assert payload["stream"] is True
    assert payload["options"]["num_predict"] == 50
    assert payload["keep_alive"] == llm.KEEP_ALIVE


def test_stream_chat_explains_a_missing_model(monkeypatch):
    error = {"error": 'model "qwen3:4b-instruct" not found, try pulling it first'}
    use_fake_ollama(monkeypatch, lambda request: httpx.Response(404, json=error))
    with pytest.raises(llm.LLMError, match="ollama pull"):
        list(llm.stream_chat([{"role": "user", "content": "hi"}]))


def test_stream_chat_reports_errors_mid_stream(monkeypatch):
    body = json.dumps({"message": {"content": "Hi"}}) + "\n" + json.dumps({"error": "out of memory"})
    use_fake_ollama(monkeypatch, lambda request: httpx.Response(200, text=body))
    with pytest.raises(llm.LLMError, match="out of memory"):
        list(llm.stream_chat([{"role": "user", "content": "hi"}]))


def test_stream_chat_reports_when_ollama_is_not_running():
    # conftest points OLLAMA_BASE_URL at a closed port
    with pytest.raises(llm.LLMError, match="isn't running"):
        list(llm.stream_chat([{"role": "user", "content": "hi"}]))


def test_transcribe_images_uses_the_vision_model_per_page(monkeypatch):
    calls = []

    def fake_stream_chat(messages, model=None, max_tokens=None, keep_alive=None):
        calls.append((model, messages[0]["images"], keep_alive))
        # Page one has no text; page two has text followed by a stray marker
        yield "NO_TEXT" if len(calls) == 1 else "Page two text\nNO_TEXT"

    monkeypatch.setattr(llm, "stream_chat", fake_stream_chat)
    assert llm.transcribe_images(["page1", "page2"]) == "Page two text"
    # The vision model stays loaded between pages and is released after the last one
    assert calls == [
        (llm.VISION_MODEL, ["page1"], None),
        (llm.VISION_MODEL, ["page2"], llm.VISION_KEEP_ALIVE),
    ]


def test_build_chat_messages_alternates_roles():
    history = [
        {"role": "assistant", "text": "leading assistant turn is dropped"},
        {"role": "user", "text": "first"},
        {"role": "user", "text": "second"},
        {"role": "assistant", "text": "answer"},
    ]
    messages = llm.build_chat_messages("follow-up", history)
    assert [(m["role"], m["content"]) for m in messages[1:]] == [
        ("user", "first\n\nsecond"),
        ("assistant", "answer"),
        ("user", "follow-up"),
    ]
    assert messages[0] == {"role": "system", "content": llm.CHAT_INSTRUCTION}


def test_model_status_checks_installed_models(monkeypatch):
    tags = {"models": [{"name": llm.CHAT_MODEL}, {"name": "other:latest"}]}
    use_fake_ollama(monkeypatch, lambda request: httpx.Response(200, json=tags))
    status = llm.model_status()
    assert status["ollama"] is True
    assert status["model_ready"] is True
    assert status["vision_ready"] is False
