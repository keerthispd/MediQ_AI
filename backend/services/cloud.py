"""Optional hosted-API fallback for when the local model cannot be used.

This is the one part of the app that sends a user's words off the machine, so nothing here runs
without the user agreeing first: the routes ask for consent and only then call `stream_chat`.

Two providers are supported:

* ``anthropic`` - the Claude Messages API through the official SDK (``pip install anthropic``).
* ``openai``    - any endpoint that speaks OpenAI's ``/v1/chat/completions``, which covers OpenAI
                  itself as well as Groq, OpenRouter, Together, vLLM and LM Studio.
"""
import json
import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Empty disables the fallback entirely. Left unset, a configured API key picks the provider.
PROVIDER = os.getenv("CLOUD_PROVIDER", "").strip().lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Hosted models generate quickly, so unlike the local models these are sized to avoid truncating a
# reply rather than to keep it short. The word limit in the prompt still controls the length.
MAX_TOKENS = int(os.getenv("CLOUD_MAX_TOKENS", "2048"))

TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Chat is latency-sensitive and the answers are short, so it does not repay deep reasoning;
# reports are worth more effort because they involve comparing values against reference ranges.
EFFORT_CHAT = "low"
EFFORT_DOCUMENT = "medium"


class CloudError(Exception):
    """The hosted model could not answer. The message is safe to show to users."""


def resolve_provider() -> str:
    """Which provider is configured, or "" when the fallback is switched off."""
    if PROVIDER in ("anthropic", "openai"):
        return PROVIDER
    if PROVIDER:
        logger.warning("Unknown CLOUD_PROVIDER %r; the hosted fallback is disabled.", PROVIDER)
        return ""
    # Nothing chosen explicitly: use whichever key is present
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    return ""


def _model_for(provider: str) -> str:
    return ANTHROPIC_MODEL if provider == "anthropic" else OPENAI_MODEL


def configured() -> bool:
    """Whether a hosted fallback could be used if the user agreed to it."""
    provider = resolve_provider()
    if provider == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    if provider == "openai":
        return bool(OPENAI_API_KEY)
    return False


def describe() -> Dict:
    """What the UI needs to explain the fallback before asking the user to accept it."""
    provider = resolve_provider()
    return {
        "configured": configured(),
        "provider": provider,
        "model": _model_for(provider) if provider else "",
        "label": {"anthropic": "Anthropic Claude", "openai": "OpenAI-compatible API"}.get(provider, ""),
    }


def _split_system(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Pull the leading system turns out, which the Claude API takes as a separate field."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


# --------------------------------------------------------------------------------------- Anthropic

# Refusal fallbacks re-run a declined request on another model inside the same call, which matters
# for medical wording that a safety classifier may decline. It is a beta parameter, so if the
# account cannot use it the first rejection turns it off for the rest of the process.
_use_refusal_fallbacks = os.getenv("ANTHROPIC_REFUSAL_FALLBACK", "1") != "0"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _anthropic_client():
    try:
        import anthropic
    except ImportError as e:
        raise CloudError(
            "The Anthropic library isn't installed. Run: pip install anthropic"
        ) from e
    if not ANTHROPIC_API_KEY:
        raise CloudError("No ANTHROPIC_API_KEY is set, so the hosted model can't be used.")
    return anthropic, anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _anthropic_content(message: Dict) -> List[Dict]:
    """Build Claude content blocks, putting any images before the text as the API docs advise."""
    blocks: List[Dict] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}}
        for image in message.get("images", [])
    ]
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    return blocks


def _stream_anthropic(messages: List[Dict], max_tokens: int, effort: str) -> Iterator[str]:
    global _use_refusal_fallbacks
    anthropic, client = _anthropic_client()
    system, turns = _split_system(messages)
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": m["role"], "content": _anthropic_content(m)} for m in turns],
        "output_config": {"effort": effort},
    }
    if system:
        payload["system"] = system

    try:
        if _use_refusal_fallbacks:
            sent_any = False
            try:
                with client.beta.messages.stream(
                    betas=[FALLBACK_BETA], fallbacks="default", **payload
                ) as stream:
                    for text in stream.text_stream:
                        sent_any = True
                        yield text
                return
            except anthropic.BadRequestError as e:
                # Only safe to retry while nothing has been shown to the user yet
                if sent_any:
                    raise
                _use_refusal_fallbacks = False
                logger.info("Refusal fallbacks unavailable on this account (%s); continuing without.", e)

        with client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.AuthenticationError as e:
        raise CloudError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from e
    except anthropic.NotFoundError as e:
        raise CloudError(f"The model '{ANTHROPIC_MODEL}' isn't available on this account.") from e
    except anthropic.RateLimitError as e:
        raise CloudError("The hosted model is rate limited right now. Please try again shortly.") from e
    except anthropic.APIStatusError as e:
        raise CloudError(f"The hosted model returned an error ({e.status_code}).") from e
    except anthropic.APIConnectionError as e:
        raise CloudError("Could not reach the hosted model. Check your internet connection.") from e


# ---------------------------------------------------------------------------- OpenAI-compatible API


def _openai_content(message: Dict):
    """OpenAI takes a plain string unless the turn carries images."""
    images = message.get("images", [])
    if not images:
        return message.get("content", "")
    parts = [{"type": "text", "text": message.get("content", "")}] if message.get("content") else []
    parts += [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        for image in images
    ]
    return parts


def _stream_openai(messages: List[Dict], max_tokens: int) -> Iterator[str]:
    if not OPENAI_API_KEY:
        raise CloudError("No OPENAI_API_KEY is set, so the hosted model can't be used.")
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": m["role"], "content": _openai_content(m)} for m in messages],
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        with httpx.Client(base_url=OPENAI_BASE_URL, timeout=TIMEOUT) as client:
            with client.stream("POST", "/chat/completions", json=payload, headers=headers) as response:
                if response.status_code != 200:
                    response.read()
                    raise CloudError(_openai_error(response))
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        choices = json.loads(data).get("choices") or [{}]
                    except ValueError:
                        continue
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        yield text
    except httpx.TimeoutException as e:
        raise CloudError("The hosted model took too long to respond. Please try again.") from e
    except httpx.HTTPError as e:
        raise CloudError(f"Could not reach the hosted model: {e}") from e


def _openai_error(response: httpx.Response) -> str:
    try:
        detail = (response.json().get("error") or {}).get("message", "")
    except ValueError:
        detail = response.text[:200]
    if response.status_code == 401:
        return "The API key was rejected. Check OPENAI_API_KEY."
    if response.status_code == 429:
        return "The hosted model is rate limited right now. Please try again shortly."
    return f"The hosted model returned an error ({response.status_code}): {detail}"


# ------------------------------------------------------------------------------------------ Public


def stream_chat(messages: List[Dict], max_tokens: int = MAX_TOKENS, effort: str = EFFORT_CHAT) -> Iterator[str]:
    """Stream a reply from the configured hosted API. Only call this once the user has agreed."""
    provider = resolve_provider()
    if provider == "anthropic":
        yield from _stream_anthropic(messages, max_tokens, effort)
    elif provider == "openai":
        yield from _stream_openai(messages, max_tokens)
    else:
        raise CloudError("No hosted model is configured.")


def transcribe_images(images: List[str], prompt: str, no_text_marker: str) -> str:
    """Read the text out of document images using the hosted model's vision support."""
    pages = []
    for image in images:
        messages = [{"role": "user", "content": prompt, "images": [image]}]
        text = "".join(stream_chat(messages, effort=EFFORT_DOCUMENT))
        text = "\n".join(line for line in text.splitlines() if line.strip() != no_text_marker).strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)
