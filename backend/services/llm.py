"""Local language model access through Ollama (https://ollama.com). No API key is needed."""
import json
import os
from typing import Dict, Iterator, List, Optional

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Images are occasional, so by default the vision model is unloaded right after use; keeping both
# models in memory can push a 16 GB machine into swapping and make replies many times slower.
VISION_KEEP_ALIVE = os.getenv("OLLAMA_VISION_KEEP_ALIVE", "0")

# On CPU a model can go quiet for a long time while it loads or reads a long prompt
TIMEOUT = httpx.Timeout(300.0, connect=5.0)

# Output caps stop small models from rambling or looping forever
CHAT_MAX_TOKENS = 1024
DOCUMENT_MAX_TOKENS = 1500
TRANSCRIBE_MAX_TOKENS = 1500

CHAT_INSTRUCTION = (
    "You are MediQ, a friendly medical information assistant for patients.\n\n"
    "Guidelines:\n"
    "- Use simple, patient-friendly language and explain any medical terms you use.\n"
    "- Format answers in Markdown with short paragraphs, bullet points, and **bold** key terms. "
    "Keep answers under about 250 words unless the user asks for more detail.\n"
    "- Give general educational information: common causes, self-care tips, and clear advice on when "
    "to see a doctor or seek urgent care.\n"
    "- Never give a definitive diagnosis. Never recommend prescription medicines or doses; for "
    "over-the-counter medicines, tell the user to follow the label or ask a pharmacist.\n"
    "- If important details are missing (age, how long, how severe, other symptoms), ask one or two "
    "short follow-up questions.\n"
    "- If symptoms could be an emergency, tell the user to contact emergency services first.\n"
    "- Use the earlier conversation for context when answering follow-up questions."
)

DOCUMENT_INSTRUCTION = (
    "You are MediQ, a medical information assistant. You will be given the text of a patient's medical "
    "document (such as a lab report or prescription) and possibly a question about it.\n\n"
    "Using only the information in the document, reply in Markdown with these sections:\n"
    "**Summary** - what the document shows, in plain language.\n"
    "**Key Findings** - values outside the normal range (include the reference range when the document "
    "gives one) and what each means.\n"
    "**Potential Causes** - common reasons for any abnormal findings.\n"
    "**Tips** - general tips for managing the findings.\n"
    "**When to See a Doctor** - findings that need prompt medical follow-up.\n"
    "If the user asked a question, answer it at the end.\n\n"
    "Before deciding whether a result is normal, compare its value with its reference range: a value "
    "above the upper limit is high and a value below the lower limit is low, even if only slightly. "
    "Never invent values that are not in the document. If the text is not a medical document or is "
    "unreadable, say so briefly. This is educational information, not a diagnosis."
)

EMERGENCY_NOTE = (
    "\n\nThe user's message mentions possible emergency symptoms ({details}). They have already been "
    "shown a warning to seek emergency care. Start by briefly repeating that they should get urgent "
    "medical help now, then give short, helpful information. Never suggest waiting to see if it improves."
)

NO_TEXT = "NO_TEXT"
TRANSCRIBE_PROMPT = (
    "Transcribe all text in this image of a medical document exactly as written, including test names, "
    "values, units and reference ranges. Put each table row on its own line. Output only the transcribed "
    f"text. If there is no readable text, output {NO_TEXT}."
)


class LLMError(Exception):
    """The local model could not produce a reply. The message is safe to show to users."""


def _client(timeout=TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=OLLAMA_BASE_URL, timeout=timeout)


def _describe_error(model: str, response: httpx.Response) -> str:
    try:
        detail = response.json().get("error", "")
    except ValueError:
        detail = response.text
    if response.status_code == 404 and "not found" in detail:
        return (
            f"The AI model '{model}' isn't installed. If the app was just started it may still be "
            f"downloading; otherwise run: `ollama pull {model}`"
        )
    return f"The AI model returned an error ({response.status_code}): {detail}"


def stream_chat(
    messages: List[Dict],
    model: Optional[str] = None,
    max_tokens: int = CHAT_MAX_TOKENS,
    keep_alive: Optional[str] = None,
) -> Iterator[str]:
    """Send a chat to Ollama and yield the reply text as it is generated."""
    model = model or CHAT_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": keep_alive or KEEP_ALIVE,
        "options": {"num_ctx": NUM_CTX, "num_predict": max_tokens},
    }
    try:
        with _client() as client, client.stream("POST", "/api/chat", json=payload) as response:
            if response.status_code != 200:
                response.read()
                raise LLMError(_describe_error(model, response))
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(f"The AI model stopped with an error: {chunk['error']}")
                text = chunk.get("message", {}).get("content")
                if text:
                    yield text
    except httpx.ConnectError as e:
        raise LLMError(
            f"The local AI model isn't running. Start Ollama (expected at {OLLAMA_BASE_URL}) and try again."
        ) from e
    except httpx.TimeoutException as e:
        raise LLMError("The AI model took too long to respond. Please try again.") from e
    except (httpx.HTTPError, ValueError) as e:
        raise LLMError(f"Could not talk to the AI model: {e}") from e


def build_chat_messages(
    prompt: str, history: Optional[List[Dict[str, str]]] = None, emergency_details: Optional[str] = None
) -> List[Dict]:
    """Build the Ollama messages for a chat turn.

    `history` holds earlier turns as {"role": "user"|"assistant", "text": ...}. Consecutive turns from the
    same speaker are merged and leading assistant turns dropped, so the conversation starts with the user
    and alternates roles.
    """
    system = CHAT_INSTRUCTION
    if emergency_details:
        system += EMERGENCY_NOTE.format(details=emergency_details)
    messages = [{"role": "system", "content": system}]
    for turn in [*(history or []), {"role": "user", "text": prompt}]:
        role = "assistant" if turn["role"] == "assistant" else "user"
        if role == "assistant" and len(messages) == 1:
            continue
        if messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + turn["text"]
        else:
            messages.append({"role": role, "content": turn["text"]})
    return messages


def build_document_messages(
    document_text: str,
    question: str = "",
    emergency_details: Optional[str] = None,
    range_checks: Optional[List[str]] = None,
) -> List[Dict]:
    system = DOCUMENT_INSTRUCTION
    if emergency_details:
        system += EMERGENCY_NOTE.format(details=emergency_details)
    checks = ""
    if range_checks:
        lines = "\n".join(f"- {check}" for check in range_checks)
        checks = f"Automatic reference-range check (these comparisons are correct, use them):\n{lines}\n\n"
    user = (
        f"Medical document text:\n<document>\n{document_text}\n</document>\n\n"
        f"{checks}{question or 'Please analyze this document.'}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def transcribe_images(images: List[str]) -> str:
    """Read the text out of document images (base64 JPEG) with the vision model, one page at a time."""
    pages = []
    for index, image in enumerate(images):
        messages = [{"role": "user", "content": TRANSCRIBE_PROMPT, "images": [image]}]
        # Keep the model loaded between pages, then release it after the last one
        keep_alive = VISION_KEEP_ALIVE if index == len(images) - 1 else None
        text = "".join(stream_chat(messages, model=VISION_MODEL, max_tokens=TRANSCRIBE_MAX_TOKENS, keep_alive=keep_alive))
        # Small models sometimes add the marker after real text, so drop it wherever it appears
        text = "\n".join(line for line in text.splitlines() if line.strip() != NO_TEXT).strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _with_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def model_status() -> Dict:
    """Report whether Ollama is reachable and the configured models are installed."""
    status = {"ollama": False, "model": CHAT_MODEL, "model_ready": False, "vision_model": VISION_MODEL, "vision_ready": False}
    try:
        with _client(timeout=3.0) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            installed = {m["name"] for m in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return status
    status.update(
        ollama=True,
        model_ready=_with_tag(CHAT_MODEL) in installed,
        vision_ready=_with_tag(VISION_MODEL) in installed,
    )
    return status
