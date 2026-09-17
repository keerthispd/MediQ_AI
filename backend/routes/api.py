import json
import logging
from datetime import timezone
from typing import Dict, Iterator, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from backend.models.db import SessionLocal, get_db
from backend.models.models import Interaction, User
from backend.services import llm
from backend.services.auth import get_current_user
from backend.services.documents import DocumentError, check_ranges, limit_text, read_document
from backend.services.safety import check_message_for_safety
from backend.services.redflag import detect_redflags
from backend.utils.file_utils import get_upload_extension, read_upload

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_TURNS = 10
MAX_HISTORY_TURN_CHARS = 6000

CRISIS_REPLY = (
    "It sounds like you may be going through something really painful, and I'm glad you reached out. "
    "I'm not able to help with this topic, but you deserve support from someone who can, right now.\n\n"
    "- If you are in immediate danger, call your local emergency number (for example **112** in India "
    "and Europe, **911** in the US).\n"
    "- In India, call **Tele-MANAS on 14416** (free, 24/7). In the US, call or text **988**.\n"
    "- Find a free helpline in your country at **findahelpline.com**.\n\n"
    "If you can, please also reach out to someone you trust and let them know how you are feeling."
)

EMERGENCY_REPLY = (
    "⚠️ Your message mentions symptoms that may be an emergency (**{details}**). "
    "Please seek immediate medical attention or call emergency services."
)

EMPTY_REPLY = "Sorry, I couldn't generate a response to that. Please try rephrasing your question."

NO_TEXT_REPLY = (
    "I couldn't find any readable text in this file. Please upload a clearer image or a PDF with selectable text."
)

# Chat and upload replies are streamed as newline-delimited JSON events, so text appears as the
# local model writes it:
#   {"type": "meta", "redflag": bool} or {"type": "meta", "blocked": true}   (always first)
#   {"type": "status", "text": "..."}   progress note, not part of the reply
#   {"type": "delta", "text": "..."}    the next piece of the reply
# Endpoints are plain `def` so the blocking model calls run in a worker thread.


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status():
    """Whether the local AI models are installed and reachable."""
    return llm.model_status()


def _event(**data) -> str:
    return json.dumps(data) + "\n"


def _stream(events: Iterator[str]) -> StreamingResponse:
    # no-transform and X-Accel-Buffering stop the dev server's compression and nginx from buffering the stream
    return StreamingResponse(
        events,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def _say(parts: List[str], text: str) -> str:
    """Add text to the reply, starting a new paragraph if something was already sent."""
    if parts:
        text = "\n\n" + text
    parts.append(text)
    return _event(type="delta", text=text)


def _model_reply(messages: List[Dict], max_tokens: int, parts: List[str]) -> Iterator[str]:
    """Stream the model's reply as delta events, collecting the text in `parts`."""
    started = False
    try:
        for text in llm.stream_chat(messages, max_tokens=max_tokens):
            if started:
                parts.append(text)
                yield _event(type="delta", text=text)
            elif text.strip():
                started = True
                yield _say(parts, text.lstrip())
    except llm.LLMError as e:
        logger.warning("Local model request failed: %s", e)
        yield _say(parts, str(e))
        return
    if not started:
        # e.g. a "thinking" model that used its whole output budget on reasoning
        yield _say(parts, EMPTY_REPLY)


def _save_interaction(username: str, message: str, reply: str, redflag: bool = False, details: Optional[str] = None):
    # Uses its own session: a streamed reply finishes after the request's dependencies are cleaned up
    with SessionLocal() as db:
        db.add(Interaction(
            user_message=message,
            assistant_reply=reply,
            redflag=redflag,
            redflag_details=details,
            username=username,
        ))
        db.commit()


def _crisis_response(username: str, message: str, found: List[str]) -> StreamingResponse:
    _save_interaction(username, message, CRISIS_REPLY, details=", ".join(found))
    return _stream(iter([_event(type="meta", blocked=True), _event(type="delta", text=CRISIS_REPLY)]))


def _emergency_details(text: str) -> Optional[str]:
    has_redflag, redflags = detect_redflags(text)
    return ", ".join(phrase for phrase, _ in redflags) if has_redflag else None


def _parse_history(raw: Optional[str]) -> List[Dict[str, str]]:
    """Parse the optional JSON list of earlier turns ({"role", "text"}) sent by the client."""
    if not raw:
        return []
    try:
        turns = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="history must be a JSON list of messages.")
    if not isinstance(turns, list):
        raise HTTPException(status_code=400, detail="history must be a JSON list of messages.")

    history = []
    for turn in turns[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        role, text = turn.get("role"), turn.get("text")
        if role in ("user", "assistant") and isinstance(text, str) and text.strip():
            history.append({"role": role, "text": text[:MAX_HISTORY_TURN_CHARS]})
    return history


def _check_length(text: str):
    if len(text) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message is too long (maximum {MAX_MESSAGE_CHARS} characters).")


@router.post("/chat")
def chat(
    message: str = Form(...),
    history: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    _check_length(message)
    turns = _parse_history(history)
    username = user.username

    # Safety layer: self-harm messages get crisis resources instead of a model reply
    allowed, found_prohibited = check_message_for_safety(message)
    if not allowed:
        return _crisis_response(username, message, found_prohibited)

    # Red-flag detection: emergency symptoms get a warning first, then the model's answer
    details = _emergency_details(message)
    messages = llm.build_chat_messages(message, turns, emergency_details=details)

    def events():
        parts: List[str] = []
        yield _event(type="meta", redflag=details is not None)
        if details:
            yield _say(parts, EMERGENCY_REPLY.format(details=details))
        yield from _model_reply(messages, llm.CHAT_MAX_TOKENS, parts)
        _save_interaction(username, message, "".join(parts), redflag=details is not None, details=details)

    return _stream(events())


@router.post("/upload")
def upload(
    file: UploadFile = File(...),
    message: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    query = (message or "").strip()
    _check_length(query)
    extension = get_upload_extension(file.filename)
    data = read_upload(file)
    try:
        document = read_document(data, extension)
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))

    username = user.username
    user_message = f"[Uploaded File: {file.filename}]" + (f"\nQuery: {query}" if query else "")

    allowed, found_prohibited = check_message_for_safety(query)
    if not allowed:
        return _crisis_response(username, user_message, found_prohibited)

    details = _emergency_details(query)

    def events():
        parts: List[str] = []
        yield _event(type="meta", redflag=details is not None)
        if details:
            yield _say(parts, EMERGENCY_REPLY.format(details=details))

        text = document.text
        if document.images:
            # Scans and photos are read by the vision model first
            yield _event(type="status", text="Reading the document…")
            try:
                text = llm.transcribe_images(document.images)
            except llm.LLMError as e:
                logger.warning("Document transcription failed: %s", e)
                text = None
                yield _say(parts, str(e))

        if text is not None:
            if text.strip():
                yield _event(type="status", text="Analyzing the report…")
                text = limit_text(text)
                messages = llm.build_document_messages(
                    text, query, emergency_details=details, range_checks=check_ranges(text)
                )
                yield from _model_reply(messages, llm.DOCUMENT_MAX_TOKENS, parts)
            else:
                yield _say(parts, NO_TEXT_REPLY)

        _save_interaction(username, user_message, "".join(parts), redflag=details is not None, details=details)

    return _stream(events())


def _isoformat_utc(value):
    if value is None:
        return None
    # SQLite returns naive datetimes; CURRENT_TIMESTAMP is UTC, so label them as UTC
    # so the browser converts them to the user's local time correctly.
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


@router.get("/history")
def history(limit: int = Query(50, ge=1, le=200), user: User = Depends(get_current_user), db=Depends(get_db)):
    """Return the current user's most recent interactions (desc by created_at)."""
    items = (
        db.query(Interaction)
        .filter(Interaction.username == user.username)
        .order_by(Interaction.created_at.desc(), Interaction.id.desc())
        .limit(limit)
        .all()
    )
    results = []
    for it in items:
        results.append({
            "id": it.id,
            "user_message": it.user_message,
            "assistant_reply": it.assistant_reply,
            "redflag": bool(it.redflag),
            "redflag_details": it.redflag_details,
            "created_at": _isoformat_utc(it.created_at),
        })
    return {"items": results}


@router.delete("/history")
def delete_history(user: User = Depends(get_current_user), db=Depends(get_db)):
    db.query(Interaction).filter(Interaction.username == user.username).delete()
    db.commit()
    return {"status": "cleared"}
