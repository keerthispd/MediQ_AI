"""Local language model access through Ollama (https://ollama.com). No API key is needed."""
import json
import os
import threading
from typing import Dict, Iterator, List, Optional

import httpx

# 127.0.0.1 rather than localhost on purpose: on Windows "localhost" resolves to ::1 first and the
# connection only succeeds after that attempt gives up, which measured ~3x slower per request.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
# Two chat models: a small fast one for everyday chat, and a larger one for the answers that need
# real reasoning (uploaded reports, possible emergencies, and lone symptoms whose causes have to be
# put in order rather than copied out). On a CPU-only machine generation
# speed scales with model size: measured with both resident, an everyday reply lands in under
# 20s and an emergency's background paragraph in 24-31s. Both stay loaded (about 5.3 GB together)
# because swapping between them would cost an emergency as much as a cold start.
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "qwen2.5:1.5b-instruct")
# Possible emergencies go to the larger model - see choose_chat_model. Set this to
# OLLAMA_FAST_MODEL on a machine too slow to carry it.
# `or` rather than a default: Docker passes the variable through whether or not it is set, and an
# empty value must mean "use the default", not "no model".
EMERGENCY_MODEL = os.getenv("OLLAMA_EMERGENCY_MODEL") or CHAT_MODEL
# Symptoms that need the causes put in order rather than copied out - see choose_chat_model.
# Defaults to the same model as an emergency; set it to OLLAMA_FAST_MODEL to trade the ordering
# back for speed on the commonest kind of question there is.
CAUTION_MODEL = os.getenv("OLLAMA_CAUTION_MODEL") or EMERGENCY_MODEL
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct")
# Turns text into vectors for the knowledge index. Small and fast: a query embeds in ~25ms, which
# is nothing next to the time the model spends reading the passages it finds.
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Optional thread-count override. Measured on an i7-1355U it made no difference either way, so the
# default leaves the choice to Ollama; it is here because the best value is machine-specific.
NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "0"))
# Images are occasional, so by default the vision model is unloaded right after use; keeping both
# models in memory can push a 16 GB machine into swapping and make replies many times slower.
VISION_KEEP_ALIVE = os.getenv("OLLAMA_VISION_KEEP_ALIVE", "0")

# On CPU a model can go quiet for a long time while it loads or reads a long prompt
TIMEOUT = httpx.Timeout(300.0, connect=5.0)

# Output caps stop small models from rambling or looping forever. Every generated token costs real
# time on CPU, so the chat cap is sized for the ~120-word target in CHAT_INSTRUCTION plus headroom.
# A small model treats that target loosely - measured replies of 200 and 330 words against a 120
# word ask - and each extra word is about a tenth of a second of waiting. 400 tokens is roughly
# 300 words: over twice what a well-behaved answer needs, but it bounds the worst case.
CHAT_MAX_TOKENS = 400
# An emergency reply is capped far harder than an ordinary one. Everything the person has to act
# on is on screen before the model is asked anything (see backend/services/redflag.py), so what
# follows is background, and background should not keep someone reading while they wait for an
# ambulance. Measured: the 4B model stops on its own at 42-54 words here, so this bounds the bad
# case rather than trimming the normal one.
EMERGENCY_MAX_TOKENS = 120
DOCUMENT_MAX_TOKENS = 1500
TRANSCRIBE_MAX_TOKENS = 1500

CHAT_INSTRUCTION = (
    "You are MediQ, a friendly medical information assistant for patients.\n\n"
    "Guidelines:\n"
    "- Use simple, patient-friendly language and explain any medical terms you use.\n"
    "- Format answers in Markdown with short paragraphs, bullet points, and **bold** key terms. "
    "Be concise: aim for about 120 words, and only go longer if the user asks for more detail. "
    "Lead with the answer; skip preamble like 'I understand you are asking about...'.\n"
    "- Give general educational information: common causes, self-care tips, and clear advice on when "
    "to see a doctor or seek urgent care.\n"
    "- Never invent numbers. No percentages, prevalence figures or study findings unless they are "
    "in the reference passages you were given; say 'common' or 'rare' in words instead.\n"
    "- Never give a definitive diagnosis. Never recommend prescription medicines or doses; for "
    "over-the-counter medicines, tell the user to follow the label or ask a pharmacist.\n"
    "- When the user describes a symptom, remember most symptoms have several possible causes. "
    "List them in order, commonest and least serious first, rarest and most serious last, and say "
    "plainly which are more likely. Never lead with the frightening one.\n"
    "- Give each cause a short practical suggestion on the same line rather than just naming it: "
    "something to try for the everyday ones, and seeing a doctor for anything that needs checking.\n"
    "- Then ask one or two short questions whose answers would narrow it down: other symptoms, how "
    "long it has lasted, how bad it is, what makes it better or worse, age if it matters. Ask "
    "them; do not answer them yourself.\n"
    "- Once they answer, do not send the same list and another round of questions back. Use what "
    "they told you and give the fuller, more specific answer the questions were asked for.\n"
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

GROUNDED_NOTE = (
    "\n\nYou have been given numbered passages from a trusted medical reference. Prefer them over "
    "your own knowledge, and cite the ones you use inline as [1], [2] right after the sentence they "
    "support. Do not cite a passage you did not use. If the passages do not cover what was asked, "
    "say so in one short sentence and then answer normally, without citations.\n"
    "A passage may be written about a particular group - children, teenagers, pregnant or "
    "postpartum women, older adults. Never assume the user belongs to that group. Use the facts, "
    "not the audience: write about the user's own situation unless they said it applies to them."
)

DISTRESS_NOTE = (
    "\n\nThe user is describing their own emotional distress, not asking a question about it. "
    "Open by acknowledging how they feel, in one or two warm, plain sentences, before any "
    "information - do not open with a definition or a list of symptoms. Do not diagnose them, and "
    "do not imply they are being dramatic or that it is minor. Somewhere in the reply, encourage "
    "them gently to talk to a doctor or someone they trust. Keep the whole answer short and kind."
)

CAUTION_NOTE = (
    "\n\nThe user has described a symptom ({details}) that is usually something ordinary and only "
    "occasionally a serious one. Open with the single most ordinary cause of it. Then list the "
    "other causes as short bullets, one line each, the everyday ones first and the serious ones "
    "last, six at most. End every bullet with what to do about that one: something they can try "
    "themselves for the everyday causes, and 'worth getting checked by a doctor' for anything that "
    "needs looking at. A list of names with no suggestion attached is no use to them. Every bullet "
    "looks like this one:\n"
    "- **Blocked nose or a cold** - very common; try steam or a saline spray and give it a few days.\n"
    "Then ask one or two short questions that would tell the causes apart. Under 180 words "
    "altogether.\n"
    "Say nothing about when to get help, and do not mention emergency services, an ambulance, 911 "
    "or 112: a more specific line is added to the end of your reply automatically, and two of them "
    "contradicting each other is worse than one. Do not say whether the user's own case is serious "
    "or harmless - you cannot know that from a sentence."
)

# Once they have answered, the list-and-ask shape must stop. Asking a second round is how a triage
# conversation turns into a treadmill: the user gives more and gets the same menu back, never the
# answer they came for. The questions were asked to earn this reply, so this is where it is paid.
CAUTION_FOLLOWUP_NOTE = (
    "\n\nThis is a follow-up: the user has already been shown the possible causes of their symptom "
    "({details}) and has answered the questions that were put to them.\n"
    "Do not list the causes again, and do not ask another round of questions - at most one, and "
    "only if you genuinely cannot answer without it. Use what they have just told you: say which "
    "cause now fits best and what in their answers points to it, name what else is still possible, "
    "and give them practical detail - what to do about it, what makes it better or worse, how long "
    "it usually takes to settle, and at what point to see a doctor about it. About 200 words.\n"
    "Say nothing about emergency services, an ambulance, 911 or 112: a more specific line is added "
    "to the end of your reply automatically."
)

EMERGENCY_NOTE = (
    "\n\nThe user's message mentions possible emergency symptoms ({details}). They have already been "
    "shown a warning to seek emergency care, and the first-aid steps to follow while they wait.\n"
    "Write nothing they should do. No instructions, no first aid, no telling them to sit, lie down, "
    "stay calm, or take, eat or drink anything - they have that already, and yours would contradict "
    "it. Only two things are yours to write: what these symptoms can mean, and what the medical team "
    "will do when they arrive.\n"
    "This overrides the length guidance above: they are reading while they wait for an ambulance. "
    "Write at most 60 words, as three or four short bullets, and then stop. Never suggest waiting to "
    "see whether it improves, and never say anything that would delay the call."
)

# Used instead of EMERGENCY_NOTE when the alarm came from one vague symptom and nothing else - see
# redflag.ALARM_AND_ASK. The warning and the first-aid steps go out identically either way; what
# changes is that the model spends its few words asking as well as explaining.
EMERGENCY_ASK_NOTE = (
    "\n\nThe user's message mentions possible emergency symptoms ({details}), and nothing else - one "
    "symptom, no detail. They have already been shown a warning to seek emergency care and the "
    "first-aid steps to follow while they wait.\n"
    "Write nothing they should do. No instructions, no first aid, no telling them to sit, lie down, "
    "stay calm, or take, eat or drink anything - they have that already, and yours would contradict "
    "it.\n"
    "Write at most 40 words on what this symptom can mean, the ordinary causes first and the "
    "dangerous one named plainly and last. Then ask exactly two short questions that would tell "
    "those apart - the ones the ambulance call handler will ask them anyway. Do not suggest waiting "
    "for an answer before calling, and never suggest waiting to see whether it improves."
)

NO_TEXT = "NO_TEXT"
TRANSCRIBE_PROMPT = (
    "Transcribe all text in this image of a medical document exactly as written, including test names, "
    "values, units and reference ranges. Put each table row on its own line. Output only the transcribed "
    f"text. If there is no readable text, output {NO_TEXT}."
)


class LLMError(Exception):
    """The local model could not produce a reply. The message is safe to show to users."""


_shared_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def _client(timeout=TIMEOUT) -> httpx.Client:
    """A pooled client, reused across requests.

    Opening a fresh connection to Ollama measured several seconds per call on Windows, which was
    being paid on every single message. Keeping the connection alive removes that entirely.
    Callers must not close it; pass per-request timeouts instead.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        with _client_lock:
            if _shared_client is None or _shared_client.is_closed:
                _shared_client = httpx.Client(base_url=OLLAMA_BASE_URL, timeout=timeout)
    return _shared_client


def embed(texts: List[str], model: Optional[str] = None, timeout: float = 20.0) -> List[List[float]]:
    """Turn text into vectors for the knowledge index."""
    response = _client().post(
        "/api/embed",
        json={"model": model or EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def _options(max_tokens: int) -> Dict:
    options = {"num_ctx": NUM_CTX, "num_predict": max_tokens}
    if NUM_THREAD > 0:
        options["num_thread"] = NUM_THREAD
    return options


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
        "options": _options(max_tokens),
    }
    try:
        with _client().stream("POST", "/api/chat", json=payload) as response:
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
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    emergency_details: Optional[str] = None,
    emergency_ask: bool = False,
    caution_details: Optional[str] = None,
    context: Optional[str] = None,
    distress: bool = False,
) -> List[Dict]:
    """Build the Ollama messages for a chat turn.

    `history` holds earlier turns as {"role": "user"|"assistant", "text": ...}. Consecutive turns from the
    same speaker are merged and leading assistant turns dropped, so the conversation starts with the user
    and alternates roles.

    `context` holds retrieved reference passages. They go in the final user turn rather than the system
    prompt so that the stable system text, and any earlier turns, stay cacheable between questions.
    """
    # An assistant turn already in the history means the causes and the questions have been put to
    # them once. Asking again instead of answering is the failure mode this guards against.
    answered = any(turn["role"] == "assistant" for turn in (history or []))

    system = CHAT_INSTRUCTION
    if context:
        system += GROUNDED_NOTE
    if distress:
        system += DISTRESS_NOTE
    if caution_details:
        system += (CAUTION_FOLLOWUP_NOTE if answered else CAUTION_NOTE).format(details=caution_details)
    if emergency_details:
        note = EMERGENCY_ASK_NOTE if emergency_ask else EMERGENCY_NOTE
        system += note.format(details=emergency_details)
    if context:
        # The citation reminder is repeated here, after the question, because a small model weights
        # the end of the prompt far more than a rule buried in the system text.
        prompt = (
            f"Reference passages:\n{context}\n\nQuestion: {prompt}\n\n"
            "Answer in about 120 words. After each sentence that uses a passage, put its number in "
            'square brackets, like: "Most sore throats are caused by a virus [1]."'
        )
    if caution_details:
        # Same trick as the citation reminder: a small model weights the end of the prompt most,
        # and the order of the answer is the whole point of this one.
        prompt += (
            "\n\nAnswer what they have told you now - no second round of questions."
            if answered else
            "\n\nOrdinary causes first, serious ones last, a suggestion on each, then your questions."
        )
    if emergency_details:
        # Repeated at the very end for the same reason as the citation reminder: a small model
        # weights the last thing it read most heavily. Here each extra word it writes anyway is
        # about a sixth of a second somebody waits for an answer whose important half is already
        # on their screen.
        prompt += (
            "\n\nIn 40 words or fewer: what this can mean, ordinary causes first. "
            "Then exactly two short questions."
            if emergency_ask else
            "\n\nIn 60 words or fewer: what this can mean, and what the medical team will do. "
            "No instructions - they already have those."
        )
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


def choose_chat_model(emergency: bool = False, caution: bool = False) -> str:
    """Pick the chat model for a turn.

    Everyday questions go to the small fast model, which generates about twice as fast on a CPU.
    Emergencies go to the larger one. What the person must act on does not come from either - it is
    fixed text from backend/services/redflag.py, on screen in milliseconds - so the model's share is
    background, and the question is only how accurate that background is. Measured on six samples
    of the same three emergencies, the 1.5B model produced 'they may perform a scope to check the
    heart and lungs' for chest pain; the 4B model named ECG, troponin, thrombolytics and
    angioplasty, correctly, every time. That is worth the seconds, and it costs fewer of them than
    it looks: both models are warmed at startup, and the emergency prompt's prefix is primed there
    too, so a real emergency pays for neither.

    A caution - a lone symptom that is usually ordinary - uses the larger model for a different
    reason: it has to put the possible causes in order, commonest first, and the small model does
    not. Measured on "I have chest pain", the 1.5B model transcribed the retrieved page's own
    list and so opened with angina, wrote its own "When to see a doctor" section next to the
    fixed one, and ran out of output before its questions; ungrounded it ordered them correctly
    but left the serious causes out altogether. The 4B model produced sore muscles, then
    heartburn, then angina, cited, in 90 words, with the questions at the end.
    """
    if emergency:
        return EMERGENCY_MODEL
    if caution:
        return CAUTION_MODEL
    return FAST_MODEL


def emergency_primer() -> List[Dict]:
    """The prompt an emergency sends, for priming the cache with at startup.

    The system text is the same for every emergency up to the symptom names, and Ollama re-reads a
    cached prefix about five times faster than a new one. Measured on the 4B model, priming with
    this took the first emergency of the day from 9.9s to 3.3s before its first word.
    """
    return build_chat_messages("hello", emergency_details="chest pain")


def warmup(model: Optional[str] = None, messages: Optional[List[Dict]] = None) -> None:
    """Load a model into memory so the first real request doesn't pay for it.

    Cold, the first reply waits ~14s for the weights and reads the prompt at ~22 tok/s; warm, the
    same prompt is read at ~900 tok/s. The options must match the ones real requests use, because
    Ollama reloads the model whenever num_ctx or num_thread changes.

    `messages` primes a particular prompt prefix as well as loading the weights; see
    emergency_primer.
    """
    model = model or FAST_MODEL
    payload = {
        "model": model,
        "messages": messages or [{"role": "user", "content": "hi"}],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": _options(1),
    }
    _client().post("/api/chat", json=payload).raise_for_status()


def _with_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def model_status() -> Dict:
    """Report whether Ollama is reachable and the configured models are installed."""
    status = {
        "ollama": False,
        "model": CHAT_MODEL,
        "model_ready": False,
        "fast_model": FAST_MODEL,
        "fast_ready": False,
        "emergency_model": EMERGENCY_MODEL,
        "emergency_ready": False,
        "vision_model": VISION_MODEL,
        "vision_ready": False,
    }
    try:
        response = _client().get("/api/tags", timeout=3.0)
        response.raise_for_status()
        installed = {m["name"] for m in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return status
    status.update(
        ollama=True,
        model_ready=_with_tag(CHAT_MODEL) in installed,
        fast_ready=_with_tag(FAST_MODEL) in installed,
        emergency_ready=_with_tag(EMERGENCY_MODEL) in installed,
        vision_ready=_with_tag(VISION_MODEL) in installed,
    )
    return status
