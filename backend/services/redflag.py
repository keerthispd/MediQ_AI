from typing import Tuple, List

from backend.utils.text_utils import compile_phrases, find_phrases

# Self-harm phrases are handled by the safety layer (backend/services/safety.py),
# which replies with crisis resources instead of a generic emergency message.
RED_FLAG_KEYWORDS = {
    "chest pain": "emergency",
    "shortness of breath": "emergency",
    "severe bleeding": "emergency",
    "loss of consciousness": "emergency",
    "sudden weakness": "emergency",
    "stroke": "emergency",
    "vision loss": "emergency",
}

_RED_FLAG_PATTERNS = compile_phrases(RED_FLAG_KEYWORDS)


def detect_redflags(message: str) -> Tuple[bool, List[Tuple[str, str]]]:
    """Detect red-flag phrases and return (has_redflag, list of (phrase, severity))."""
    found = [(phrase, RED_FLAG_KEYWORDS[phrase]) for phrase in find_phrases(message, _RED_FLAG_PATTERNS)]
    return (len(found) > 0), found
