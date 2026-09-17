from typing import Tuple, List

from backend.utils.text_utils import compile_phrases, find_phrases

PROHIBITED_TERMS = [
    "self-harm",
    "suicide",
    "suicidal",
    "kill myself",
    "harm myself",
    "hurt myself",
    "end my life",
    "take my own life",
]

_PROHIBITED_PATTERNS = compile_phrases(PROHIBITED_TERMS)


def check_message_for_safety(message: str) -> Tuple[bool, List[str]]:
    """Check message for prohibited content.

    Returns (is_allowed, found_terms).
    If is_allowed is False, the caller should refuse to provide medical advice and
    escalate appropriately.
    """
    found = find_phrases(message, _PROHIBITED_PATTERNS)
    is_allowed = len(found) == 0
    return is_allowed, found
