import re
from typing import Iterable, List, Pattern, Tuple


def compile_phrases(phrases: Iterable[str]) -> List[Tuple[str, Pattern]]:
    """Compile phrases into case-insensitive patterns that match whole words.

    Spaces and hyphens are interchangeable ("self-harm" also matches "self harm"), and the
    match must start at a word boundary so "stroke" does not fire on "backstroke". Suffixes
    are allowed so "chest pain" still matches "chest pains".
    """
    compiled = []
    for phrase in phrases:
        words = re.split(r"[\s-]+", phrase.strip())
        pattern = r"\b" + r"[\s-]+".join(re.escape(w) for w in words)
        compiled.append((phrase, re.compile(pattern, re.IGNORECASE)))
    return compiled


def find_phrases(message: str, compiled: List[Tuple[str, Pattern]]) -> List[str]:
    """Return the phrases whose pattern occurs in `message`."""
    return [phrase for phrase, pattern in compiled if pattern.search(message)]
