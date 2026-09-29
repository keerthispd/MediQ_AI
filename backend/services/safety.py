"""Recognising messages that need support rather than a medical answer.

Two kinds of message are held back from the model:

* **Self-harm.** The reply is support and real help: crisis lines, and an invitation to reach out
  to someone. Never a bare refusal - someone who says this has reached out, and the reply is the
  one thing the app can get right for them. Coverage is deliberately broad, because people rarely
  use the word "suicide"; they say "I want to die", "what's the point", "better off without me".
* **Lethality.** Asking how much of a medicine would kill. The app indexes FDA drug labels, so
  without this the answer could come from a real source. Many people asking are in crisis, so the
  reply offers the same support rather than only declining.

An overdose that has already happened is not here - that person needs urgent care, and it is
handled as an emergency in backend/services/redflag.py.
"""
from typing import List, Optional, Tuple

from backend.utils.text_utils import compile_phrases, find_phrases

SELF_HARM_TERMS = [
    # Explicit
    "self-harm",
    "self harm",
    "self injury",
    "self-injure",
    "suicide",
    "suicidal",
    "kill myself",
    "killing myself",
    "harm myself",
    "hurt myself",
    "cut myself",
    "cutting myself",
    "end my life",
    "ending my life",
    "take my own life",
    "taking my own life",
    "end it all",
    # How people much more often put it
    "want to die",
    "wanna die",
    "want to be dead",
    "wish I was dead",
    "wish I were dead",
    "better off dead",
    "better off without me",
    "do not want to live",
    "don't want to live",
    "dont want to live",
    "no reason to live",
    "nothing to live for",
    "no point in living",
    "tired of living",
    "can't go on",
    "cant go on",
    "cannot go on",
    "give up on life",
    "do not want to wake up",
    "don't want to wake up",
    "dont want to wake up",
    "disappear forever",
]

LETHALITY_TERMS = [
    "lethal dose",
    "fatal dose",
    "deadly dose",
    "lethal amount",
    "fatal amount",
    "enough to kill",
    "enough to die",
    "how much would kill",
    "how many would kill",
    "to overdose",
    "kill someone",
    "kill a person",
]

# Someone struggling, but not in crisis: "I've been crying every day", "I can't cope". These are
# answered normally - withholding an answer would be its own kind of unhelpful - but the reply is
# asked to lead with warmth rather than with a clinical description of depression, and it ends
# with somewhere to turn. Phrases are first-person on purpose, so that "what causes anxiety?" is
# treated as the information question it is.
DISTRESS_TERMS = [
    "i feel hopeless",
    "feeling hopeless",
    "i feel worthless",
    "feeling worthless",
    "i feel empty",
    "feeling empty",
    "i feel numb",
    "i feel alone",
    "so lonely",
    "i can't cope",
    "i cant cope",
    "cannot cope",
    "can't take it anymore",
    "cant take it anymore",
    "falling apart",
    "breaking down",
    "mental breakdown",
    "overwhelmed",
    "burnt out",
    "burned out",
    "crying every day",
    "crying all the time",
    "cry myself to sleep",
    "panic attack",
    "panic attacks",
    "i am depressed",
    "i'm depressed",
    "im depressed",
    "i feel depressed",
    "feeling depressed",
    "i feel anxious all the time",
    "no motivation",
    "nothing makes me happy",
    "hate myself",
    "i am a burden",
    "i'm a burden",
]

_SELF_HARM_PATTERNS = compile_phrases(SELF_HARM_TERMS)
_LETHALITY_PATTERNS = compile_phrases(LETHALITY_TERMS)
_DISTRESS_PATTERNS = compile_phrases(DISTRESS_TERMS)

SELF_HARM = "self_harm"
LETHALITY = "lethality"


def detect_distress(message: str) -> List[str]:
    """Phrases suggesting the person is struggling emotionally, for a gentler reply.

    This does not block anything. A message that also matches self-harm is handled by `classify`
    first, which takes precedence.
    """
    return find_phrases(message, _DISTRESS_PATTERNS)


def classify(message: str) -> Tuple[Optional[str], List[str]]:
    """Return (category, matched phrases), or (None, []) for an ordinary message.

    Self-harm is checked first: when a message is both ("I want to die, what is a lethal dose"),
    the person matters more than the question.
    """
    found = find_phrases(message, _SELF_HARM_PATTERNS)
    if found:
        return SELF_HARM, found
    found = find_phrases(message, _LETHALITY_PATTERNS)
    if found:
        return LETHALITY, found
    return None, []


def check_message_for_safety(message: str) -> Tuple[bool, List[str]]:
    """Check message for content that must not be answered normally.

    Returns (is_allowed, found_terms). If is_allowed is False, the caller replies with support
    instead of asking the model.
    """
    category, found = classify(message)
    return category is None, found
