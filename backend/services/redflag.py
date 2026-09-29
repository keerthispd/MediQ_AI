"""Emergency red flags, and the fixed first-aid steps that go with them.

Detection is a phrase match rather than a model call, so an emergency is recognised in well under
a millisecond and the warning reaches the screen before anything else happens.

The steps in `GUIDANCE` are written here rather than generated. On a CPU-only machine the local
model needs tens of seconds to produce its first paragraph, and this is the content that cannot
wait for it: someone with crushing chest pain needs "call an ambulance, sit down, do not drive"
now, not in forty seconds, and the wording must not depend on what a 1.5B model decides to say.
The model still answers afterwards - it adds background, not instructions.

**The phrase list is tuned for recall, not precision.** Nobody having a stroke types "I am
experiencing sudden unilateral weakness"; they type "my face is drooping" and "I can't lift my
arm". An earlier version of this file held fourteen textbook phrases and missed every one of
thirteen realistic emergencies tried against it - no warning, no steps, and the message answered
as an ordinary question. The cost of the two errors is not symmetric: a false positive shows
someone a caution box they did not need, and a false negative leaves someone having a heart attack
reading a paragraph about indigestion. So "what are the symptoms of a heart attack" trips this
list, and that is the intended trade.

Self-harm phrases are handled by the safety layer (backend/services/safety.py), which replies with
crisis resources instead of a generic emergency message.
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple

from backend.utils.text_utils import compile_phrases, find_phrases

# Categories, most immediately life-threatening first. The order is what decides which steps are
# shown when a message matches more than one: airway and circulation before everything else.
UNCONSCIOUS = "unconscious"
CHOKING = "choking"
ANAPHYLAXIS = "anaphylaxis"
BLEEDING = "bleeding"
BREATHING = "breathing"
SEIZURE = "seizure"
CARDIAC = "cardiac"
STROKE = "stroke"
OBSTETRIC = "obstetric"
POISONING = "poisoning"

CATEGORY_ORDER = [
    UNCONSCIOUS, CHOKING, ANAPHYLAXIS, BLEEDING, BREATHING, SEIZURE, CARDIAC, STROKE,
    OBSTETRIC, POISONING,
]

# Phrases match whole words, treat spaces and hyphens alike, and allow suffixes, so "chest pain"
# also catches "chest pains" and "faint" also catches "fainted" and "fainting". See
# backend/utils/text_utils.py.
RED_FLAG_KEYWORDS = {
    # Someone is down. Checked first because the steps for it outrank everything else.
    "loss of consciousness": UNCONSCIOUS,
    "unconscious": UNCONSCIOUS,
    "unresponsive": UNCONSCIOUS,
    "passed out": UNCONSCIOUS,
    "blacked out": UNCONSCIOUS,
    "fainted": UNCONSCIOUS,
    "fainting": UNCONSCIOUS,
    "collapsed": UNCONSCIOUS,
    "won't wake up": UNCONSCIOUS,
    "will not wake up": UNCONSCIOUS,
    "cannot wake": UNCONSCIOUS,
    "can't wake": UNCONSCIOUS,
    "not breathing": UNCONSCIOUS,
    "no pulse": UNCONSCIOUS,
    "no heartbeat": UNCONSCIOUS,

    "choking": CHOKING,
    "choked on": CHOKING,
    "something stuck in his throat": CHOKING,
    "something stuck in her throat": CHOKING,
    "something stuck in my throat": CHOKING,

    # Anaphylaxis. "allergic reaction" on its own is left out - it is how people ask about rashes -
    # but anything describing a swelling airway is here.
    "anaphylaxis": ANAPHYLAXIS,
    "anaphylactic": ANAPHYLAXIS,
    "severe allergic reaction": ANAPHYLAXIS,
    "bad allergic reaction": ANAPHYLAXIS,
    "throat is swelling": ANAPHYLAXIS,
    "throat swelling": ANAPHYLAXIS,
    "throat is closing": ANAPHYLAXIS,
    "throat closing": ANAPHYLAXIS,
    "tongue is swelling": ANAPHYLAXIS,
    "tongue swelling": ANAPHYLAXIS,
    "lips are swelling": ANAPHYLAXIS,
    "face is swelling": ANAPHYLAXIS,
    "epipen": ANAPHYLAXIS,
    "epi pen": ANAPHYLAXIS,

    "severe bleeding": BLEEDING,
    "heavy bleeding": BLEEDING,
    "bleeding heavily": BLEEDING,
    "bleeding badly": BLEEDING,
    "bleeding a lot": BLEEDING,
    "won't stop bleeding": BLEEDING,
    "will not stop bleeding": BLEEDING,
    "keeps bleeding": BLEEDING,
    "lost a lot of blood": BLEEDING,
    "losing a lot of blood": BLEEDING,
    "blood everywhere": BLEEDING,
    "deep cut": BLEEDING,
    "coughing up blood": BLEEDING,
    "vomiting blood": BLEEDING,

    "shortness of breath": BREATHING,
    "short of breath": BREATHING,
    "can't breathe": BREATHING,
    "cannot breathe": BREATHING,
    "cant breathe": BREATHING,
    "struggling to breathe": BREATHING,
    "trouble breathing": BREATHING,
    "difficulty breathing": BREATHING,
    "hard to breathe": BREATHING,
    "gasping": BREATHING,
    "blue lips": BREATHING,
    "lips are blue": BREATHING,
    "turning blue": BREATHING,
    "asthma attack": BREATHING,

    "having a seizure": SEIZURE,
    "is seizing": SEIZURE,
    "convulsing": SEIZURE,
    "convulsion": SEIZURE,
    "having a fit": SEIZURE,
    "epileptic fit": SEIZURE,

    "chest pain": CARDIAC,
    "chest pressure": CARDIAC,
    "chest tightness": CARDIAC,
    "tight chest": CARDIAC,
    "tightness in my chest": CARDIAC,
    "tightness in the chest": CARDIAC,
    "pressure in my chest": CARDIAC,
    "pressure in the chest": CARDIAC,
    "chest feels tight": CARDIAC,
    "chest feels heavy": CARDIAC,
    "crushing pain": CARDIAC,
    "heart attack": CARDIAC,
    "pain in my left arm": CARDIAC,
    "pain down my arm": CARDIAC,
    "spreading to my arm": CARDIAC,
    "spreading to my jaw": CARDIAC,

    "stroke": STROKE,
    "sudden weakness": STROKE,
    "vision loss": STROKE,
    "drooping": STROKE,
    "slurred": STROKE,
    "slurring": STROKE,
    "can't lift my arm": STROKE,
    "cannot lift my arm": STROKE,
    "can't raise my arm": STROKE,
    "numbness on one side": STROKE,
    "weakness on one side": STROKE,
    "one side of my body": STROKE,
    "one side of his body": STROKE,
    "one side of her body": STROKE,
    "sudden confusion": STROKE,
    "worst headache of my life": STROKE,

    # An overdose that has already happened. Asking *how* to overdose is caught earlier by the
    # safety layer and answered with crisis resources; this is for someone who needs urgent care
    # now - including a parent whose child has swallowed something.
    "overdose": POISONING,
    "overdosed": POISONING,
    "took too many": POISONING,
    "poisoning": POISONING,
    "swallowed pills": POISONING,
    "swallowed bleach": POISONING,
    "drank bleach": POISONING,
    "swallowed poison": POISONING,
    "ate rat poison": POISONING,
}

# A few phrases have an ordinary meaning this table would otherwise read as an emergency:
# "I can't breathe through my nose" is a blocked nose, not respiratory distress. Only the phrase
# the context explains is dropped, so a message that also says something else - "I can't breathe
# through my nose and my lips are blue" - is still an emergency. Suppressing a red flag is the
# dangerous direction, so this table stays tiny and each entry has to be unambiguous.
_NOSE = [
    "through my nose", "through his nose", "through her nose", "through the nose",
    "through my nostrils", "blocked nose", "stuffy nose", "blocked up nose",
]
PHRASE_EXCEPTIONS = {
    "can't breathe": _NOSE,
    "cannot breathe": _NOSE,
    "cant breathe": _NOSE,
}

# Bleeding in pregnancy is not a wound: telling someone to press on it and raise the limb would be
# nonsense, and it needs a maternity unit rather than a plaster. When a bleeding phrase appears
# alongside any of these, the category is swapped. This is the one rule the phrase table cannot
# express on its own, because it depends on two parts of the sentence at once.
PREGNANCY_TERMS = ["pregnant", "pregnancy", "miscarriage", "miscarrying", "in labour", "in labor"]

# Two levels. EMERGENCY means the message is specific enough to act on: the warning and the
# first-aid steps go out immediately. CAUTION means a symptom that is usually ordinary and
# occasionally serious - answering "I have trouble breathing" with a heart-attack warning is
# wrong far more often than it is right, and frightening every time. A CAUTION is answered
# normally, with causes commonest-first, questions, and a fixed line on when to call for help.
EMERGENCY = "emergency"
CAUTION = "caution"

# Phrases that only raise a CAUTION on their own. Everything not listed here is specific enough
# to stand alone: "gasping", "blue lips", "unresponsive", any stroke sign, any poisoning.
AMBIGUOUS_PHRASES = {
    # Breathlessness is the example that prompted this: a blocked nose, a cold, being unfit and
    # a panic attack all produce it, and all are commoner than a heart attack.
    "shortness of breath", "short of breath", "trouble breathing", "difficulty breathing",
    "hard to breathe", "can't breathe", "cannot breathe", "cant breathe", "asthma attack",
    # Chest discomfort with nothing else alongside it. Reflux, muscle strain and anxiety are all
    # commoner causes; the questions in the reply are the ones that separate them from angina.
    "chest pain", "chest pressure", "chest tightness", "tight chest", "tightness in my chest",
    "tightness in the chest", "pressure in my chest", "pressure in the chest",
    "chest feels tight", "chest feels heavy",
    # A simple faint is common and usually benign. Not waking up is not, and is listed separately.
    "fainted", "fainting", "passed out", "blacked out",
    "deep cut", "keeps bleeding",
    # Asking about an auto-injector is usually a question, not an event in progress.
    "epipen", "epi pen",
}

# Categories where a lone ambiguous symptom still raises the alarm. Chest pain is usually not
# cardiac - reflux, muscle strain and anxiety are all commoner - but it is the one symptom where
# being wrong the other way costs the most, so it gets the warning anyway *and* gets asked the
# questions that would narrow it down. Everything else ambiguous stays a caution.
ALARM_AND_ASK = {CARDIAC}

# Words that turn an ambiguous symptom into an emergency. No negation handling: "it is not
# severe" escalates too, which is the safe direction to be wrong in.
SEVERITY_QUALIFIERS = [
    "sudden", "suddenly", "all of a sudden", "out of nowhere", "severe", "severely",
    "crushing", "worst", "getting worse", "unbearable", "excruciating", "agony",
    "can't speak", "cannot speak", "can't finish a sentence",
]

# Closes a CAUTION reply. Fixed text for the same reason the first-aid steps are: this is the
# part that has to be right, and it is what makes it safe not to alarm someone up front.
SAFETY_NET = {
    BREATHING:
        "**Call your emergency number straight away if** your lips, face or fingertips turn blue "
        "or grey, you cannot finish a sentence, you are fighting for every breath, or it comes on "
        "suddenly along with chest pain.",
    CARDIAC:
        "**Call your emergency number straight away if** the pain spreads to your arm, neck or "
        "jaw, you go sweaty, sick or breathless with it, it lasts more than 15 minutes, or it "
        "feels crushing or comes on suddenly.",
    UNCONSCIOUS:
        "**Call your emergency number straight away if** it happens again, they do not wake fully "
        "within a minute or two, it happened alongside chest pain or breathlessness, or it "
        "happened while they were sitting, lying down or exercising.",
    BLEEDING:
        "**Call your emergency number straight away if** the bleeding soaks through a dressing, "
        "does not stop after 10 minutes of firm pressure, spurts, or the person goes pale, cold, "
        "faint or confused.",
    ANAPHYLAXIS:
        "**Use the auto-injector and call your emergency number straight away if** lips, tongue or "
        "throat start to swell, breathing gets tight or wheezy, or they feel faint — do not wait "
        "to see whether it settles.",
}

# Shown for every red flag, before the steps for the specific symptom.
CALL_FIRST = [
    "**Call your emergency number now** — **112** in India and across Europe, **911** in the US — "
    "and ask for an ambulance. Do not drive yourself.",
    "Stay with someone if you can, and unlock the door so help can get in.",
]

GUIDANCE = {
    UNCONSCIOUS: (
        "If someone has collapsed", [
            "Check whether they are breathing normally. Put your phone on speaker so you can keep "
            "your hands free while the call handler talks you through it.",
            "If they are breathing, roll them onto their side with their head tilted back, so their "
            "airway stays open and they cannot choke.",
            "If they are not breathing normally, start chest compressions in the centre of the "
            "chest — push hard and fast. The call handler will keep time with you.",
            "Do not put anything in their mouth, and do not leave them on their own.",
        ],
    ),
    CHOKING: (
        "While you wait — choking", [
            "If they can cough, speak or breathe, let them keep coughing. Do not hit their back "
            "while coughing is working.",
            "If they cannot: five sharp blows between the shoulder blades with the heel of your "
            "hand, then five abdominal thrusts. Keep alternating.",
            "For a baby under one year, use back blows and chest thrusts — never abdominal thrusts.",
            "If they go limp, start chest compressions and keep the call handler on speaker.",
        ],
    ),
    ANAPHYLAXIS: (
        "While you wait — a swelling airway or severe allergic reaction", [
            "If there is an adrenaline auto-injector (EpiPen, Jext, Emerade), use it now, into the "
            "outer thigh — through clothing is fine. It is far more dangerous to wait than to use it.",
            "Lie them flat with their legs raised. If breathing is hard let them sit up; if they are "
            "being sick, roll them onto their side. Do not let them stand up or walk about.",
            "Tell the call handler it may be anaphylaxis, and what they ate, touched or were stung by.",
            "If there is no improvement after 5 minutes and a second injector is available, use it.",
        ],
    ),
    BLEEDING: (
        "While you wait — heavy bleeding", [
            "Press hard on the wound with a clean cloth or pad, and keep pressing without lifting "
            "it to look.",
            "If blood soaks through, put another pad on top — do not take the first one off.",
            "Raise the injured part above the level of the heart if you can, and lie the person "
            "down.",
            "Do not use a tourniquet unless you have been trained to and the bleeding will not stop.",
        ],
    ),
    BREATHING: (
        "While you wait — trouble breathing", [
            "Sit upright and lean forward slightly. Do not lie flat.",
            "Loosen tight clothing, open a window, and try to slow your breathing.",
            "If you have a reliever inhaler prescribed for you, use it the way your label says.",
            "Blue or grey lips, confusion, or not being able to finish a sentence means calling "
            "back to say it is getting worse.",
        ],
    ),
    SEIZURE: (
        "While you wait — a seizure", [
            "Note the time it started. Move anything hard or sharp out of the way and put something "
            "soft under their head.",
            "Do not hold them down, and do not put anything in their mouth.",
            "When the jerking stops, roll them onto their side and stay with them until they are "
            "properly awake.",
            "Tell the call handler if it lasts more than 5 minutes, another one follows, it is their "
            "first, they are injured, or they do not come round.",
        ],
    ),
    CARDIAC: (
        "While you wait — chest pain", [
            "Stop what you are doing, sit down and rest. Loosen anything tight around your neck "
            "or waist.",
            "Tell the call handler your age and what the pain feels like. They will say whether to "
            "take aspirin — do not decide that on your own.",
            "If a doctor has prescribed you a heart spray or tablet, take it the way you were told.",
            "Do not wait to see whether it passes, even if the pain comes and goes.",
        ],
    ),
    STROKE: (
        "While you wait — signs of a stroke", [
            "Check **FAST**: **F**ace dropped on one side, **A**rm that cannot stay raised, "
            "**S**peech slurred or muddled — **T**ime to call for help.",
            "Note the exact time the symptoms started, or when the person was last seen well. The "
            "treatment they can be given depends on it.",
            "Do not eat, drink or take medicine until someone has examined you — swallowing may be "
            "affected.",
            "Symptoms that clear up on their own still need to be seen today, not next week.",
        ],
    ),
    OBSTETRIC: (
        "While you wait — bleeding in pregnancy", [
            "Call your maternity unit or labour ward as well — they will say where to go, and they "
            "are open through the night.",
            "Keep the pads or clothing you have used so they can see how much blood there has been. "
            "Do not put anything inside, and do not use a tampon.",
            "Lie down on your left side and stay still until help arrives.",
            "Tell them how many weeks pregnant you are, and whether the baby is still moving as usual.",
        ],
    ),
    POISONING: (
        "While you wait — something was swallowed or taken", [
            "Call your emergency number or your local poison control centre even if the person "
            "seems fine. Paracetamol and several other medicines do their damage hours before "
            "anyone feels ill.",
            "Keep the packet, bottle or leaflet with you and say what was taken, how much, and when.",
            "Do not make them vomit, and do not give salt water, milk or anything else to drink "
            "unless you are told to.",
            "Say straight away if they are drowsy, fitting or hard to wake.",
        ],
    ),
}

# What the warning calls each category. The matched phrase is not usable here: it is tuned for
# recall, so it is often a single word. The raw phrases are still kept for the history entry,
# where they are the audit trail of why a message was flagged.
LABELS = {
    UNCONSCIOUS: "loss of consciousness",
    CHOKING: "choking",
    ANAPHYLAXIS: "a severe allergic reaction",
    BLEEDING: "heavy bleeding",
    BREATHING: "trouble breathing",
    SEIZURE: "a seizure",
    CARDIAC: "chest pain",
    STROKE: "stroke symptoms",
    OBSTETRIC: "bleeding in pregnancy",
    POISONING: "a possible poisoning or overdose",
}

# A message that mentions three symptoms still gets a list someone can act on. Beyond two blocks
# the steps stop being read, which defeats the point of showing them instantly.
MAX_GUIDANCE_BLOCKS = 2

_RED_FLAG_PATTERNS = compile_phrases(RED_FLAG_KEYWORDS)
_PREGNANCY_PATTERNS = compile_phrases(PREGNANCY_TERMS)
_EXCEPTION_PATTERNS = {phrase: compile_phrases(contexts)
                      for phrase, contexts in PHRASE_EXCEPTIONS.items()}
_QUALIFIER_PATTERNS = compile_phrases(SEVERITY_QUALIFIERS)


def _explained_away(message: str, phrase: str) -> bool:
    """Whether the rest of the message gives this phrase its ordinary meaning."""
    patterns = _EXCEPTION_PATTERNS.get(phrase)
    return bool(patterns) and bool(find_phrases(message, patterns))


def detect_redflags(message: str) -> Tuple[bool, List[Tuple[str, str]]]:
    """Detect red-flag phrases and return (has_redflag, list of (phrase, category))."""
    found = [(phrase, RED_FLAG_KEYWORDS[phrase]) for phrase in find_phrases(message, _RED_FLAG_PATTERNS)
             if not _explained_away(message, phrase)]
    if any(category == BLEEDING for _, category in found) and find_phrases(message, _PREGNANCY_PATTERNS):
        found = [(phrase, OBSTETRIC if category == BLEEDING else category) for phrase, category in found]
        # OBSTETRIC has no safety net, so triage treats it as an emergency: bleeding in pregnancy
        # is never the "see how it goes" case, whatever words were used to describe it.
    return (len(found) > 0), found


def _categories(redflags: Sequence[Tuple[str, str]]) -> List[str]:
    """The categories found, most immediately life-threatening first."""
    matched = {category for _, category in redflags}
    return [c for c in CATEGORY_ORDER if c in matched]


def emergency_label(redflags: Sequence[Tuple[str, str]]) -> Optional[str]:
    """What to call this emergency when telling the user why they were warned."""
    categories = _categories(redflags)
    return ", ".join(LABELS[c] for c in categories) if categories else None


class Triage(NamedTuple):
    """How a message was classified, with the fixed text that goes with it.

    All of it is phrase matching and dictionary lookups, so it costs microseconds and can be
    decided before anything slow - retrieval, or the model - has been started.
    """

    level: str                  # EMERGENCY or CAUTION
    ask: bool                   # alarmed on one vague symptom, so the reply asks as well
    label: str                  # what to call it: "stroke symptoms", "trouble breathing"
    details: str                # the phrases that matched, kept as the history audit trail
    guidance: Optional[str]     # first-aid steps, on an EMERGENCY
    safety_net: Optional[str]   # when to call for help, on a CAUTION


def _level(message: str, redflags: Sequence[Tuple[str, str]]) -> Tuple[str, bool]:
    """Return (level, whether the reply should also ask what would narrow it down)."""
    if any(phrase not in AMBIGUOUS_PHRASES for phrase, _ in redflags):
        return EMERGENCY, False
    # Two systems at once is its own corroboration: chest pain *and* breathlessness is not the
    # reflux-or-strain picture that either one alone usually is.
    if len(_categories(redflags)) > 1:
        return EMERGENCY, False
    if find_phrases(message, _QUALIFIER_PATTERNS):
        return EMERGENCY, False
    # Nothing corroborates it. High-stakes categories are alarmed anyway, and asked about;
    # the rest are answered calmly.
    if _categories(redflags)[0] in ALARM_AND_ASK:
        return EMERGENCY, True
    return CAUTION, False


def triage(message: str) -> Optional[Triage]:
    """Classify a message, or None if it holds no red flag at all."""
    has_redflag, redflags = detect_redflags(message)
    if not has_redflag:
        return None
    level, ask = _level(message, redflags)
    categories = _categories(redflags)
    # Every caution-capable category has a safety net; if one day one does not, an emergency is
    # the safe thing to fall back to rather than a reply with nothing at the end of it.
    if level == CAUTION and categories[0] not in SAFETY_NET:
        level = EMERGENCY
    return Triage(
        level=level,
        ask=ask,
        label=emergency_label(redflags),
        details=", ".join(phrase for phrase, _ in redflags),
        guidance=emergency_guidance(redflags) if level == EMERGENCY else None,
        safety_net=SAFETY_NET[categories[0]] if level == CAUTION else None,
    )


def carried_forward(message: str) -> Optional[Triage]:
    """The symptom from an earlier turn, as a caution for the follow-up that answers it.

    A follow-up - "about three days, and worse when I lie down" - names no symptom, so triaging it
    on its own finds nothing and the reply loses both its shape and its safety net. The symptom is
    in the turn before. Whatever that turn was, what comes back is the calm version: an emergency's
    steps were already shown on the turn that earned them and must not be sent twice.
    """
    triaged = triage(message)
    if triaged is None:
        return None
    categories = _categories(detect_redflags(message)[1])
    if not categories or categories[0] not in SAFETY_NET:
        return None
    return triaged._replace(level=CAUTION, guidance=None, safety_net=SAFETY_NET[categories[0]])


def emergency_guidance(redflags: Sequence[Tuple[str, str]]) -> Optional[str]:
    """Fixed first-aid steps for the detected red flags, as Markdown, or None if there are none.

    Costs a dictionary lookup, so it can be sent in the same breath as the warning - before
    retrieval, and long before the model has read the question.
    """
    categories = _categories(redflags)
    if not categories:
        return None

    blocks = ["**Do this now**\n" + "\n".join(f"- {step}" for step in CALL_FIRST)]
    for category in categories[:MAX_GUIDANCE_BLOCKS]:
        heading, steps = GUIDANCE[category]
        blocks.append(f"**{heading}**\n" + "\n".join(f"- {step}" for step in steps))
    return "\n\n".join(blocks)
