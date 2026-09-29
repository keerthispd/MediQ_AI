"""Emergency messages, and how long they take to say something useful.

What matters on the emergency path is not how long the whole reply takes but how long it takes to
reach the first instruction the person can act on. On a CPU-only machine the local model needs tens
of seconds to write its first paragraph, so anything that has to be right - call now, and what to do
while you wait - is fixed text from backend/services/redflag.py, sent before the model is asked
anything. These tests hold that line: the steps must survive the model failing, must not wait on
retrieval, and the model's own share must stay small enough not to keep someone reading.
"""
import json

import pytest

from backend.routes.api import EMERGENCY_FOOTER
from backend.services import knowledge, llm
from backend.services.redflag import GUIDANCE, MAX_GUIDANCE_BLOCKS, SAFETY_NET, detect_redflags
from conftest import get_history, reply_text, stream_events

CHEST_PAIN = "I have sudden chest pain"


def chat(client, auth, message, **data):
    return stream_events(client.post("/api/chat", data={"message": message, **data}, headers=auth))


def test_the_steps_are_sent_ahead_of_the_model(client, auth, fake_llm):
    events = chat(client, auth, CHEST_PAIN)
    deltas = [e["text"] for e in events if e["type"] == "delta"]

    # Warning, then what to do, then whatever the model has to add
    assert deltas[0].startswith("⚠️") and "**chest pain**" in deltas[0]
    assert "**Call your emergency number now**" in deltas[1]
    assert "sit down and rest" in deltas[1]
    assert "".join(deltas[2:-1]) == "\n\nModel reply."
    # And the last word is fixed text again, settling which half of the reply wins
    assert deltas[-1] == "\n\n" + EMERGENCY_FOOTER
    # All of it is kept, so the history shows what the person was actually told
    assert get_history(client, auth)[0]["assistant_reply"] == reply_text(events)


def test_the_steps_survive_the_model_failing(client, auth, fake_llm):
    """The proof that they never waited on it: they are there when it produces nothing at all."""
    fake_llm.chunks = []
    fake_llm.error = "The local AI model isn't running."

    text = reply_text(chat(client, auth, CHEST_PAIN))
    assert "**Call your emergency number now**" in text
    assert "**112**" in text and "**911**" in text
    assert "Do not drive yourself" in text
    assert text.endswith(EMERGENCY_FOOTER)


def test_an_emergency_skips_retrieval(client, auth, fake_llm, monkeypatch):
    """Passages are read by the model before it writes a word - seconds of CPU, for background."""
    searched = []
    monkeypatch.setattr(knowledge, "search", lambda query, **kw: searched.append(query) or [])

    events = chat(client, auth, CHEST_PAIN)
    assert searched == []
    assert not [e for e in events if e["type"] == "sources"]

    # An ordinary question still gets grounded
    chat(client, auth, "what causes a sore throat")
    assert searched == ["what causes a sore throat"]


def test_an_emergency_uses_the_warm_model_on_a_short_leash(client, auth, fake_llm):
    chat(client, auth, CHEST_PAIN)
    chat(client, auth, "I have a mild headache")

    emergency, ordinary = fake_llm.calls
    # The stronger model, because the background it writes has to be right - and it is warmed and
    # primed at startup, so an emergency pays for neither the weights nor the prompt
    assert emergency["model"] == llm.EMERGENCY_MODEL == llm.CHAT_MODEL
    assert emergency["max_tokens"] == llm.EMERGENCY_MAX_TOKENS < llm.CHAT_MAX_TOKENS
    assert ordinary["model"] == llm.FAST_MODEL
    assert ordinary["max_tokens"] == llm.CHAT_MAX_TOKENS


def test_the_model_is_told_to_be_brief_and_not_to_contradict_the_steps(client, auth, fake_llm):
    chat(client, auth, CHEST_PAIN)
    messages = fake_llm.calls[0]["messages"]

    system = messages[0]["content"]
    assert "emergency symptoms (chest pain)" in system
    # Not "do not contradict the steps" - that was measured failing. The model is kept out of the
    # instructions business altogether: causes, and what the medical team will do.
    assert "Write nothing they should do" in system
    assert "at most 60 words" in system
    # Repeated at the end of the prompt, where a small model weights it most
    assert messages[-1]["content"].endswith("No instructions - they already have those.")


@pytest.mark.parametrize("message, step", [
    ("I have sudden chest pain", "Stop what you are doing, sit down and rest"),
    ("I think my mother is having a stroke", "**F**ace dropped on one side"),
    ("there is severe bleeding from his arm", "keep pressing without lifting"),
    ("he is gasping for air", "Sit upright and lean forward"),
    ("there was a loss of consciousness", "start chest compressions"),
    ("my son swallowed pills from my handbag", "Do not make them vomit"),
])
def test_the_steps_match_the_symptom(client, auth, fake_llm, message, step):
    text = reply_text(chat(client, auth, message))
    assert step in text
    assert "**Call your emergency number now**" in text


def test_several_symptoms_still_give_a_list_someone_can_act_on(client, auth, fake_llm):
    """Beyond two blocks the steps stop being read, so the least urgent ones are dropped."""
    text = reply_text(chat(client, auth, "he has chest pain, severe bleeding and loss of consciousness"))

    shown = [heading for heading, _ in GUIDANCE.values() if f"**{heading}**" in text]
    assert len(shown) == MAX_GUIDANCE_BLOCKS
    # Kept: someone not breathing, then the bleeding. Dropped: the chest pain steps.
    assert "start chest compressions" in text
    assert "keep pressing without lifting" in text
    assert "sit down and rest" not in text
    # The warning still names every symptom that was found, most serious first - the phrase table is
    # ordered by category, so the naming follows the same priority as the steps
    assert "**loss of consciousness, heavy bleeding, chest pain**" in text


def test_an_ordinary_question_gets_no_emergency_steps(client, auth, fake_llm):
    events = chat(client, auth, "what are the side effects of metformin")
    assert events[0] == {"type": "meta", "redflag": False}
    assert reply_text(events) == "Model reply."


# How people actually type an emergency. An earlier phrase list held the textbook wording only and
# caught none of these - no warning, no steps, and the message answered as an ordinary question.
REAL_PHRASINGS = [
    ("my face is drooping and my arm feels weak", "stroke"),
    ("my speech is slurred all of a sudden", "stroke"),
    ("this is the worst headache of my life", "stroke"),
    ("I cannot breathe properly", "breathing"),
    ("he is struggling to breathe", "breathing"),
    ("his lips are blue", "breathing"),
    ("she passed out and I cannot wake her", "unconscious"),
    ("he collapsed and is unresponsive", "unconscious"),
    ("she is not breathing", "unconscious"),
    ("the wound will not stop bleeding", "bleeding"),
    ("I am losing a lot of blood", "bleeding"),
    ("there is a crushing pressure in my chest", "cardiac"),
    ("my chest feels tight and my left arm hurts", "cardiac"),
    ("I think I am having a heart attack", "cardiac"),
    ("my throat is swelling after eating peanuts", "anaphylaxis"),
    ("she is having a severe allergic reaction", "anaphylaxis"),
    ("my son is having a seizure right now", "seizure"),
    ("he is convulsing on the floor", "seizure"),
    ("my daughter is choking on a grape", "choking"),
    ("he drank bleach from under the sink", "poisoning"),
]

# The other side of the trade. A false positive costs a caution box; these must not pay it.
NOT_EMERGENCIES = [
    "what are the side effects of metformin",
    "is ibuprofen safe to take for a headache",
    "can I take omeprazole long term",
    "does atorvastatin cause muscle pain",
    "I have a sore throat",
    "My shoulder hurts after backstroke practice",
    "what causes a sore throat",
    "I feel hopeless and cry every day",
    "I'm burnt out and have no motivation",
    "I am having panic attacks at work",
    # A blocked nose is not respiratory distress
    "I cannot breathe properly through my nose",
    "I have a blocked nose and cannot breathe",
]


@pytest.mark.parametrize("message, category", REAL_PHRASINGS)
def test_everyday_wording_is_recognised(message, category):
    found, redflags = detect_redflags(message)
    assert found, f"{message!r} would get no warning and no steps"
    assert category in {c for _, c in redflags}


@pytest.mark.parametrize("message", NOT_EMERGENCIES)
def test_ordinary_messages_do_not_trip_the_detector(message):
    assert not detect_redflags(message)[0], f"{message!r} is not an emergency"


def test_an_explained_phrase_is_dropped_without_disarming_the_rest():
    """The exception cancels the phrase it explains, not the whole message."""
    assert not detect_redflags("I can't breathe through my nose")[0]
    # Still an emergency: the blue lips are not explained by the blocked nose
    found, redflags = detect_redflags("I can't breathe through my nose and my lips are blue")
    assert found and {c for _, c in redflags} == {"breathing"}


def test_bleeding_in_pregnancy_gets_maternity_steps_not_wound_care(client, auth, fake_llm):
    """Pressing on the wound and raising the limb is the wrong advice, and it would be the default."""
    text = reply_text(chat(client, auth, "I am 7 months pregnant and bleeding heavily"))
    assert "maternity unit" in text
    assert "how many weeks pregnant you are" in text
    assert "Press hard on the wound" not in text


def test_startup_warms_both_models_and_primes_the_emergency_prompt(monkeypatch):
    """Neither the weights nor the prompt should be read for the first time during an emergency."""
    from backend import main

    warmed = []
    monkeypatch.setattr(llm, "warmup", lambda model=None, messages=None: warmed.append((model, messages)))
    main._warm_models()

    assert [model for model, _ in warmed] == [llm.EMERGENCY_MODEL, llm.FAST_MODEL]
    emergency_primer, everyday_primer = warmed[0][1], warmed[1][1]
    assert everyday_primer is None
    # The primed prefix is the one a real emergency sends, not a bare "hi"
    assert emergency_primer[0] == llm.build_chat_messages("x", emergency_details="chest pain")[0]


def test_an_uploaded_report_mentioning_a_red_flag_gets_the_steps_too(client, auth, fake_llm):
    response = client.post(
        "/api/upload",
        files={"file": ("ecg.txt", b"Sinus rhythm", "text/plain")},
        data={"message": "I have sudden chest pain, is this normal?"},
        headers=auth,
    )
    text = reply_text(stream_events(response))
    assert "**Call your emergency number now**" in text
    assert "Stop what you are doing, sit down and rest" in text


# Chest pain is mostly not cardiac - reflux, strain and anxiety are commoner - but it is the symptom
# where being wrong the other way costs most. So a lone "I have chest pain" is alarmed like any
# emergency AND asked about, rather than being sorted into one bucket or the other.
def test_a_lone_chest_pain_is_alarmed_and_asked_about(client, auth, fake_llm):
    events = chat(client, auth, "I have chest pain")

    assert events[0] == {"type": "meta", "redflag": True}
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    # Alarmed: the same warning and the same fixed steps as any other emergency
    assert deltas[0].startswith("⚠️") and "**chest pain**" in deltas[0]
    assert "Stop what you are doing, sit down and rest" in deltas[1]

    # And asked: the model spends its few words on the questions that would narrow it down
    system = fake_llm.calls[0]["messages"][0]["content"]
    assert "one symptom, no detail" in system
    assert "ask exactly two short questions" in system
    assert fake_llm.calls[0]["messages"][-1]["content"].endswith("Then exactly two short questions.")


def test_a_detailed_emergency_is_not_asked_about(client, auth, fake_llm):
    """Nothing to narrow down: they have already said what is happening."""
    chat(client, auth, "I have sudden chest pain spreading to my arm")
    system = fake_llm.calls[0]["messages"][0]["content"]
    assert "ask exactly two short questions" not in system
    assert "what the medical team will do" in system


def test_answering_the_questions_gives_the_fuller_answer(client, auth, fake_llm):
    """The steps were shown on the turn that earned them; this turn is the explanation."""
    history = json.dumps([
        {"role": "user", "text": "I have chest pain"},
        {"role": "assistant", "text": "Does it spread to your arm? Are you breathless?"},
    ])
    events = chat(client, auth, "No spreading, it hurts more when I press on it", history=history)

    # Not alarmed a second time
    assert events[0] == {"type": "meta", "redflag": False}
    text = reply_text(events)
    assert "⚠️" not in text and "**Do this now**" not in text
    system = fake_llm.calls[0]["messages"][0]["content"]
    assert "This is a follow-up" in system
    # Still closed by the fixed cardiac line
    assert text.endswith(SAFETY_NET["cardiac"])
