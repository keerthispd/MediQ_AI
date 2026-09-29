"""Symptoms that are usually ordinary and occasionally serious.

Most symptoms have several causes, and the common ones are common. "I have trouble breathing" is a
blocked nose or a cold far more often than it is a heart attack, so answering it with a red warning
and first-aid steps is wrong most of the time and frightening every time.

These get a CAUTION instead: the ordinary answer path, with the model asked for causes
commonest-first and a question or two, plus one fixed line at the end saying what would change the
picture. The alarm is still one qualifier away - "sudden", "severe", or a second symptom.
"""
import json

import pytest

from backend.services import knowledge, llm
from backend.services.redflag import CAUTION, EMERGENCY, SAFETY_NET, triage
from conftest import get_history, reply_text, stream_events

BREATHLESS = "I have trouble breathing"


def chat(client, auth, message, **data):
    return stream_events(client.post("/api/chat", data={"message": message, **data}, headers=auth))


@pytest.mark.parametrize("message", [
    "I have trouble breathing",
    "I have shortness of breath",
    "I fainted this morning",
    "I have a deep cut on my hand",
    "how do I use an epipen",
])
def test_a_lone_ambiguous_symptom_is_not_an_emergency(message):
    triaged = triage(message)
    assert triaged is not None and triaged.level == CAUTION
    assert triaged.guidance is None
    assert triaged.safety_net


@pytest.mark.parametrize("message", [
    # Specific enough on its own
    "his lips are blue",
    "he is gasping for air",
    "she is unresponsive",
    "my face is drooping",
    # A severity word
    "I have sudden chest pain",
    "I have severe trouble breathing",
    "this came on all of a sudden and my chest feels tight",
    # Two systems at once corroborate each other
    "I have chest pain and shortness of breath",
])
def test_the_alarm_is_one_qualifier_away(message):
    triaged = triage(message)
    assert triaged is not None and triaged.level == EMERGENCY, message
    assert triaged.guidance and triaged.safety_net is None


def test_a_caution_gets_no_warning_no_steps_and_no_red_flag(client, auth, fake_llm):
    events = chat(client, auth, BREATHLESS)

    assert events[0] == {"type": "meta", "redflag": False}
    text = reply_text(events)
    assert "⚠️" not in text
    assert "**Do this now**" not in text
    assert "ask for an ambulance" not in text


def test_a_caution_ends_with_the_fixed_safety_net(client, auth, fake_llm):
    """Not alarming them up front is only safe because this is always underneath."""
    text = reply_text(chat(client, auth, BREATHLESS))
    assert text == "Model reply.\n\n" + SAFETY_NET["breathing"]
    assert "cannot finish a sentence" in text
    # And it survives the model failing, like the emergency steps do
    fake_llm.chunks, fake_llm.error = [], "The local AI model isn't running."
    assert reply_text(chat(client, auth, BREATHLESS)).endswith(SAFETY_NET["breathing"])


def test_the_safety_net_matches_the_symptom(client, auth, fake_llm):
    assert "soaks through a dressing" in reply_text(chat(client, auth, "I have a deep cut"))
    assert "wake fully within a minute" in reply_text(chat(client, auth, "I fainted this morning"))


def test_the_model_is_asked_for_common_causes_first_and_questions(client, auth, fake_llm):
    chat(client, auth, BREATHLESS)
    system = fake_llm.calls[0]["messages"][0]["content"]

    assert "usually something ordinary" in system
    assert "everyday ones first and the serious ones" in system
    assert "two short questions" in system
    # Measured: told only not to say it, the model wrote its own worse 'When to Call' section
    # next to the fixed one. It is now barred from the whole subject.
    assert "say nothing about when to get help" in system.lower()
    assert "do not mention emergency services" in system.lower()
    # The reminder is repeated at the end of the prompt, where a small model weights it most
    assert fake_llm.calls[0]["messages"][-1]["content"].endswith(
        "Ordinary causes first, serious ones last, a suggestion on each, then your questions."
    )


def test_a_caution_is_answered_on_the_ordinary_path(client, auth, fake_llm, monkeypatch):
    """Not an emergency, so it keeps retrieval and the fast model - no 25-second wait."""
    searched = []
    monkeypatch.setattr(knowledge, "search", lambda query, **kw: searched.append(query) or [])

    chat(client, auth, BREATHLESS)
    assert searched == [BREATHLESS]
    # Grounded and given the full output budget, unlike an emergency - but on the larger model,
    # because ordering the causes is the whole point and the small one copies them out instead
    assert fake_llm.calls[0]["model"] == llm.CAUTION_MODEL == llm.CHAT_MODEL
    assert fake_llm.calls[0]["max_tokens"] == llm.CHAT_MAX_TOKENS


def test_a_caution_is_not_flagged_in_history_but_is_recorded(client, auth, fake_llm):
    """It was not an emergency, so it must not carry the badge - but the match is auditable."""
    chat(client, auth, BREATHLESS)
    item = get_history(client, auth)[0]
    assert item["redflag"] is False
    assert item["redflag_details"] == "trouble breathing"


def test_the_blocked_nose_case_end_to_end(client, auth, fake_llm):
    """The case this was built for: no alarm, and nothing about a heart attack."""
    text = reply_text(chat(client, auth, "I have difficulty breathing"))
    assert "⚠️" not in text and "heart attack" not in text.lower()
    assert text.endswith(SAFETY_NET["breathing"])


def test_each_cause_carries_a_suggestion_not_just_a_name(client, auth, fake_llm):
    chat(client, auth, BREATHLESS)
    system = fake_llm.calls[0]["messages"][0]["content"]
    assert "End every bullet with what to do about that one" in system
    assert "worth getting checked by a doctor" in system


def test_answering_the_questions_gets_an_answer_not_another_round(client, auth, fake_llm):
    """The failure this guards against: the same menu back, forever, instead of a conclusion."""
    history = json.dumps([
        {"role": "user", "text": BREATHLESS},
        {"role": "assistant", "text": "Possible causes... How long has it lasted?"},
    ])
    chat(client, auth, "About three days, and it is worse when I lie down", history=history)

    system = fake_llm.calls[0]["messages"][0]["content"]
    assert "This is a follow-up" in system
    assert "Do not list the causes again" in system
    assert "do not ask another round of questions" in system
    # And the first-pass instructions are gone, not merely added to
    assert "Open with the single most ordinary cause" not in system
    assert fake_llm.calls[0]["messages"][-1]["content"].endswith(
        "Answer what they have told you now - no second round of questions."
    )


def test_the_safety_net_still_closes_a_follow_up(client, auth, fake_llm):
    history = json.dumps([
        {"role": "user", "text": BREATHLESS},
        {"role": "assistant", "text": "How long has it lasted?"},
    ])
    text = reply_text(chat(client, auth, "three days", history=history))
    assert text.endswith(SAFETY_NET["breathing"])


def test_the_apps_own_text_is_not_fed_back_to_the_model(client, auth, fake_llm):
    """Measured: shown its own safety net in the history, the model wrote a second, worse one."""
    from backend.routes.api import _without_app_text

    reply = (
        "\u26a0\ufe0f Your message mentions symptoms that may be an emergency (**chest pain**).\n\n"
        "**Do this now**\n- Call your emergency number now\n\n"
        "What the model actually wrote.\n\n"
        "**Call your emergency number straight away if** the pain spreads to your arm.\n\n"
        "\u2139\ufe0f Follow the steps above"
    )
    assert _without_app_text(reply) == "What the model actually wrote."


def test_a_follow_up_sends_back_only_what_the_model_wrote(client, auth, fake_llm):
    history = json.dumps([
        {"role": "user", "text": BREATHLESS},
        {"role": "assistant", "text": "Causes and a question.\n\n" + SAFETY_NET["breathing"]},
    ])
    chat(client, auth, "three days, worse lying down", history=history)

    earlier = [m for m in fake_llm.calls[0]["messages"] if m["role"] == "assistant"]
    assert earlier and SAFETY_NET["breathing"] not in earlier[0]["content"]
    assert earlier[0]["content"] == "Causes and a question."
