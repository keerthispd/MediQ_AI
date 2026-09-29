"""How the app responds to someone who is struggling.

Four tiers, in the order they are checked:

  self-harm  -> support and helplines, written by hand, model never called
  lethality  -> declines the number, still offers help
  distress   -> answered warmly by the model, with a helpline footer appended
  ordinary   -> answered normally

The ordering matters: a message that is both distressed and self-harming is a self-harm message.
"""
import pytest

from backend.routes.api import CRISIS_REPLY, DISTRESS_FOOTER, LETHALITY_REPLY
from backend.services import safety
from conftest import get_history, reply_text, stream_events

SELF_HARM = [
    "I feel suicidal",
    "I want to die",
    "I don't want to live anymore",
    "everyone would be better off without me",
    "I wish I was dead",
    "I can't go on like this",
    "I don't want to wake up tomorrow",
    "I've been thinking about ending my life",
    "I hurt myself last night",
    "nothing to live for",
]

DISTRESS = [
    "I feel hopeless and cry every day",
    "I have been feeling worthless lately",
    "I am having panic attacks at work",
    "I can't cope with everything right now",
    "I feel so lonely since moving here",
    "I'm burnt out and have no motivation",
]

INFORMATIONAL = [
    "what causes anxiety disorders",
    "how is depression usually treated",
    "what are the side effects of metformin",
    "I have a sore throat",
]


@pytest.mark.parametrize("message", SELF_HARM)
def test_self_harm_phrasing_is_recognised(message):
    category, found = safety.classify(message)
    assert category == safety.SELF_HARM, f"{message!r} must be offered support, not a medical answer"
    assert found


@pytest.mark.parametrize("message", DISTRESS)
def test_distress_is_recognised_without_blocking(message):
    assert safety.classify(message)[0] is None, f"{message!r} should still get an answer"
    assert safety.detect_distress(message), f"{message!r} should be answered gently"


@pytest.mark.parametrize("message", INFORMATIONAL)
def test_questions_about_mental_health_are_not_treated_as_distress(message):
    """"What causes anxiety" is a question, not a disclosure - it should not trigger the footer."""
    assert safety.classify(message)[0] is None
    assert not safety.detect_distress(message)


def test_self_harm_takes_precedence_over_distress():
    message = "I feel hopeless and I want to die"
    assert safety.classify(message)[0] == safety.SELF_HARM


def test_self_harm_reply_offers_help_and_never_reaches_the_model(client, auth, fake_llm):
    events = stream_events(client.post("/api/chat", data={"message": "I want to die"}, headers=auth))

    assert events[0] == {"type": "meta", "blocked": True}
    reply = reply_text(events)
    assert reply == CRISIS_REPLY
    # It must give somewhere to go, not just decline
    assert "14416" in reply and "988" in reply and "findahelpline.com" in reply
    assert "116 123" in reply
    assert fake_llm.chats == [], "a local model must not improvise the reply to this"
    # Still saved, so the person sees it in their history
    assert get_history(client, auth)[0]["assistant_reply"] == CRISIS_REPLY


def test_lethality_reply_declines_but_still_offers_support(client, auth, fake_llm):
    events = stream_events(
        client.post("/api/chat", data={"message": "how many pills would kill someone"}, headers=auth)
    )
    reply = reply_text(events)
    assert reply == LETHALITY_REPLY
    assert "poison control" in reply.lower()
    assert "14416" in reply
    assert fake_llm.chats == []


def test_distress_is_answered_and_ends_with_where_to_get_help(client, auth, fake_llm):
    events = stream_events(
        client.post("/api/chat", data={"message": "I feel hopeless and cry every day"}, headers=auth)
    )
    reply = reply_text(events)

    assert "Model reply." in reply, "the person should still get an answer"
    assert DISTRESS_FOOTER in reply, "and should be told where help is"
    # The model was asked to lead with warmth
    system = fake_llm.chats[0][0]["content"]
    assert "acknowledging how they feel" in system


def test_an_ordinary_question_gets_no_footer(client, auth, fake_llm):
    events = stream_events(client.post("/api/chat", data={"message": "I have a sore throat"}, headers=auth))
    assert DISTRESS_FOOTER not in reply_text(events)
    assert "acknowledging how they feel" not in fake_llm.chats[0][0]["content"]
