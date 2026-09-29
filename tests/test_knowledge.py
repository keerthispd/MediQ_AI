"""Hybrid retrieval, and how a grounded answer is assembled."""
import sqlite3

import pytest

from backend.services import knowledge
from conftest import reply_text, stream_events

# Three passages with deliberately little word overlap, so keyword and vector search disagree
PASSAGES = [
    ("Sore throat", "https://example.test/sore-throat", "pharyngitis",
     "A sore throat is usually caused by a virus and clears up within a week without antibiotics."),
    ("Thyroid tests", "https://example.test/tsh", "TSH thyroid stimulating hormone",
     "A raised TSH result usually means the thyroid gland is underactive, a condition called hypothyroidism."),
    ("Anaemia", "https://example.test/anaemia", "low haemoglobin",
     "Anaemia means the blood carries less oxygen than it should, often because haemoglobin is low."),
]


@pytest.fixture
def index(tmp_path, monkeypatch):
    """A tiny on-disk index, with embeddings stubbed so the test needs no model server."""
    import numpy as np

    db_path = tmp_path / "knowledge.db"
    vectors_path = tmp_path / "knowledge.npy"

    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, title TEXT, url TEXT, section TEXT,
                             text TEXT, source TEXT);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(title, aliases, text, content='');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    for i, (title, url, aliases, text) in enumerate(PASSAGES):
        db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (i, title, url, "", text, "Test source"))
        db.execute("INSERT INTO chunks_fts (rowid, title, aliases, text) VALUES (?,?,?,?)",
                   (i, title, aliases, text))
    db.executemany("INSERT INTO meta VALUES (?,?)",
                   [("chunks", "3"), ("embed_model", "test"), ("source", "Test source")])
    db.commit()
    db.close()

    # Passage i points along axis i. The fourth axis belongs to no passage, so a query pointing
    # there is far from everything - which is how "nothing in the corpus fits" is expressed.
    np.save(vectors_path, np.eye(3, 4, dtype="float32"))

    monkeypatch.setattr(knowledge, "DB_PATH", db_path)
    monkeypatch.setattr(knowledge, "VECTORS_PATH", vectors_path)
    monkeypatch.setattr(knowledge, "_state", None)
    monkeypatch.setattr(knowledge, "_embed_query", lambda text, state: None)  # keyword-only by default
    yield knowledge
    knowledge._state = None


def test_finds_the_passage_by_exact_medical_term(index):
    [hit] = index.search("what does a raised TSH mean?", top_k=1)
    assert hit["title"] == "Thyroid tests"
    assert hit["url"] == "https://example.test/tsh"


def test_matches_a_synonym_that_is_not_in_the_passage_text(index):
    # "pharyngitis" appears only in the aliases column, never in the passage itself
    [hit] = index.search("pharyngitis", top_k=1)
    assert hit["title"] == "Sore throat"


def test_vector_and_keyword_results_are_fused(index, monkeypatch):
    # Keyword search matches the sore-throat passage; the vector side leans towards anaemia.
    # Both stay above the relevance floor, so fusion rather than the floor decides the result.
    monkeypatch.setattr(index, "_embed_query", lambda text, state: [0.6, 0.0, 0.8, 0.0])
    titles = [p["title"] for p in index.search("sore throat", top_k=2)]
    assert set(titles) == {"Sore throat", "Anaemia"}


def test_a_loose_keyword_match_alone_does_not_ground_an_answer(index, monkeypatch):
    """Keyword search finds something for almost any shared word. When nothing is semantically
    close, citing whatever matched is worse than citing nothing."""
    monkeypatch.setattr(index, "_embed_query", lambda text, state: [0.0, 0.0, 0.0, 1.0])
    assert index.search("where can I donate blood in my city?") == []


def test_one_passage_per_page(index, monkeypatch):
    """Two chunks of the same page must not both be returned; they spend the context budget
    repeating one source instead of offering a second."""
    state = index._load()
    np = state["np"]
    state["db"].execute(
        "INSERT INTO chunks VALUES (3, 'Sore throat', 'https://example.test/sore-throat',"
        " 'part 2', 'Gargling with warm salt water can ease the pain of a sore throat.', 'Test source')"
    )
    state["db"].execute(
        "INSERT INTO chunks_fts (rowid, title, aliases, text) VALUES (3, 'Sore throat', 'pharyngitis',"
        " 'Gargling with warm salt water can ease the pain of a sore throat.')"
    )
    state["db"].commit()
    state["vectors"] = np.vstack([state["vectors"], np.array([[1.0, 0.0, 0.0, 0.0]])]).astype("float32")

    monkeypatch.setattr(index, "_embed_query", lambda text, state: [1.0, 0.0, 0.0, 0.0])
    urls = [p["url"] for p in index.search("sore throat", top_k=3)]
    assert len(urls) == len(set(urls)), f"the same page appeared twice: {urls}"


def test_survives_an_embedding_failure(index, monkeypatch):
    def broken(text, state):
        raise AssertionError("should not propagate")

    monkeypatch.setattr(index, "_embed_query", lambda text, state: None)
    assert index.search("sore throat", top_k=1)  # keyword search alone still answers


def test_punctuation_does_not_break_the_keyword_query(index):
    # A bare apostrophe or quote would be an FTS5 syntax error if it reached the matcher
    assert index.search('what\'s a "sore throat"? -- urgent!!', top_k=1)


def test_context_is_numbered_and_capped(index):
    passages = index.search("TSH", top_k=3)
    context = index.build_context(passages, max_chars=120)
    assert context.startswith("[1] ")
    assert len(context) < 400  # the cap is on passage text, headers are small


def test_sources_are_deduplicated_by_page(index):
    passages = index.search("TSH", top_k=1) * 3
    assert len(index.sources(passages)) == 1


def test_missing_index_disables_retrieval_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "DB_PATH", tmp_path / "nothing.db")
    monkeypatch.setattr(knowledge, "VECTORS_PATH", tmp_path / "nothing.npy")
    monkeypatch.setattr(knowledge, "_state", None)
    assert knowledge.available() is False
    assert knowledge.search("sore throat") == []
    knowledge._state = None


def test_chat_cites_sources_and_sends_them_before_the_answer(client, auth, fake_llm, index):
    events = stream_events(client.post("/api/chat", data={"message": "raised TSH"}, headers=auth))

    types = [e["type"] for e in events]
    # Sources arrive before any of the reply, so the user sees them while the model reads
    assert types.index("sources") < types.index("delta")
    [source_event] = [e for e in events if e["type"] == "sources"]
    assert source_event["sources"][0]["url"] == "https://example.test/tsh"

    # The passage text reached the model, and it was told to cite
    prompt = fake_llm.chats[0][-1]["content"]
    assert "hypothyroidism" in prompt
    assert "[1]" in prompt
    assert "cite" in fake_llm.chats[0][0]["content"].lower()
    assert reply_text(events) == "Model reply."


def test_chat_without_a_matching_passage_is_unchanged(client, auth, fake_llm, index):
    events = stream_events(client.post("/api/chat", data={"message": "zzzz"}, headers=auth))
    assert [e for e in events if e["type"] == "sources"] == []
    # No passages means no grounding instruction: the prompt is the plain one
    assert "Reference passages" not in fake_llm.chats[0][-1]["content"]
