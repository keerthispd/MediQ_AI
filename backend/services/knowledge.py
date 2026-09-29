"""Grounded answers: find passages from a trusted medical source to answer from.

Retrieval is hybrid. Keyword search (SQLite FTS5/BM25) finds exact terms - "TSH", "HbA1c",
"metformin" - that embeddings blur into their neighbours, and vector search finds passages that
answer the question without sharing its words ("my chest feels tight" -> angina). Their rankings are
combined with reciprocal rank fusion, which needs no score calibration between the two.

The index is built by scripts/build_index.py. With no index present every call returns nothing and
the app answers exactly as it did before.
"""
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("KNOWLEDGE_DB", ROOT / "database" / "knowledge.db"))
VECTORS_PATH = Path(os.getenv("KNOWLEDGE_VECTORS", ROOT / "database" / "knowledge.npy"))

# Each retrieved token is read by the model before it writes a word, and on CPU that is the single
# biggest cost of answering: measured at roughly 100 tokens/second. Three short passages is the
# budget that keeps a grounded reply near the speed of an ungrounded one.
TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "3"))
CANDIDATES = 20
MAX_CONTEXT_CHARS = int(os.getenv("KNOWLEDGE_MAX_CHARS", "2600"))
RRF_K = 60  # standard reciprocal-rank-fusion constant; damps the influence of low ranks
# Cosine similarity a passage must reach before it is allowed to ground an answer. Calibrated by
# measuring questions the corpus can answer against ones it cannot; see docs/RAG.md.
MIN_SIMILARITY = float(os.getenv("KNOWLEDGE_MIN_SIMILARITY", "0.35"))

_lock = threading.Lock()
_state: Optional[Dict] = None


def _load():
    """Open the index once, on first use. Returns None when it hasn't been built."""
    global _state
    if _state is not None:
        return _state.get("ready") and _state or None
    with _lock:
        if _state is not None:
            return _state.get("ready") and _state or None
        if not DB_PATH.exists() or not VECTORS_PATH.exists():
            logger.info("No knowledge index found; answers will not be grounded in sources.")
            _state = {"ready": False}
            return None
        try:
            import numpy as np

            vectors = np.load(VECTORS_PATH)
            # check_same_thread=False: read-only, and requests are served from a thread pool
            db = sqlite3.connect(DB_PATH, check_same_thread=False)
            db.row_factory = sqlite3.Row
            meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
            _state = {"ready": True, "db": db, "vectors": vectors, "meta": meta, "np": np}
            logger.info("Knowledge index: %s chunks from %s", meta.get("chunks"), meta.get("source"))
            return _state
        except Exception as e:  # noqa: BLE001 - a broken index must not take the app down
            logger.warning("Could not open the knowledge index: %s", e)
            _state = {"ready": False}
            return None


def available() -> bool:
    return _load() is not None


def describe() -> Dict:
    state = _load()
    if not state:
        return {"ready": False}
    return {"ready": True, **state["meta"]}


def _embed_query(text: str, state) -> Optional[List[float]]:
    from backend.services import llm

    try:
        # Uses the pooled client in llm: a fresh connection per query measured seconds, not
        # milliseconds, which would have dwarfed the search itself.
        return llm.embed([text], model=state["meta"].get("embed_model"))[0]
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        # Keyword search alone still returns useful passages, so this is not fatal
        logger.warning("Query embedding failed, falling back to keyword search: %s", e)
        return None


def _fts_query(text: str) -> str:
    """Turn a sentence into an FTS5 OR-query, dropping anything that would be a syntax error."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{1,}", text)
    return " OR ".join(f'"{w}"' for w in words[:24])


def _keyword_ranking(text: str, state) -> List[int]:
    query = _fts_query(text)
    if not query:
        return []
    try:
        rows = state["db"].execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (query, CANDIDATES),
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error as e:
        logger.warning("Keyword search failed: %s", e)
        return []


def _vector_ranking(text: str, state):
    """Return (ranked ids, similarity scores for every chunk or None).

    The full score array comes back so a passage found only by keyword search can still be scored
    for relevance, rather than being dropped for lacking a score.
    """
    embedding = _embed_query(text, state)
    if embedding is None:
        return [], None
    np = state["np"]
    query = np.asarray(embedding, dtype="float32")
    norm = float(np.linalg.norm(query))
    if norm == 0:
        return [], None
    # Vectors were normalised at build time, so a dot product is the cosine similarity
    scores = state["vectors"] @ (query / norm)
    # argpartition raises if asked for more neighbours than the index holds
    wanted = min(CANDIDATES, scores.shape[0])
    if wanted == 0:
        return [], scores
    top = np.argpartition(scores, -wanted)[-wanted:]
    ranked = [int(i) for i in top[np.argsort(scores[top])[::-1]]]
    return ranked, scores


def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Return the passages most likely to answer `query`, best first."""
    state = _load()
    if not state or not query.strip():
        return []

    vector_ids, scores = _vector_ranking(query, state)
    rankings = [r for r in (_keyword_ranking(query, state), vector_ids) if r]
    if not rankings:
        return []

    fused: Dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + position + 1)

    def similarity_of(chunk_id):
        return None if scores is None else float(scores[chunk_id])

    # Keyword search always returns something for any shared word, so without this a question the
    # corpus cannot answer would still be "grounded" in whatever matched loosely. Citing an
    # unrelated page is worse than citing nothing, so drop weak matches entirely.
    if scores is not None:
        if max((similarity_of(i) for i in fused), default=0.0) < MIN_SIMILARITY:
            logger.debug("No passage was close enough for %r; answering without sources.", query)
            return []
        fused = {i: s for i, s in fused.items() if similarity_of(i) >= MIN_SIMILARITY}

    # One passage per topic: two chunks of the same page crowd out a second opinion and spend the
    # context budget saying the same thing twice.
    chosen, seen_urls = [], set()
    for chunk_id in sorted(fused, key=fused.get, reverse=True):
        row = state["db"].execute(
            "SELECT id, title, url, section, text, source FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None or row["url"] in seen_urls:
            continue
        seen_urls.add(row["url"])
        chosen.append({**dict(row), "score": fused[chunk_id], "similarity": similarity_of(chunk_id)})
        if len(chosen) == top_k:
            break
    return chosen


def build_context(passages: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format passages for the prompt, numbered so the model can cite them as [1], [2]."""
    blocks, used = [], 0
    for number, passage in enumerate(passages, 1):
        text = passage["text"]
        if used + len(text) > max_chars:
            remaining = max_chars - used
            if remaining < 300:
                break
            text = text[:remaining].rsplit(". ", 1)[0] + "."
        blocks.append(f"[{number}] {passage['title']} ({passage['source']})\n{text}")
        used += len(text)
    return "\n\n".join(blocks)


def sources(passages: List[Dict]) -> List[Dict]:
    """The citation list shown under the answer, de-duplicated by page."""
    seen, out = set(), []
    for passage in passages:
        if passage["url"] in seen:
            continue
        seen.add(passage["url"])
        out.append({"title": passage["title"], "url": passage["url"], "source": passage["source"]})
    return out
