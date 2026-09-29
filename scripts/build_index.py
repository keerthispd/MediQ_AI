"""Build the retrieval index from MedlinePlus health topics.

MedlinePlus is written by the US National Library of Medicine for patients rather than clinicians,
which is the register MediQ answers in, and it is in the public domain. The bulk XML carries one
plain-language summary per topic plus the synonyms people actually search with ("HbA1C" for A1C).

Run it with the backend's virtualenv, with Ollama running:

    python scripts/build_index.py

It writes database/knowledge.db (text + a full-text index) and database/knowledge.npy (embeddings).
Both are build artefacts: delete them and re-run to rebuild.
"""
import argparse
import html
import io
import json
import re
import sqlite3
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

import fetch_drugs  # noqa: E402

from backend.services import llm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "knowledge.db"
VECTORS_PATH = ROOT / "database" / "knowledge.npy"

LISTING_URL = "https://medlineplus.gov/xml.html"
SOURCE_NAME = "MedlinePlus (NIH)"

# Roughly 300 tokens. Every retrieved token costs real time on CPU at generation, so chunks are
# kept near the size of a typical topic summary rather than as large as the context allows.
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
EMBED_BATCH = 64


def log(message):
    print(message, flush=True)


def latest_archive_url() -> str:
    """MedlinePlus publishes a dated bundle each day; take the newest one listed."""
    with urllib.request.urlopen(LISTING_URL, timeout=60) as response:
        page = response.read().decode("utf-8", "replace")
    urls = re.findall(r'href="(https://medlineplus\.gov/xml/mplus_topics_compressed_[\d-]+\.zip)"', page)
    if not urls:
        raise SystemExit("Could not find a MedlinePlus archive to download.")
    return sorted(urls)[-1]


def download_topics(cache: Path) -> bytes:
    if cache.exists():
        log(f"Using cached {cache.name}")
        return cache.read_bytes()
    url = latest_archive_url()
    log(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        data = z.read(z.namelist()[0])
    cache.write_bytes(data)
    log(f"Saved {cache.name} ({len(data) / 1e6:.1f} MB)")
    return data


def plain_text(markup: str) -> str:
    """MedlinePlus summaries are HTML fragments; keep the words and drop the markup."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", markup or "", flags=re.S | re.I)
    text = re.sub(r"</(p|li|div|h\d)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sentences(text: str):
    # Good enough for prose: split after a full stop that is followed by a capital letter.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def chunk_text(text: str):
    """Pack whole sentences up to the size limit, so no chunk ends mid-thought."""
    chunks, current = [], ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        # A short tail reads better merged into the previous chunk than stranded alone
        if chunks and len(current) < MIN_CHUNK_CHARS:
            chunks[-1] = f"{chunks[-1]} {current}"
        else:
            chunks.append(current)
    return chunks


def parse_topics(xml_bytes: bytes):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    for topic in root:
        if topic.get("language") != "English":
            continue
        summary = topic.find("full-summary")
        if summary is None:
            continue
        text = plain_text(summary.text)
        if not text:
            continue
        yield {
            "title": topic.get("title") or "",
            "url": topic.get("url") or "",
            "aliases": [a.text for a in topic.findall("also-called") if a.text],
            "groups": [g.text for g in topic.findall("group") if g.text],
            "text": text,
        }


def build_chunks(topics):
    """One row per chunk. The title is prepended to the embedded text so a chunk taken out of its
    page still carries what it is about."""
    rows = []
    for topic in topics:
        pieces = chunk_text(topic["text"])
        for index, piece in enumerate(pieces):
            rows.append({
                "title": topic["title"],
                "url": topic["url"],
                "aliases": " ".join(topic["aliases"]),
                "section": f"part {index + 1} of {len(pieces)}" if len(pieces) > 1 else "",
                "text": piece,
                "source": SOURCE_NAME,
                "embed_text": f"{topic['title']}. {piece}",
            })
    return rows


def build_drug_chunks(records):
    """One chunk per labelled section, so "side effects of X" and "what is X for" are separate
    passages rather than one long label nobody asked all of."""
    rows = []
    for record in records:
        # Brand names go in the searchable aliases: people ask about Advil, not ibuprofen
        aliases = " ".join([record["drug"], *record["brands"]])
        for label, text in record["sections"]:
            rows.append({
                "title": record["title"],
                "url": record["url"],
                "aliases": aliases,
                "section": label,
                "text": text,
                "source": fetch_drugs.SOURCE_NAME,
                "embed_text": f"{record['title']} medicine. {label}. {text}",
            })
    return rows


def embed_all(rows):
    import numpy as np

    vectors, started = [], time.perf_counter()
    with httpx.Client(base_url=llm.OLLAMA_BASE_URL, timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        for start in range(0, len(rows), EMBED_BATCH):
            batch = [r["embed_text"] for r in rows[start:start + EMBED_BATCH]]
            response = client.post("/api/embed", json={
                "model": llm.EMBED_MODEL, "input": batch, "keep_alive": llm.KEEP_ALIVE,
            })
            if response.status_code != 200:
                raise SystemExit(
                    f"Embedding failed ({response.status_code}). Is the model installed?\n"
                    f"  ollama pull {llm.EMBED_MODEL}\n  {response.text[:300]}"
                )
            vectors.extend(response.json()["embeddings"])
            done = min(start + EMBED_BATCH, len(rows))
            rate = done / (time.perf_counter() - started)
            log(f"  embedded {done}/{len(rows)} ({rate:.1f}/s, ~{(len(rows) - done) / rate / 60:.1f} min left)")

    matrix = np.asarray(vectors, dtype="float32")
    # Pre-normalise so search is a single dot product instead of a cosine at query time
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-9)
    return matrix


def write_database(rows, source):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE chunks (
            id      INTEGER PRIMARY KEY,   -- matches the row order in knowledge.npy
            title   TEXT NOT NULL,
            url     TEXT NOT NULL,
            section TEXT,
            text    TEXT NOT NULL,
            source  TEXT NOT NULL
        );
        -- FTS5 ships with Python's sqlite3, so keyword search needs no extra dependency. It is
        -- what finds exact terms like "TSH" or "metformin" that embeddings blur together.
        CREATE VIRTUAL TABLE chunks_fts USING fts5(title, aliases, text, content='');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    db.executemany(
        "INSERT INTO chunks (id, title, url, section, text, source) VALUES (?,?,?,?,?,?)",
        [(i, r["title"], r["url"], r["section"], r["text"], r["source"]) for i, r in enumerate(rows)],
    )
    db.executemany(
        "INSERT INTO chunks_fts (rowid, title, aliases, text) VALUES (?,?,?,?)",
        [(i, r["title"], r["aliases"], r["text"]) for i, r in enumerate(rows)],
    )
    db.executemany("INSERT INTO meta (key, value) VALUES (?,?)", list(source.items()))
    db.commit()
    db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Only index this many topics (for a quick trial run)")
    parser.add_argument("--no-drugs", action="store_true", help="Skip the FDA drug labels")
    parser.add_argument("--drugs", nargs="*", help="Only these drugs (for a quick trial run)")
    args = parser.parse_args()

    cache = ROOT / "database" / "medlineplus_topics.xml"
    cache.parent.mkdir(parents=True, exist_ok=True)
    topics = list(parse_topics(download_topics(cache)))
    if args.limit:
        topics = topics[:args.limit]
    log(f"Parsed {len(topics)} English topics")

    rows = build_chunks(topics)

    if not args.no_drugs:
        names = args.drugs or fetch_drugs.DRUGS
        log(f"Fetching {len(names)} drug labels from openFDA…")
        drug_rows = build_drug_chunks(fetch_drugs.fetch_all(names, log=log))
        log(f"  {len(drug_rows)} drug chunks")
        rows += drug_rows

    chars = sum(len(r["text"]) for r in rows)
    log(f"Built {len(rows)} chunks (~{chars // 4:,} tokens, {chars / len(rows):.0f} chars each)")

    log(f"Embedding with {llm.EMBED_MODEL}…")
    matrix = embed_all(rows)

    import numpy as np
    np.save(VECTORS_PATH, matrix)
    sources = sorted({r["source"] for r in rows})
    write_database(rows, {
        "source": " + ".join(sources),
        "licence": "Public domain (US National Library of Medicine; US Food and Drug Administration)",
        "embed_model": llm.EMBED_MODEL,
        "dimensions": str(matrix.shape[1]),
        "chunks": str(len(rows)),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log(f"\nWrote {DB_PATH.name} and {VECTORS_PATH.name}")
    log(f"  {len(rows)} chunks, {matrix.shape[1]} dimensions, {matrix.nbytes / 1e6:.1f} MB of vectors")


if __name__ == "__main__":
    main()
