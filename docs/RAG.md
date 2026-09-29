# Grounded answers (RAG)

MediQ answers from a local language model. A small model has shallow, frozen medical knowledge, it
cannot tell you where an answer came from, and it states wrong things as confidently as right ones.
Retrieval fixes the part that matters most for a medical app: answers are built from passages of a
trusted reference, and every answer can show its sources.

This document records how it is built, what it costs, and what is measured rather than assumed.

## Corpus

| Source | Licence | In use |
| --- | --- | --- |
| **MedlinePlus health topics** (US National Library of Medicine) | Public domain | Yes |
| **FDA drug labels** via openFDA / DailyMed | Public domain (US government work) | Yes |
| MedlinePlus **drug** pages | **© ASHP, all rights reserved** | **No - not reusable** |
| NHS Health A-Z | Open Government Licence v3 (attribution) | No |
| CDC | Public domain | No |
| WHO fact sheets | CC BY-NC-SA 3.0 IGO | No - non-commercial **and** share-alike |

MedlinePlus *health topics* are public domain, but its *drug* pages are AHFS Consumer Medication
Information, copyrighted by the American Society of Health-System Pharmacists. They cannot be
reused, so drug information comes from FDA labels instead, which are US government works.

Two limits on the drug labels, both deliberate:

- **Dosage text is never indexed.** MediQ is instructed never to recommend doses, and putting
  `dosage_and_administration` in front of the model only invites it to break that rule.
- **Single-ingredient labels only.** Searching a generic name also matches combination products - a
  "metformin" search returns a sitagliptin/metformin tablet - which answers a different question.

Brand names are indexed as search aliases, so "can I take Advil" reaches the ibuprofen label. A
drug's **boxed warning** - the FDA's most serious warning class - is indexed first, so when a drug
carries one it is the passage most likely to reach the model.

### Safety: what indexing drug labels changes

Basic drug questions *should* be answerable - what a medicine is for, and what its side effects are
- and now they are, from the label itself. But the same corpus makes two questions dangerous to
answer plainly, and they are routed differently:

| Message | Route | Why |
| --- | --- | --- |
| "how much paracetamol is a lethal dose" | **Crisis resources**, model never called | Answering from a real label would be actively harmful; someone asking is likely in crisis |
| "I think I overdosed on paracetamol" | **Emergency warning and first-aid steps**, then an ungrounded answer | This person needs urgent care now, not a helpline and a refusal - and not a wait while passages are read |
| "what are the side effects of metformin" | Answered normally, with sources | An ordinary question that deserves a grounded answer |

The distinction between the first two rows is the point: treating a reported overdose as a crisis
refusal would withhold help from someone who needs it, and treating a lethality question as an
ordinary drug query would answer it from an authoritative source. The safety layer runs before the
red-flag layer, so intent takes precedence over incident.

Dosage text is never indexed, so no retrieved passage can supply a dose for the model to repeat.
`tests/test_drug_safety.py` holds these cases.

MedlinePlus is the starting corpus because it is written for patients rather than clinicians, which
is the register MediQ answers in, and because it is public domain with no attribution conditions.

Deliberately avoided: PubMed abstracts are written for clinicians, hedge heavily, and report single
studies that contradict each other - the wrong shape for patient-facing answers. UpToDate, Mayo
Clinic and WebMD are copyrighted; scraping them is a licence violation.

Measured on the 2026-09-19 bundle:

- 2,033 topics, of which **1,017 are English** and 1,014 carry a summary
- 2.09 M characters, roughly **523 k tokens**
- median topic **287 tokens**; 34 % exceed 600 tokens and get split

## Architecture

Every component is either already in the project or a single small dependency.

| Layer | Choice | Added dependency |
| --- | --- | --- |
| Embeddings | Ollama `/api/embed`, `all-minilm` (384-d) | none - Ollama is already running |
| Keyword search | SQLite **FTS5** with `bm25()` | none - compiled into Python's `sqlite3` |
| Vector search | numpy brute force | numpy |
| Storage | SQLite + a `.npy` matrix | none |

No torch, no FAISS, no Chroma, no vector database. At this corpus size a brute-force dot product
takes **1.13 ms** over 15,000 chunks, against an LLM call costing ten thousand times that; a vector
database would add operational weight to optimise 0.01 % of the request. `sentence-transformers`
was rejected for the same reason - it pulls in torch (~250 MB) to replace a service already running.

Retrieval is **hybrid**, because the two methods fail differently:

- **BM25** finds exact tokens - `TSH`, `HbA1c`, `metformin` - which embeddings blur into their
  neighbours. Topic synonyms (`also-called`) are indexed too, so "HbA1C" reaches the A1C page.
- **Vector search** finds passages that answer a question without sharing its words ("my chest feels
  tight" → angina).

Their rankings are combined with reciprocal rank fusion, which needs no score calibration between
two incomparable scales.

## Cost, measured on the target machine

All figures from an i7-1355U running CPU-only (`ollama ps` reports `100% CPU`; the Iris Xe cannot
be used for inference).

### Building the index

| Corpus | Time |
| --- | --- |
| 2,197 chunks (MedlinePlus, current) | **~4 min** |
| 15,000 chunks (hypothetical, larger corpus) | ~27 min |

Throughput is **9.4 chunks/s** and plateaus after batch 32 - it is CPU-bound and Ollama serialises
requests, so larger batches do not help. Storage is **3.4 MB** of vectors plus ~2 MB of text.

### Answering a question

| Step | Cost | Share |
| --- | --- | --- |
| Embed the query | 24 ms | 0.3 % |
| Keyword search (FTS5) | ~2 ms | <0.1 % |
| Vector search | 1.13 ms | <0.1 % |
| **Model reading the retrieved passages** | **~7,000 ms** | **99.6 %** |

**Retrieval is free; reading what was retrieved is the entire cost.** This is the single most
important fact about RAG on this hardware, and it inverts the usual advice.

Prompt processing runs at roughly **100 tokens/s for text the model has not seen**, against
**505 tokens/s** for a cached prefix. Retrieved passages differ every query, so they can never be
cached:

| Retrieved context | Added time before the first word |
| --- | --- |
| 500 tokens | ~9.7 s |
| 1,000 tokens | ~11.8 s |
| 2,000 tokens | ~23.5 s |
| 3,000 tokens | ~26.1 s |

Budget it as `added_seconds ≈ retrieved_tokens / 100`.

## Design consequences

These follow from the measurements and run against conventional RAG guidance.

**Retrieve few, short passages.** The usual advice is to pass 5-10 chunks and let the model sort it
out. Here each chunk costs about two seconds of wall clock, so the default is **three** passages
capped at 2,600 characters (`KNOWLEDGE_TOP_K`, `KNOWLEDGE_MAX_CHARS`). Hybrid retrieval earns its
keep by making a small `k` sufficient.

**One passage per page.** Two chunks of the same topic spend the budget saying the same thing twice
instead of offering a second opinion, so results are de-duplicated by URL.

**No cross-encoder reranking.** It would cost seconds per candidate on this CPU. Hybrid retrieval
recovers most of the precision for about 2 ms.

**Weak matches are dropped rather than cited.** Keyword search returns something for almost any
shared word, so without a floor an unanswerable question would still be "grounded" in whatever
matched loosely. Citing a page that does not answer the question is worse than citing nothing, so a
passage must reach `KNOWLEDGE_MIN_SIMILARITY` (default **0.35**) or retrieval returns nothing and
the app answers as it did before.

That floor is calibrated, not guessed:

| | Maximum cosine similarity |
| --- | --- |
| Lowest scoring answerable question ("high TSH" → *Thyroid Tests*) | 0.380 |
| Highest scoring unanswerable question ("metformin side effects") | 0.329 |

The margin is thin (0.05). It separates 9 of 10 test questions correctly, and it is a provisional
value that should be re-derived from a proper evaluation set before anyone relies on it.

**Grounding is strictly additive.** No index, no embedding server, or no passage above the floor
means the app answers exactly as it did before. A medical assistant should never be *less*
available because a retrieval feature was added.

**Emergencies retrieve nothing.** A message the red-flag detector marks skips retrieval outright.
Grounding buys accuracy at `retrieved_tokens / 100` seconds before the first word - about six on a
full context - and the part of that reply that has to be right is not the model's to write: the
warning and the first-aid steps are fixed text from `backend/services/redflag.py`, on screen in
milliseconds. Six seconds of cited background, added to advice someone should be acting on right
now, is the wrong purchase. See **Emergency messages** in the README.

## Latency fixes found along the way

Two defects were costing seconds on **every** Ollama call, not only retrieval:

| | Median per call |
| --- | --- |
| New client per call, `localhost` (what the code did) | 4,833 ms |
| New client per call, `127.0.0.1` | 1,726 ms |
| Pooled client, `localhost` | 696 ms |
| **Pooled client, `127.0.0.1`** | **291 ms** |

A fresh HTTP connection per request, and `localhost` resolving to IPv6 `::1` first on Windows and
only succeeding after that attempt fails. The client is now pooled and the default base URL is
`127.0.0.1`.

## Known gaps

- **`all-minilm` may still limit quality.** An earlier note here claimed it retrieved loosely on
  lab-value questions, based on eyeballing one result. The evaluation set disproved that - it
  scores 17/17 on general health questions - so the claim is withdrawn. `nomic-embed-text` (768-d)
  may still retrieve better, but there is now a harness to prove it rather than assume it.
- **Inline citations and the latency budget cannot both be had.** Writing `[1]` markers is a
  capability of model size, not of prompting - moving the instruction to the end of the prompt,
  where small models weight it most, changed nothing (0/4 either way).

  | Model | Grounded reply, warm | Inline citations |
  | --- | --- | --- |
  | `qwen2.5:1.5b-instruct` (default) | 15-20 s | 0/4 |
  | `qwen3:4b-instruct` | median 74 s | 3/3 |

  The 4 B model cites correctly and blows the 60-second budget. The 1.5 B model stays well inside
  it and never writes a marker. The default keeps the faster model, because the **source list shown
  above every answer comes from retrieval rather than from the model, so it is accurate either
  way** - an inline marker is only as reliable as a small model's bookkeeping. Set
  `OLLAMA_FAST_MODEL=qwen3:4b-instruct` to trade the speed for inline markers.
- **Passage framing leaks into the answer.** A retrieved paragraph written about children produced
  "if your child has a sore throat..." for an adult asking about themselves; passages about
  postpartum depression produced advice about postpartum depression for someone who never said
  they had given birth. The model adopts a passage's *audience* along with its facts. The grounding
  instruction now tells it to use the facts and not the audience, which is a mitigation rather than
  a fix - a retrieval-side filter on population-specific pages would be stronger.

## Measuring quality

`scripts/eval_cases.json` holds questions with the pages that should answer them. Two things are
scored, because retrieval fails in two directions: **recall** (did an expected page reach the top
k?) and **abstention** (for questions the corpus cannot answer, did retrieval correctly return
nothing?). The second matters as much as the first - keyword search finds something for almost any
shared word, and citing a page that does not answer the question is worse than citing none.

```bash
python scripts/eval_retrieval.py            # score the current index
python scripts/eval_retrieval.py --verbose  # also show what came back
```

Run it before and after any change to the corpus, the embedding model, or the relevance floor.
Every claim about retrieval quality in this document comes from it.

Current scores, and what adding the FDA drug labels changed:

| | MedlinePlus only | + FDA drug labels |
| --- | --- | --- |
| Answerable health questions | 17/17 | 17/17 |
| Drug questions | **0/5** | **5/5** |
| Correct abstention | 4/4 | 4/4 |
| **Overall** | 21/26 (80.8 %) | **26/26 (100 %)** |

Abstention holding at 4/4 is the result worth checking on any corpus change: more chunks means
more chances that something clears the relevance floor for a question the corpus cannot answer.
Adding 351 drug chunks did not cause a single false grounding.

A 100 % score means the evaluation set is now too easy, not that retrieval is solved. Its value
from here is as a regression check, and the next useful work on it is adding questions it fails.

## Rebuilding

```bash
python scripts/build_index.py              # full corpus, ~5 minutes
python scripts/build_index.py --limit 40   # quick trial run
python scripts/build_index.py --no-drugs   # health topics only
```

Writes `database/knowledge.db` and `database/knowledge.npy`. Both are build artefacts, ignored by
git; delete them and re-run to rebuild. The app picks the index up at startup and runs without it.
