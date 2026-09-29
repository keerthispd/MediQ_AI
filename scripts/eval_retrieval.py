"""Measure retrieval quality against scripts/eval_cases.json.

Two things are scored, because retrieval can fail in two directions:

  recall     - for answerable questions, did an expected page reach the top k?
  abstention - for questions the corpus cannot answer, did retrieval correctly return nothing?

The second matters as much as the first here. Keyword search finds something for almost any shared
word, and citing a page that does not answer the question is worse than citing nothing at all.

    python scripts/eval_retrieval.py            # score the current index
    python scripts/eval_retrieval.py --verbose  # also show what was retrieved
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import knowledge  # noqa: E402

CASES = Path(__file__).with_name("eval_cases.json")


def matches(expected, passages):
    """A case passes if any expected name appears in a retrieved title (case-insensitive)."""
    titles = " | ".join(p["title"] for p in passages).lower()
    return any(name.lower() in titles for name in expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=knowledge.TOP_K)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip cases whose corpus is not indexed yet (marked with 'needs')")
    args = parser.parse_args()

    if not knowledge.available():
        raise SystemExit("No index found. Run: python scripts/build_index.py")

    data = json.loads(CASES.read_text(encoding="utf-8"))
    meta = knowledge.describe()
    print(f"index: {meta.get('chunks')} chunks, {meta.get('source')}, {meta.get('embed_model')}")
    print(f"top_k={args.top_k}  min_similarity={knowledge.MIN_SIMILARITY}\n")

    groups = {"answerable": [0, 0], "drugs": [0, 0], "abstention": [0, 0]}
    failures = []

    for case in data["cases"]:
        passages = knowledge.search(case["q"], top_k=args.top_k)
        if case.get("expect_none"):
            group, ok = "abstention", not passages
        else:
            group = "drugs" if case.get("needs") == "drugs" else "answerable"
            ok = matches(case["expect_any"], passages)
        groups[group][1] += 1
        groups[group][0] += ok

        if args.verbose or not ok:
            got = ", ".join(p["title"] for p in passages) or "(nothing)"
            want = "nothing" if case.get("expect_none") else " / ".join(case["expect_any"])
            print(f"  {'PASS' if ok else 'FAIL'}  {case['q'][:52]:54s}")
            print(f"        want: {want}")
            print(f"        got : {got}")
        if not ok:
            failures.append(case["q"])

    print("\n" + "=" * 64)
    total_ok = total_n = 0
    for name, (ok, n) in groups.items():
        if not n:
            continue
        total_ok += ok
        total_n += n
        print(f"  {name:12s} {ok:2d}/{n:<2d}  {ok / n * 100:5.1f}%")
    print(f"  {'OVERALL':12s} {total_ok:2d}/{total_n:<2d}  {total_ok / total_n * 100:5.1f}%")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
