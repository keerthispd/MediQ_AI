"""Fetch patient-facing drug information from openFDA.

MedlinePlus drug pages are AHFS content copyrighted by ASHP, so they cannot be reused. FDA drug
labels are US government works in the public domain, and openFDA serves them as JSON.

Two deliberate limits:

* **No dosage text.** MediQ is told never to recommend doses, so `dosage_and_administration` is
  never indexed - putting it in front of the model only invites it to break that rule.
* **Single-ingredient labels only.** Searching a generic name also matches combination products
  ("metformin" matches a sitagliptin/metformin tablet), which answer a different question.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

API = "https://api.fda.gov/drug/label.json"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
SOURCE_NAME = "FDA drug label (DailyMed)"

# The medicines patients actually ask about: common prescriptions, and the over-the-counter
# painkillers and remedies people take without being told to.
DRUGS = [
    "metformin", "atorvastatin", "simvastatin", "rosuvastatin", "amlodipine", "lisinopril",
    "ramipril", "losartan", "bisoprolol", "atenolol", "propranolol", "furosemide",
    "omeprazole", "lansoprazole", "ranitidine", "famotidine",
    "ibuprofen", "naproxen", "aspirin", "acetaminophen", "paracetamol", "diclofenac",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "penicillin",
    "cephalexin", "clarithromycin", "nitrofurantoin", "trimethoprim",
    "levothyroxine", "warfarin", "clopidogrel", "apixaban", "rivaroxaban",
    "sertraline", "fluoxetine", "citalopram", "escitalopram", "amitriptyline", "mirtazapine",
    "gabapentin", "pregabalin", "tramadol", "codeine", "morphine",
    "prednisone", "prednisolone", "salbutamol", "albuterol", "montelukast", "fluticasone",
    "insulin", "gliclazide", "glipizide", "sitagliptin", "empagliflozin",
    "cetirizine", "loratadine", "fexofenadine", "ondansetron", "metoclopramide",
    "allopurinol", "alendronate", "tamsulosin", "finasteride", "sildenafil",
    "hydrochlorothiazide", "spironolactone", "digoxin", "methotrexate", "azathioprine",
]

# Kept in retrieval priority order. Patient-facing sections first; adverse reactions last because
# they are long and clinical, and get truncated hardest.
FIELDS = [
    # A boxed warning is the FDA's most serious safety warning ("WARNING: BLEEDING RISK - can
    # cause major or fatal bleeding"). It goes first so that when a drug carries one, it is the
    # passage most likely to be retrieved and put in front of the model.
    ("boxed_warning", "Most serious warning", 1200),
    ("indications_and_usage", "What it is used for", 1500),
    ("information_for_patients", "What patients are told", 2500),
    ("patient_medication_information", "Patient information", 2500),
    ("warnings_and_cautions", "Warnings", 1500),
    ("warnings", "Warnings", 1500),
    ("adverse_reactions", "Side effects", 1800),
    ("contraindications", "Who should not take it", 800),
]

# Labels open with a section number and an ALL-CAPS heading ("17 PATIENT COUNSELING INFORMATION").
# Stop before the first normally-capitalised word so the sentence itself survives intact.
HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\s+(?:[A-Z][A-Z\-/&,.']*\s+)+(?=[A-Z][a-z])")


def clean(text: str, limit: int) -> str:
    text = HEADING.sub("", " ".join(text) if isinstance(text, list) else text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        # Cut at a sentence end so a passage never stops mid-clause
        cut = text[:limit].rsplit(". ", 1)[0]
        text = (cut + ".") if len(cut) > limit * 0.5 else text[:limit]
    return text


def _get(url: str, attempts: int = 3) -> Optional[Dict]:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # no label matches; not an error worth retrying
            if e.code == 429 and attempt < attempts - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, ValueError):
            if attempt < attempts - 1:
                time.sleep(2)
                continue
            return None
    return None


def _search(expression: str, limit: int = 5) -> List[Dict]:
    data = _get(f"{API}?search={urllib.parse.quote(expression)}&limit={limit}")
    return data.get("results", []) if data else []


def _single_ingredient(result: Dict, drug: str) -> bool:
    names = [n.upper() for n in result.get("openfda", {}).get("generic_name", [])]
    return any(drug.upper() == n or n.startswith(drug.upper()) and " AND " not in n for n in names)


def fetch_drug(drug: str) -> Optional[Dict]:
    """Best available label for one drug, preferring richer patient-facing text."""
    results = _search(f'openfda.generic_name.exact:"{drug.upper()}"') \
        or _search(f'openfda.generic_name:"{drug}"') \
        or _search(f'openfda.substance_name:"{drug.upper()}"')
    candidates = [r for r in results if _single_ingredient(r, drug)] or results
    if not candidates:
        return None

    def richness(result):
        return sum(len(" ".join(result[f])) for f, _, _ in FIELDS if f in result)

    best = max(candidates, key=richness)
    sections = []
    seen_labels = set()
    for field, label, limit in FIELDS:
        if field not in best or label in seen_labels:
            continue
        text = clean(best[field], limit)
        if len(text) > 80:
            sections.append((label, text))
            seen_labels.add(label)
    if not sections:
        return None

    brands = [b.title() for b in best.get("openfda", {}).get("brand_name", [])[:3]]
    return {
        "drug": drug,
        "title": drug.title(),
        "brands": brands,
        "set_id": best.get("set_id", ""),
        "url": DAILYMED + best.get("set_id", "") if best.get("set_id") else
               "https://www.fda.gov/drugs",
        "sections": sections,
    }


def fetch_all(drugs: List[str] = None, log=print) -> List[Dict]:
    drugs = drugs or DRUGS
    out, missing = [], []
    for index, drug in enumerate(drugs, 1):
        record = fetch_drug(drug)
        if record:
            out.append(record)
        else:
            missing.append(drug)
        if index % 10 == 0 or index == len(drugs):
            log(f"  fetched {index}/{len(drugs)} ({len(out)} with usable text)")
        time.sleep(0.3)  # openFDA allows 240 requests/minute unauthenticated
    if missing:
        log(f"  no usable label for: {', '.join(missing)}")
    return out


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or ["metformin", "ibuprofen", "omeprazole"]
    for record in fetch_all(names):
        print(f"\n=== {record['title']} ({', '.join(record['brands']) or 'no brand'}) ===")
        print(f"    {record['url']}")
        for label, text in record["sections"]:
            print(f"  [{label}] {len(text)} chars: {text[:130]}")
