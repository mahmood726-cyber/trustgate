"""
TrustGate enrichment data builder.
Produces:
  1. data/review_groups.csv   — review_id_prefix -> review_group (primary domain)
  2. data/who_matches.json    — review_id_prefix -> bool (WHO medicine match)
  3. data/nice_cache.json     — {} placeholder (to be enriched via NICE API later)
"""

import csv
import json
import re
import sys
import io
from pathlib import Path
from collections import Counter

# ── paths ─────────────────────────────────────────────────────────────────────
SOURCE_CSV = Path(r"C:\Archive\Backups\ctgov-search-strategies_backup_20260113_193020\data\review_conditions.csv")
DATA_DIR   = Path(r"C:\Models\TrustGate\data")
WHO_CSV    = DATA_DIR / "who_medicines.csv"

OUT_GROUPS = DATA_DIR / "review_groups.csv"
OUT_WHO    = DATA_DIR / "who_matches.json"
OUT_NICE   = DATA_DIR / "nice_cache.json"

# ── helpers ───────────────────────────────────────────────────────────────────
_PUB_RE = re.compile(r"_pub\d+_data$", re.IGNORECASE)

def strip_suffix(dataset_id: str) -> str:
    """CD000028_pub4_data  →  CD000028"""
    return _PUB_RE.sub("", dataset_id)

def first_condition(conditions_str: str) -> str:
    """'cardiovascular, neurological' → 'Cardiovascular'"""
    if not conditions_str:
        return "Other"
    first = conditions_str.split(",")[0].strip()
    return first.capitalize() if first else "Other"

# ── load WHO medicines ────────────────────────────────────────────────────────
who_medicines: list[str] = []
with open(WHO_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["medicine_name"].strip().lower()
        if name:
            who_medicines.append(name)

print(f"WHO medicines loaded: {len(who_medicines)}")
print(f"  {who_medicines}")

# ── read source CSV ───────────────────────────────────────────────────────────
rows: list[dict] = []
with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"\nSource rows: {len(rows)}")

# ── TASK 1: review_groups.csv ─────────────────────────────────────────────────
seen_prefixes: set[str] = set()
group_records: list[dict] = []

for row in rows:
    prefix = strip_suffix(row["dataset_id"].strip())
    if prefix in seen_prefixes:
        continue          # keep first occurrence
    seen_prefixes.add(prefix)
    group = first_condition(row.get("conditions", ""))
    group_records.append({"review_id_prefix": prefix, "review_group": group})

with open(OUT_GROUPS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["review_id_prefix", "review_group"])
    writer.writeheader()
    writer.writerows(group_records)

domain_counts = Counter(r["review_group"] for r in group_records)
print(f"\nTask 1 — review_groups.csv")
print(f"  Unique review prefixes written: {len(group_records)}")
print(f"  Domain distribution:")
for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
    print(f"    {domain:<30s} {count:>4d}")

# ── TASK 2: who_matches.json ──────────────────────────────────────────────────
who_matches: dict[str, bool] = {}
match_count = 0
match_details: list[str] = []   # for summary

# Build a set of prefixes we have already de-duplicated; we want one bool per prefix.
# We union across all rows with the same prefix (any match = True).
prefix_match_map: dict[str, bool] = {}

for row in rows:
    prefix = strip_suffix(row["dataset_id"].strip())
    if prefix_match_map.get(prefix, False):
        continue       # already True, no need to re-check

    haystack = " ".join([
        row.get("sample_analyses", "") or "",
        row.get("search_keywords", "") or "",
    ]).lower()

    matched = False
    for med in who_medicines:
        # whole-word match to avoid false positives (e.g. "aspirin" inside "warfarin" — actually fine,
        # but "in" inside "insulin" etc. — use word-boundary)
        pattern = r"\b" + re.escape(med) + r"\b"
        if re.search(pattern, haystack):
            matched = True
            match_details.append(f"{prefix} <- {med}")
            break

    prefix_match_map[prefix] = matched

# fill who_matches for every prefix in group_records (same order)
for record in group_records:
    p = record["review_id_prefix"]
    who_matches[p] = prefix_match_map.get(p, False)

match_count = sum(1 for v in who_matches.values() if v)

with open(OUT_WHO, "w", encoding="utf-8") as f:
    json.dump(who_matches, f, indent=2)

print(f"\nTask 2 — who_matches.json")
print(f"  Reviews with WHO medicine match: {match_count} / {len(who_matches)}")
print(f"  Match rate: {match_count/len(who_matches)*100:.1f}%")
print(f"  Sample matches (first 10):")
for line in match_details[:10]:
    print(f"    {line}")
if len(match_details) > 10:
    print(f"    ... ({len(match_details) - 10} more)")

# ── TASK 3: nice_cache.json ───────────────────────────────────────────────────
# Placeholder — real enrichment requires NICE API/scraping (501 HTTP calls).
# The engine reads this dict as {review_id_prefix: guideline_count}.
# An empty dict means all counts default to 0 downstream.
nice_cache: dict = {}

with open(OUT_NICE, "w", encoding="utf-8") as f:
    json.dump(nice_cache, f, indent=2)

print(f"\nTask 3 — nice_cache.json")
print(f"  Created placeholder empty dict (0 entries).")
print(f"  NOTE: Enrich via NICE Evidence Search API: https://www.evidence.nhs.uk/search?q={{CD_number}}")
print(f"  501 reviews would require ~501 HTTP GET requests.")

print(f"\nAll 3 files written to {DATA_DIR}")
print(f"  {OUT_GROUPS}")
print(f"  {OUT_WHO}")
print(f"  {OUT_NICE}")
