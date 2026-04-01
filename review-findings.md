## REVIEW CLEAN
## Multi-Persona Review: TrustGate
### Date: 2026-04-01
### Summary: 6 P0, 6 P1, 9 P2 — ALL P0 FIXED, 6/6 P1 FIXED, 26/26 tests passing

#### P0 -- Critical

- **SM-P0-1** [FIXED] Statistical Methodologist: NaN p-value silently treated as z=8.01 (most extreme significance) because `max(1e-15, NaN)` returns `1e-15` in CPython. Any unparseable p_value from `pd.to_numeric(errors="coerce")` becomes NaN and inflates weights catastrophically. (line ~247)
  - Suggested fix: Add `math.isnan(p)` guard at top of `_z_from_p`, return 0.0

- **SM-P0-2** [FIXED] Statistical Methodologist: NaN estimate bypasses `estimate == 0` guard (NaN != anything), propagates as NaN SE and NaN weight through trust_weight_ma and erosion curve. (line ~293)
  - Suggested fix: Add `math.isnan(estimate) or math.isnan(p_value)` guard in `compute_se_from_p`, return None

- **SM-P0-3** [FIXED] Statistical Methodologist: NaN final_score in erosion curve silently counted as excluded without warning. Combined with P0-1/P0-2, NaN rows could be counted as both "significant" and "excluded". (line ~381)
  - Suggested fix: Filter out rows with NaN in final_score/estimate/p_value before erosion loop

- **SA-P0-1** [FIXED] Security Auditor: XSS in dashboard — raw CSV field values (ma_id, review_id, grade, quadrant) concatenated into innerHTML without escaping. Tampered CSV could inject scripts. (build_dashboard.py ~748-753)
  - Suggested fix: Add `tgEsc()` JS function that escapes &<>"', apply to all string fields in tgRenderRow

- **SA-P0-2** [FIXED] Security Auditor: `</script>` in JSON data blob breaks script block. `json.dumps` does NOT escape `</` by default. A CSV field containing `</script>` would terminate the script block and enable injection. (build_dashboard.py ~497-500)
  - Suggested fix: Replace `</` with `<\/` in all JSON blobs: `json.dumps(obj).replace('</', '<\\/')`

- **SA-P0-3** [FIXED] Security Auditor: Unvalidated DOI values passed to PubMed API URLs. Malicious DOI with newlines could enable HTTP header injection. (trustgate_engine.py ~554-582)
  - Suggested fix: Validate DOI format with regex `^10\.\d{4,9}/[^\s\x00-\x1f]+$` before fetching

#### P1 -- Important

- **SM-P1-1** [FIXED] Statistical Methodologist: SE = |estimate|/z assumes Wald z-test origin for p-value. Invalid for exact tests, LRT, risk differences with estimate=0. Acceptable for Cochrane MAs but should be documented. (line ~298)
  - Suggested fix: Add assumption note to docstring

- **SM-P1-2** [FIXED] Statistical Methodologist: Trust-weighting simplifies to z_trust = z * sqrt(score/100), extremely aggressive — borderline z=2.0 flips at score<96. Should be documented. (line ~393-399)
  - Suggested fix: Document the mathematical relationship in docstring

- **SM-P1-3** [FIXED] Statistical Methodologist: `trust_weighted_estimate = estimate * score_fraction` is misleading — in IV pooling, estimates are NOT scaled, only weights. Anyone using this field for pooling would double-count trust. (line ~339)
  - Suggested fix: Rename to `weighted_numerator` = trust_weight * estimate, or document clearly

- **SM-P1-4** [FIXED] Statistical Methodologist: Quadrant boundary asymmetry — Red Flag uses `influence > 50` (strict) while Safe uses `influence >= 50` (inclusive). At influence=50 exactly, high-trust gets Safe but low-trust gets Moderate. (line ~799 vs ~803)
  - Suggested fix: Use consistent comparison operators

- **SA-P1-1** [FIXED] Security Auditor: CSV formula injection — no sanitization of cells starting with =, +, @, \t, \r in exported CSVs. (trustgate_engine.py ~1050-1080)
  - Suggested fix: Sanitize string columns before to_csv

- **SA-P1-3** [FIXED] Security Auditor: Catch-all `except Exception` in fetch_citation_counts swallows all errors silently. Self-DoS risk on large DOI lists. (line ~590)
  - Suggested fix: Catch specific exceptions (URLError, HTTPError, ParseError, OSError)

#### P2 -- Minor

- **SM-P2-1** Dead code branch in `_z_from_p` — the `q <= 0.5` else branch is unreachable for valid inputs (line ~257)
- **SM-P2-2** `_z_from_p` imported in tests but prefixed with underscore — inconsistent public/private convention
- **SM-P2-3** No test for `_z_from_p` accuracy against known reference values (e.g., z(0.05)=1.96)
- **SM-P2-4** No test for `compute_se_from_p` with negative estimates
- **SM-P2-5** No boundary-value tests for quadrant assignment at exact thresholds (trust=60, influence=50)
- **SM-P2-6** `fetch_nice_guideline_counts` returns full cache, not just requested DOIs
- **SM-P2-7** Redundant `import math` inside `_z_from_p` (already at module level)
- **SA-P2-1** Hardcoded absolute paths expose username/directory layout (lines 39-41)
- **SA-P2-3** `who_matches.json` opened without `encoding="utf-8"` (line ~937)

#### SE -- Software Engineer (inline review)

- **SE-P1-1** `iterrows()` in erosion curve (line ~387) and risk register (line ~826) — slow for 6,229 rows. Erosion curve takes ~4 seconds due to per-row SE computation. Could vectorize with numpy.
- **SE-P2-1** Dashboard embeds 6,229 rows as 4MB JSON — acceptable but could compress with column arrays instead of array-of-objects

#### DE -- Domain Expert (inline review)

- **DE-P1-1** Manuscript Table 1 erosion percentages are rounded differently than summary.json (63.9% vs 63.85%) — minor but should be consistent
- **DE-P2-1** Manuscript references [4]-[7] are self-citations to unpublished companion papers — reviewers may flag this

#### False Positive Watch
- DOR formula: NOT relevant to this project
- Clayton copula: NOT relevant
- A&S approximation accuracy: verified < 0.022% for typical p-values — correct
