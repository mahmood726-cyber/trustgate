# sentinel:skip-file — hardcoded paths / templated placeholders are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""
TrustGate Engine
================
Trust-weighted significance analysis and guideline exposure mapping for
6,229 Cochrane meta-analyses.

Pipeline:
  EvidenceScore trust scores (0-100)  +  ActionableEvidence verdicts
  --> trust-adjusted p-values / significance rulings
  --> guideline exposure mapping via DOI / WHO / NICE lookups
  --> risk register export

Data sources
------------
SCORES_CSV   : C:\\Models\\EvidenceScore\\results\\scores.csv
VERDICTS_CSV : C:\\Models\\ActionableEvidence\\results\\verdicts.csv
DOIS_CSV     : C:\\Users\\user\\asreview_pairwise70_metadata.csv
"""

import json
import math
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CITATION_CACHE = DATA_DIR / "citation_cache.json"

_MODELS_DIR = BASE_DIR.parent  # C:\Models
_HOME_DIR = Path.home()
SCORES_CSV = _MODELS_DIR / "EvidenceScore" / "results" / "scores.csv"
VERDICTS_CSV = _MODELS_DIR / "ActionableEvidence" / "results" / "verdicts.csv"
DOIS_CSV = _HOME_DIR / "asreview_pairwise70_metadata.csv"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

EROSION_THRESHOLDS = [50, 60, 70, 80, 90]

INFLUENCE_WEIGHTS = {
    "citation_percentile": 0.4,
    "who_essential": 20.0,
    "nice_guideline_cap": 30.0,
    "nice_per_guideline": 10.0,
    "group_size_percentile": 0.1,
}

TRUST_LOW_THRESHOLD = 60
TRUST_HIGH_THRESHOLD = 80
INFLUENCE_LOW_THRESHOLD = 30
INFLUENCE_HIGH_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_scores(path: str | None = None) -> pd.DataFrame:
    """Load EvidenceScore trust scores CSV.

    Parameters
    ----------
    path : str or None
        Path to scores CSV.  Defaults to ``SCORES_CSV``.

    Returns
    -------
    pd.DataFrame
        Columns include: ma_id, review_id, audit_score, consistency_score,
        robustness_score, stability_score, power_score, final_score,
        grade, grade_label.
        ``final_score`` is guaranteed numeric (float/int).
    """
    csv_path = path if path is not None else SCORES_CSV
    df = pd.read_csv(csv_path)

    # Ensure final_score is numeric; coerce silently (non-parseable → NaN)
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")

    return df


def load_verdicts(path: str | None = None) -> pd.DataFrame:
    """Load ActionableEvidence verdicts CSV.

    Parameters
    ----------
    path : str or None
        Path to verdicts CSV.  Defaults to ``VERDICTS_CSV``.

    Returns
    -------
    pd.DataFrame
        Columns include: ma_id, review_id, k, total_n, estimate, p_value,
        significant, pi_crosses_null, robustness, n_audit_fails,
        has_pub_bias, verdict, failed_criteria.

        String "True"/"False" values in ``significant``, ``pi_crosses_null``,
        and ``has_pub_bias`` are converted to Python ``bool``.
        ``p_value``, ``k``, ``total_n``, and ``estimate`` are numeric.
    """
    csv_path = path if path is not None else VERDICTS_CSV
    df = pd.read_csv(csv_path)

    # Convert "True"/"False" strings to actual bools for these three columns
    bool_cols = ["significant", "pi_crosses_null", "has_pub_bias"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip() == "True"
                else (False if str(x).strip() == "False"
                else x)
            ).astype(bool)

    # Ensure numeric types
    numeric_cols = ["p_value", "k", "total_n", "estimate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_dois(path: str | None = None) -> pd.DataFrame:
    """Load pairwise70 metadata CSV with DOIs.

    Parameters
    ----------
    path : str or None
        Path to DOIs CSV.  Defaults to ``DOIS_CSV``.

    Returns
    -------
    pd.DataFrame
        Original columns (review_id, n_studies, doi, years) plus a new
        ``review_id_prefix`` column created by stripping the ``_pub\\d+``
        suffix from ``review_id`` (e.g. ``CD000028_pub4`` → ``CD000028``).
        Deduplicated by prefix so each CD number appears once (first DOI kept).
    """
    csv_path = path if path is not None else DOIS_CSV
    df = pd.read_csv(csv_path)

    # Strip _pub\d+ suffix to get the bare CD number
    df["review_id_prefix"] = df["review_id"].str.replace(
        r"_pub\d+$", "", regex=True
    )

    # Deduplicate: keep first row per prefix (preserves earliest DOI record)
    df = df.drop_duplicates(subset="review_id_prefix", keep="first").reset_index(drop=True)

    return df


def merge_data(
    scores_df: pd.DataFrame,
    verdicts_df: pd.DataFrame,
    dois_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge scores, verdicts, and DOIs into a single analysis-ready DataFrame.

    Join logic
    ----------
    1. Inner join ``scores_df`` and ``verdicts_df`` on ``ma_id``.
    2. Left join the result with ``dois_df`` on:
           scores.review_id  ==  dois.review_id_prefix
       so each Cochrane review CD number links to its pairwise70 DOI.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of :func:`load_scores`.
    verdicts_df : pd.DataFrame
        Output of :func:`load_verdicts`.
    dois_df : pd.DataFrame
        Output of :func:`load_dois`.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from scores, verdicts (suffixed
        ``_y`` for duplicate names), and doi/n_studies/years from dois.
    """
    # Step 1: inner join on ma_id — select specific columns to avoid duplicates
    score_cols = ["ma_id", "review_id", "final_score", "grade", "grade_label",
                  "audit_score", "consistency_score", "robustness_score",
                  "stability_score", "power_score"]
    score_cols = [c for c in score_cols if c in scores_df.columns]
    verdict_cols = ["ma_id", "k", "total_n", "estimate", "p_value", "significant"]
    verdict_cols = [c for c in verdict_cols if c in verdicts_df.columns]

    combined = pd.merge(scores_df[score_cols], verdicts_df[verdict_cols],
                        on="ma_id", how="inner")

    # Step 2: left join DOIs on review_id = review_id_prefix
    if "review_id_prefix" not in dois_df.columns:
        dois_df = dois_df.copy()
        dois_df["review_id_prefix"] = dois_df["review_id"].str.replace(
            r"_pub\d+$", "", regex=True
        )

    merged = pd.merge(
        combined,
        dois_df[["review_id_prefix", "doi"]].rename(
            columns={"review_id_prefix": "review_id"}
        ),
        on="review_id",
        how="left",
    )

    return merged


# ---------------------------------------------------------------------------
# Trust-weighting helpers
# ---------------------------------------------------------------------------

def z_from_p(p: float) -> float:
    """Convert a two-sided p-value to an absolute z-score.

    Uses the Abramowitz & Stegun rational approximation (26.2.23) for the
    inverse normal CDF applied to the one-sided tail probability p/2.

    Parameters
    ----------
    p : float
        Two-sided p-value (0 < p < 1).

    Returns
    -------
    float
        Absolute z-score corresponding to ``p``.
    """
    # Guard against NaN — would silently produce extreme z via max(1e-15, NaN)
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 0.0

    # Clamp to avoid log(0); lower bound must keep t_val > 0 in float64
    p = max(1e-15, min(p, 1.0 - 1e-15))
    # One-sided tail probability
    t_val = p / 2.0
    # We need the upper-tail quantile, so work with 1 - t_val
    q = 1.0 - t_val
    # A&S 26.2.23 rational approximation for the inverse normal CDF
    # q is always > 0.5 after clamping (t_val = p/2 < 0.5, so q = 1 - t_val > 0.5)
    inner = 1.0 - q
    if inner <= 0.0:
        # Extreme p-value — return a very large z directly
        return 37.5
    t = math.sqrt(-2.0 * math.log(inner))
    # Rational approximation coefficients from A&S 26.2.23
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    z = t - numerator / denominator
    return abs(z)


def compute_se_from_p(estimate: float, p_value: float) -> float | None:
    """Derive standard error from effect size and two-sided p-value.

    Formula: SE = |estimate| / z  where z = z_from_p(p_value).

    Parameters
    ----------
    estimate : float
        Point estimate (e.g. log-OR, SMD).
    p_value : float
        Two-sided p-value.

    Returns
    -------
    float or None
        Derived SE, or ``None`` if ``estimate == 0`` or ``z == 0``.
    """
    # Guard NaN: pd.to_numeric(errors="coerce") produces NaN for unparseable values
    if estimate is None or p_value is None:
        return None
    if (isinstance(estimate, float) and math.isnan(estimate)) or \
       (isinstance(p_value, float) and math.isnan(p_value)):
        return None
    if estimate == 0:
        return None
    z = z_from_p(p_value)
    if z == 0:
        return None
    return abs(estimate) / z


def trust_weight_ma(
    estimate: float,
    p_value: float,
    trust_score: float,
) -> dict:
    """Compute trust-weighted statistics for one meta-analysis.

    Parameters
    ----------
    estimate : float
        Point estimate.
    p_value : float
        Two-sided p-value.
    trust_score : float
        EvidenceScore trust score (0–100).

    Returns
    -------
    dict
        Keys: original_weight, trust_weight, trust_weighted_estimate, se.
        All zero if SE cannot be derived (estimate=0 or z=0).
    """
    se = compute_se_from_p(estimate, p_value)
    if se is None:
        return {
            "original_weight": 0.0,
            "trust_weight": 0.0,
            "weighted_numerator": 0.0,
            "se": None,
        }

    original_weight = 1.0 / (se ** 2)
    score_fraction = trust_score / 100.0
    tw = original_weight * score_fraction

    return {
        "original_weight": original_weight,
        "trust_weight": tw,
        "weighted_numerator": tw * estimate,  # numerator for IV pooling: w_i * theta_i
        "se": se,
    }


def compute_erosion_curve(
    merged_df: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Compute trust-erosion statistics across score thresholds.

    For each threshold T in ``EROSION_THRESHOLDS``:

    1. Start with all originally-significant MAs.
    2. MAs with ``final_score < T`` are *excluded*.
    3. Among surviving MAs (score >= T), apply linear trust-weighting
       (trust_weight = original_weight * score/100) and recompute z.
       MAs where the trust-adjusted z < 1.96 are *weakened*.
    4. n_surviving  = n_remaining - n_weakened
    5. erosion_rate = (n_excluded + n_weakened) / n_total_sig * 100

    Parameters
    ----------
    merged_df : pd.DataFrame
        Must contain columns: final_score, significant, estimate, p_value.
    thresholds : list of int, optional
        Defaults to ``EROSION_THRESHOLDS``.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, n_total_sig, n_remaining, n_surviving,
        n_weakened, n_excluded, erosion_rate.
    """
    if thresholds is None:
        thresholds = EROSION_THRESHOLDS

    sig_df = merged_df[merged_df["significant"] == True].copy()
    # Drop rows with NaN in critical columns to prevent silent misclassification
    sig_df = sig_df.dropna(subset=["final_score", "estimate", "p_value"])
    n_total_sig = len(sig_df)

    # Vectorized trust-adjusted z computation (avoids slow iterrows)
    # z_trust = z_original * sqrt(score/100), where z_original = |estimate| / SE
    # and SE = |estimate| / z_from_p(p). So z_trust = z_from_p(p) * sqrt(score/100).
    sig_df["_z_orig"] = sig_df["p_value"].apply(z_from_p)
    sig_df["_z_trust"] = sig_df["_z_orig"] * np.sqrt(sig_df["final_score"] / 100.0)
    # MAs where z_from_p returned 0 (estimate=0 or p=NaN) are conservatively weakened
    sig_df["_can_assess"] = (sig_df["_z_orig"] > 0) & (sig_df["estimate"] != 0)

    rows = []
    for t in thresholds:
        remaining = sig_df[sig_df["final_score"] >= t]
        n_remaining = len(remaining)
        n_excluded = n_total_sig - n_remaining

        # Count weakened: assessable MAs whose trust-adjusted z < 1.96, plus unassessable
        assessable = remaining[remaining["_can_assess"]]
        unassessable = remaining[~remaining["_can_assess"]]
        n_weakened = int((assessable["_z_trust"] < 1.96).sum()) + len(unassessable)

        n_surviving = n_remaining - n_weakened

        if n_total_sig > 0:
            erosion_rate = (n_excluded + n_weakened) / n_total_sig * 100.0
        else:
            erosion_rate = 0.0

        rows.append({
            "threshold": t,
            "n_total_sig": n_total_sig,
            "n_remaining": n_remaining,
            "n_surviving": n_surviving,
            "n_weakened": n_weakened,
            "n_excluded": n_excluded,
            "erosion_rate": erosion_rate,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Domain grouping
# ---------------------------------------------------------------------------

def load_review_groups(path=None) -> pd.DataFrame:
    """Load review domain groupings from CSV.

    Parameters
    ----------
    path : str, Path, or None
        Path to review_groups.csv.  Defaults to ``DATA_DIR / "review_groups.csv"``.

    Returns
    -------
    pd.DataFrame
        Columns: review_id_prefix, review_group.
        Returns an empty DataFrame (with those columns) if the file is missing
        or contains no data rows.
    """
    csv_path = Path(path) if path is not None else DATA_DIR / "review_groups.csv"
    empty = pd.DataFrame(columns=["review_id_prefix", "review_group"])
    if not csv_path.exists():
        return empty
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return empty
    if df.empty or "review_id_prefix" not in df.columns or "review_group" not in df.columns:
        return empty
    return df


def compute_domain_erosion(
    merged_df: pd.DataFrame,
    thresholds: list | None = None,
) -> pd.DataFrame:
    """Compute per-domain trust-erosion statistics across score thresholds.

    For each ``review_group`` domain and each threshold T, count how many
    originally-significant MAs in that domain are excluded (``final_score < T``).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Must contain columns: final_score, significant, review_group (optional;
        missing values filled with "Unknown").
    thresholds : list of int, optional
        Defaults to ``EROSION_THRESHOLDS``.

    Returns
    -------
    pd.DataFrame
        Columns: domain, threshold, n_total_sig, n_excluded, erosion_rate, n_mas.
        One row per (domain, threshold) combination.
    """
    if thresholds is None:
        thresholds = EROSION_THRESHOLDS

    df = merged_df.copy()
    if "review_group" not in df.columns:
        df["review_group"] = "Unknown"
    else:
        df["review_group"] = df["review_group"].fillna("Unknown")

    rows = []
    for domain, group_df in df.groupby("review_group"):
        n_mas = len(group_df)
        sig_df = group_df[group_df["significant"] == True]
        n_total_sig = len(sig_df)
        for t in thresholds:
            n_excluded = int((sig_df["final_score"] < t).sum())
            if n_total_sig > 0:
                erosion_rate = n_excluded / n_total_sig * 100.0
            else:
                erosion_rate = 0.0
            rows.append({
                "domain": domain,
                "threshold": t,
                "n_total_sig": n_total_sig,
                "n_excluded": n_excluded,
                "erosion_rate": erosion_rate,
                "n_mas": n_mas,
            })

    return pd.DataFrame(rows, columns=["domain", "threshold", "n_total_sig",
                                        "n_excluded", "erosion_rate", "n_mas"])


# ---------------------------------------------------------------------------
# Citation fetching + influence score
# ---------------------------------------------------------------------------

def fetch_citation_counts(dois: list, cache_path=None) -> dict:
    """Fetch citation counts from PubMed eutils for a list of DOIs.

    For each DOI:
      1. esearch DOI → PMID
      2. elink PMID → cited-by count (citedin linkname)

    Results are cached to ``cache_path`` (JSON).  Cached entries are used on
    subsequent calls; only missing DOIs are fetched from the network.

    Parameters
    ----------
    dois : list of str
        DOIs to look up (e.g. ["10.1002/14651858.CD000028.pub4"]).
    cache_path : str, Path, or None
        Path to JSON cache file.  Defaults to ``DATA_DIR / "citation_cache.json"``.

    Returns
    -------
    dict
        Mapping {doi: citation_count (int)}.  Returns 0 for any DOI that
        could not be resolved or whose network request failed.
    """
    _cache_path = Path(cache_path) if cache_path is not None else DATA_DIR / "citation_cache.json"

    # Load existing cache
    cache: dict = {}
    if _cache_path.exists():
        try:
            with open(_cache_path, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    # Validate DOI format to prevent injection via malformed values
    _doi_re = re.compile(r'^10\.\d{4,9}/[^\s\x00-\x1f]+$')

    results: dict = {}
    needs_fetch = [d for d in dois if d not in cache and _doi_re.match(str(d))]
    # Cache invalid DOIs as 0 so they're not re-attempted
    for d in dois:
        if d not in cache and not _doi_re.match(str(d)):
            cache[d] = 0

    for doi in needs_fetch:
        try:
            # Step 1: DOI → PMID via esearch
            params = urllib.parse.urlencode({
                "db": "pubmed",
                "term": f"{doi}[DOI]",
                "retmode": "xml",
                "retmax": "1",
            })
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            id_list = root.findall(".//Id")
            if not id_list:
                cache[doi] = 0
                time.sleep(0.34)
                continue
            pmid = id_list[0].text.strip()

            time.sleep(0.34)

            # Step 2: PMID → cited-by count via elink
            params2 = urllib.parse.urlencode({
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "linkname": "pubmed_pubmed_citedin",
                "retmode": "xml",
            })
            url2 = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?{params2}"
            with urllib.request.urlopen(url2, timeout=10) as resp2:
                xml_bytes2 = resp2.read()
            root2 = ET.fromstring(xml_bytes2)
            cited_ids = root2.findall(".//LinkSetDb/Link/Id")
            cache[doi] = len(cited_ids)

            time.sleep(0.34)

        except (urllib.error.URLError, urllib.error.HTTPError,
                ET.ParseError, OSError, ValueError, TimeoutError):
            cache[doi] = 0

    # Persist updated cache
    try:
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception:
        pass

    for doi in dois:
        results[doi] = cache.get(doi, 0)

    return results


def compute_citation_percentile(citations: pd.Series) -> pd.Series:
    """Normalize citation counts to percentile rank 0-100.

    Parameters
    ----------
    citations : pd.Series
        Raw citation counts (numeric).

    Returns
    -------
    pd.Series
        Percentile ranks in [0, 100].  If the series is empty, returns an
        empty Series.  If all values are identical, returns 50.0 for all rows.
    """
    if citations.empty:
        return citations.astype(float)

    # All-same guard: rank(pct=True) returns 1.0 for all when all values equal,
    # giving 100.0 instead of 50.0.  Detect and override.
    if citations.nunique() == 1:
        return pd.Series(50.0, index=citations.index)

    return citations.rank(pct=True) * 100.0


# ---------------------------------------------------------------------------
# WHO Essential Medicines matching
# ---------------------------------------------------------------------------

def load_who_medicines(path=None) -> pd.DataFrame:
    """Load WHO Essential Medicines list from CSV.

    Parameters
    ----------
    path : str, Path, or None
        Path to who_medicines.csv.  Defaults to ``DATA_DIR / "who_medicines.csv"``.

    Returns
    -------
    pd.DataFrame
        Columns: medicine_name, category.
        Returns an empty DataFrame (with those columns) if the file is missing
        or cannot be read.
    """
    csv_path = Path(path) if path is not None else DATA_DIR / "who_medicines.csv"
    empty = pd.DataFrame(columns=["medicine_name", "category"])
    if not csv_path.exists():
        return empty
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return empty
    if df.empty or "medicine_name" not in df.columns or "category" not in df.columns:
        return empty
    return df


def match_who_medicines(interventions: dict, who_df: pd.DataFrame) -> dict:
    """Match intervention texts against WHO Essential Medicines names.

    For each review in ``interventions``, check whether any WHO medicine name
    appears as a case-insensitive substring in the intervention text.

    Parameters
    ----------
    interventions : dict
        Mapping {review_id: intervention_text (str)}.
    who_df : pd.DataFrame
        DataFrame with at least a ``medicine_name`` column (from
        :func:`load_who_medicines`).

    Returns
    -------
    dict
        Mapping {review_id: bool} — ``True`` if any WHO medicine name is found
        in the intervention text, ``False`` otherwise.
    """
    if who_df.empty or "medicine_name" not in who_df.columns:
        return {rid: False for rid in interventions}

    medicine_names = [str(m).lower() for m in who_df["medicine_name"] if pd.notna(m)]

    results: dict = {}
    for review_id, text in interventions.items():
        text_lower = str(text).lower()
        matched = any(med in text_lower for med in medicine_names)
        results[review_id] = matched

    return results


_REVIEW_SUFFIX_RE = re.compile(r"_pub\d+$", re.IGNORECASE)


def normalize_review_id(value: str | None) -> str:
    """Normalize review identifiers to a bare uppercase Cochrane prefix."""
    if value is None:
        return ""
    review_id = str(value).strip()
    if not review_id:
        return ""
    return _REVIEW_SUFFIX_RE.sub("", review_id).upper()


def _safe_nonnegative_int(value) -> int:
    """Convert cache values to non-negative integers, defaulting to 0."""
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def load_nice_guideline_cache(path: str | Path | None = None) -> dict:
    """Load and normalize NICE guideline counts from the local cache.

    Supported cache shapes
    ----------------------
    1. Legacy flat dict keyed by DOI:
         ``{"10.1002/...": 2}``
    2. Legacy flat dict keyed by review prefix:
         ``{"CD000028": 1}``
    3. Structured cache with both namespaces:
         ``{"by_doi": {...}, "by_review_id_prefix": {...}}``

    Parameters
    ----------
    path : str, Path, or None
        Path to ``nice_cache.json``. Defaults to ``DATA_DIR / "nice_cache.json"``.

    Returns
    -------
    dict
        Normalized mapping with two keys:
        ``{"by_doi": {...}, "by_review_id_prefix": {...}}``.
    """
    cache_path = Path(path) if path is not None else DATA_DIR / "nice_cache.json"
    empty = {"by_doi": {}, "by_review_id_prefix": {}}
    if not cache_path.exists():
        return empty

    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            raw_cache = json.load(fh)
    except Exception:
        return empty

    if not isinstance(raw_cache, dict):
        return empty

    raw_by_doi = {}
    raw_by_review = {}
    if "by_doi" in raw_cache or "by_review_id_prefix" in raw_cache:
        if isinstance(raw_cache.get("by_doi"), dict):
            raw_by_doi = raw_cache.get("by_doi", {})
        if isinstance(raw_cache.get("by_review_id_prefix"), dict):
            raw_by_review = raw_cache.get("by_review_id_prefix", {})
    else:
        for key, value in raw_cache.items():
            key_str = str(key).strip()
            if not key_str:
                continue
            if key_str.lower().startswith("10."):
                raw_by_doi[key_str] = value
            else:
                raw_by_review[key_str] = value

    by_doi = {}
    for key, value in raw_by_doi.items():
        doi = str(key).strip()
        if doi:
            by_doi[doi] = _safe_nonnegative_int(value)

    by_review_id_prefix = {}
    for key, value in raw_by_review.items():
        review_id = normalize_review_id(key)
        if review_id:
            by_review_id_prefix[review_id] = _safe_nonnegative_int(value)

    return {
        "by_doi": by_doi,
        "by_review_id_prefix": by_review_id_prefix,
    }


def fetch_nice_guideline_counts(
    dois: list,
    review_ids: list | None = None,
    cache_path: str | Path | None = None,
) -> dict:
    """Resolve NICE guideline counts from the local cache.

    Parameters
    ----------
    dois : list of str
        DOIs to look up.
    review_ids : list of str, optional
        Review IDs (for example ``CD000028``) to look up in the same cache.
    cache_path : str, Path, or None
        Optional override for the NICE cache file location.

    Returns
    -------
    dict
        Mapping keyed by the requested DOI values and normalized review IDs.
        Missing entries default to 0.
    """
    cache = load_nice_guideline_cache(cache_path)

    results = {}
    for doi in dois:
        doi_key = str(doi).strip()
        if doi_key:
            results[doi_key] = cache["by_doi"].get(doi_key, 0)

    for review_id in review_ids or []:
        normalized = normalize_review_id(review_id)
        if normalized:
            results[normalized] = cache["by_review_id_prefix"].get(normalized, 0)

    return results


def compute_influence_score(
    citation_percentile: float,
    who_essential: bool,
    nice_guideline_count: int,
    group_size_percentile: float,
) -> float:
    """Compute a composite influence score in [0, 100].

    Formula
    -------
    ::

        influence = citation_percentile * 0.4
                  + (20 if who_essential else 0)
                  + min(nice_guideline_count * 10, 30)
                  + group_size_percentile * 0.1

    Clamped to [0, 100].

    Parameters
    ----------
    citation_percentile : float
        Percentile rank of citations (0–100).
    who_essential : bool
        Whether the drug/intervention appears on the WHO Essential Medicines List.
    nice_guideline_count : int
        Number of NICE guidelines referencing this review.
    group_size_percentile : float
        Percentile rank of the review-group size (0–100).

    Returns
    -------
    float
        Composite influence score clamped to [0, 100].
    """
    score = (
        citation_percentile * INFLUENCE_WEIGHTS["citation_percentile"]
        + (INFLUENCE_WEIGHTS["who_essential"] if who_essential else 0.0)
        + min(nice_guideline_count * INFLUENCE_WEIGHTS["nice_per_guideline"],
              INFLUENCE_WEIGHTS["nice_guideline_cap"])
        + group_size_percentile * INFLUENCE_WEIGHTS["group_size_percentile"]
    )
    return float(max(0.0, min(100.0, score)))


# ---------------------------------------------------------------------------
# Risk register
# ---------------------------------------------------------------------------

def assign_quadrant(trust: float, influence: float) -> str:
    """Assign a risk quadrant based on trust and influence scores.

    Quadrant rules (using module-level threshold constants):

    - **Red Flag**   : trust < TRUST_LOW_THRESHOLD  AND influence > INFLUENCE_HIGH_THRESHOLD
    - **Hidden Gem** : trust >= TRUST_HIGH_THRESHOLD AND influence < INFLUENCE_LOW_THRESHOLD
    - **Safe**       : trust >= TRUST_HIGH_THRESHOLD AND influence >= INFLUENCE_HIGH_THRESHOLD
    - **Low Stakes** : trust < TRUST_LOW_THRESHOLD  AND influence < INFLUENCE_LOW_THRESHOLD
    - **Moderate**   : everything else (middle zone)

    Parameters
    ----------
    trust : float
        Trust score (0-100), typically ``final_score`` from EvidenceScore.
    influence : float
        Influence score (0-100), from :func:`compute_influence_score`.

    Returns
    -------
    str
        One of: "Red Flag", "Hidden Gem", "Safe", "Low Stakes", "Moderate".
    """
    if trust < TRUST_LOW_THRESHOLD and influence > INFLUENCE_HIGH_THRESHOLD:
        return "Red Flag"
    if trust >= TRUST_HIGH_THRESHOLD and influence < INFLUENCE_LOW_THRESHOLD:
        return "Hidden Gem"
    if trust >= TRUST_HIGH_THRESHOLD and influence > INFLUENCE_HIGH_THRESHOLD:
        return "Safe"
    if trust < TRUST_LOW_THRESHOLD and influence < INFLUENCE_LOW_THRESHOLD:
        return "Low Stakes"
    return "Moderate"


def build_risk_register(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`assign_quadrant` to all rows and return a risk register.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``final_score`` (trust) and ``influence_score``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an additional ``quadrant`` column (str).
    """
    result = df.copy()
    result["quadrant"] = np.vectorize(assign_quadrant)(
        result["final_score"].values, result["influence_score"].values
    )
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    scores_path=None,
    verdicts_path=None,
    dois_path=None,
    fetch_citations=True,
) -> dict:
    """Execute the full TrustGate pipeline end-to-end.

    Steps
    -----
    1. Load data: scores, verdicts, DOIs.
    2. Merge into a single analysis-ready DataFrame.
    3. Part A — Erosion: overall erosion curve + domain erosion.
    4. Part B — Influence: citation counts, WHO/NICE lookups, influence score.
    5. Build risk register with quadrant labels.
    6. Assemble summary dict.

    Parameters
    ----------
    scores_path : str or None
        Path to scores CSV.  Defaults to ``SCORES_CSV``.
    verdicts_path : str or None
        Path to verdicts CSV.  Defaults to ``VERDICTS_CSV``.
    dois_path : str or None
        Path to DOIs CSV.  Defaults to ``DOIS_CSV``.
    fetch_citations : bool
        If True and DOIs are available, fetch citation counts from PubMed.
        If False (or no DOIs), load from ``CITATION_CACHE`` if it exists, else
        use empty dict (all citation counts = 0).

    Returns
    -------
    dict
        Keys:
        - ``merged``        : pd.DataFrame — all MAs with all computed columns
        - ``erosion_curve`` : pd.DataFrame — overall erosion curve
        - ``domain_erosion``: pd.DataFrame — per-domain erosion
        - ``risk_register`` : pd.DataFrame — merged + quadrant column
        - ``summary``       : dict — headline statistics
    """
    # ------------------------------------------------------------------ #
    # Step 1: Load
    # ------------------------------------------------------------------ #
    scores_df = load_scores(path=scores_path)
    verdicts_df = load_verdicts(path=verdicts_path)
    dois_df = load_dois(path=dois_path)

    # ------------------------------------------------------------------ #
    # Step 2: Merge
    # ------------------------------------------------------------------ #
    merged = merge_data(scores_df, verdicts_df, dois_df)

    # ------------------------------------------------------------------ #
    # Step 3: Part A — Erosion
    # ------------------------------------------------------------------ #
    erosion_curve = compute_erosion_curve(merged)

    # Domain erosion: join review_groups if available
    review_groups = load_review_groups()
    if not review_groups.empty:
        merged = pd.merge(
            merged,
            review_groups.rename(columns={"review_id_prefix": "review_id"}),
            on="review_id",
            how="left",
        )
    domain_erosion = compute_domain_erosion(merged)

    # ------------------------------------------------------------------ #
    # Step 4: Part B — Influence
    # ------------------------------------------------------------------ #
    unique_dois = [d for d in merged["doi"].dropna().unique().tolist() if d]

    # Citation counts
    if fetch_citations and unique_dois:
        citation_counts = fetch_citation_counts(unique_dois)
    else:
        # Try loading from cache, otherwise default to zeros
        if CITATION_CACHE.exists():
            try:
                with open(CITATION_CACHE, "r", encoding="utf-8") as fh:
                    citation_counts = json.load(fh)
            except Exception:
                citation_counts = {}
        else:
            citation_counts = {}

    # Map citation counts to merged rows (default 0 for missing)
    merged["citation_count"] = merged["doi"].map(citation_counts).fillna(0).astype(int)

    # Compute citation percentile per review (one citation_count per review_id)
    review_citations = (
        merged.groupby("review_id")["citation_count"]
        .first()
    )
    review_citation_pct = compute_citation_percentile(review_citations)
    merged["citation_percentile"] = merged["review_id"].map(review_citation_pct).fillna(50.0)

    # WHO Essential Medicines — load pre-computed matches from build_enrichment.py
    who_matches_path = DATA_DIR / "who_matches.json"
    if who_matches_path.exists():
        with open(who_matches_path, "r", encoding="utf-8") as f:
            who_matches = json.load(f)
        merged["who_essential"] = merged["review_id"].map(
            lambda rid: who_matches.get(rid, False)
        )
    else:
        merged["who_essential"] = False

    # NICE guideline counts come from the local cache, supporting both DOI
    # keys and review-prefix keys generated by the enrichment builder.
    nice_cache = load_nice_guideline_cache()

    def _resolve_nice_count(row) -> int:
        doi = row.get("doi")
        if pd.notna(doi):
            doi_key = str(doi).strip()
            if doi_key in nice_cache["by_doi"]:
                return int(nice_cache["by_doi"][doi_key])

        review_key = normalize_review_id(row.get("review_id"))
        return int(nice_cache["by_review_id_prefix"].get(review_key, 0))

    merged["nice_guideline_count"] = merged.apply(_resolve_nice_count, axis=1).astype(int)

    # Group size percentile
    group_sizes = merged.groupby("review_id").size()
    group_size_pct = compute_citation_percentile(group_sizes)
    merged["group_size_percentile"] = merged["review_id"].map(group_size_pct).fillna(50.0)

    # Compute influence score row-by-row
    merged["influence_score"] = merged.apply(
        lambda row: compute_influence_score(
            citation_percentile=row["citation_percentile"],
            who_essential=bool(row["who_essential"]),
            nice_guideline_count=int(row["nice_guideline_count"]),
            group_size_percentile=row["group_size_percentile"],
        ),
        axis=1,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Risk register
    # ------------------------------------------------------------------ #
    risk_register = build_risk_register(merged)

    # ------------------------------------------------------------------ #
    # Step 6: Summary
    # ------------------------------------------------------------------ #
    total_mas = len(merged)
    total_reviews = merged["review_id"].nunique()
    total_significant = int(merged["significant"].sum()) if "significant" in merged.columns else 0

    # erosion_rate at threshold=70
    ec70 = erosion_curve[erosion_curve["threshold"] == 70]
    erosion_rate_at_70 = float(ec70["erosion_rate"].iloc[0]) if len(ec70) > 0 else 0.0

    # Quadrant counts
    quadrant_counts: dict = {}
    if "quadrant" in risk_register.columns:
        quadrant_counts = risk_register["quadrant"].value_counts().to_dict()
    red_flag_count = int(quadrant_counts.get("Red Flag", 0))
    hidden_gem_count = int(quadrant_counts.get("Hidden Gem", 0))

    mean_trust_score = float(merged["final_score"].mean()) if total_mas > 0 else 0.0
    mean_influence_score = float(merged["influence_score"].mean()) if total_mas > 0 else 0.0

    summary = {
        "total_mas": total_mas,
        "total_reviews": total_reviews,
        "total_significant": total_significant,
        "erosion_rate_at_70": erosion_rate_at_70,
        "red_flag_count": red_flag_count,
        "hidden_gem_count": hidden_gem_count,
        "mean_trust_score": mean_trust_score,
        "mean_influence_score": mean_influence_score,
        "quadrant_counts": quadrant_counts,
        "erosion_thresholds": EROSION_THRESHOLDS,
    }

    return {
        "merged": merged,
        "erosion_curve": erosion_curve,
        "domain_erosion": domain_erosion,
        "risk_register": risk_register,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Results saver
# ---------------------------------------------------------------------------

def save_results(pipeline_results: dict, output_dir=None) -> dict:
    """Save all pipeline outputs to CSV and JSON files.

    Files written
    -------------
    - ``risk_register.csv``   — full risk register with quadrant column
    - ``erosion_curve.csv``   — overall erosion curve
    - ``domain_erosion.csv``  — per-domain erosion statistics
    - ``trust_weighted.csv``  — key columns subset for trust-weighted analysis
    - ``guideline_exposure.csv`` — deduplicated per-review exposure table
    - ``summary.json``        — headline summary statistics

    Parameters
    ----------
    pipeline_results : dict
        Output of :func:`run_pipeline`.
    output_dir : str, Path, or None
        Directory to write files into.  Defaults to ``RESULTS_DIR``.

    Returns
    -------
    dict
        The ``summary`` sub-dict from ``pipeline_results`` (for convenience).
    """
    out_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_csv(df):
        """Prevent CSV formula injection: prepend ' to cells starting with =+@\\t\\r."""
        result = df.copy()
        for col in result.select_dtypes(include="object").columns:
            result[col] = result[col].apply(
                lambda v: "'" + v if isinstance(v, str) and v and v[0] in ('=', '+', '@', '\t', '\r') else v
            )
        return result

    merged = pipeline_results["merged"]
    erosion_curve = pipeline_results["erosion_curve"]
    domain_erosion = pipeline_results["domain_erosion"]
    risk_register = pipeline_results["risk_register"]
    summary = pipeline_results["summary"]

    # Full risk register (sanitized for CSV formula injection)
    _sanitize_csv(risk_register).to_csv(out_dir / "risk_register.csv", index=False)

    # Erosion curve
    erosion_curve.to_csv(out_dir / "erosion_curve.csv", index=False)

    # Domain erosion
    domain_erosion.to_csv(out_dir / "domain_erosion.csv", index=False)

    # trust_weighted.csv — subset of key columns
    _trust_cols = [
        "ma_id", "review_id", "final_score", "grade", "estimate", "p_value",
        "significant", "k", "total_n", "doi", "citation_count",
        "citation_percentile", "who_essential", "nice_guideline_count",
        "influence_score", "quadrant",
    ]
    # Only include columns that exist in risk_register
    tw_cols = [c for c in _trust_cols if c in risk_register.columns]
    risk_register[tw_cols].to_csv(out_dir / "trust_weighted.csv", index=False)

    # guideline_exposure.csv — one row per review_id
    _exposure_cols = [
        "review_id", "doi", "citation_count", "citation_percentile",
        "who_essential", "nice_guideline_count", "influence_score",
    ]
    exp_cols = [c for c in _exposure_cols if c in risk_register.columns]
    guideline_exposure = (
        risk_register[exp_cols]
        .drop_duplicates(subset="review_id")
        .reset_index(drop=True)
    )
    guideline_exposure.to_csv(out_dir / "guideline_exposure.csv", index=False)

    # Summary JSON
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fetch = "--no-fetch" not in sys.argv
    results = run_pipeline(fetch_citations=fetch)
    summary = save_results(results)

    print("TrustGate Pipeline Complete")
    print(f"  Total MAs        : {summary['total_mas']}")
    print(f"  Total Reviews    : {summary['total_reviews']}")
    print(f"  Total Significant: {summary['total_significant']}")
    print(f"  Erosion @ T=70   : {summary['erosion_rate_at_70']:.1f}%")
    print(f"  Red Flags        : {summary['red_flag_count']}")
    print(f"  Hidden Gems      : {summary['hidden_gem_count']}")
    print(f"  Mean Trust Score : {summary['mean_trust_score']:.1f}")
    print(f"  Mean Influence   : {summary['mean_influence_score']:.1f}")
    print(f"  Quadrants        : {summary['quadrant_counts']}")
