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

SCORES_CSV = r"C:\Models\EvidenceScore\results\scores.csv"
VERDICTS_CSV = r"C:\Models\ActionableEvidence\results\verdicts.csv"
DOIS_CSV = r"C:\Users\user\asreview_pairwise70_metadata.csv"

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
INFLUENCE_HIGH_THRESHOLD = 70


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

def _z_from_p(p: float) -> float:
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
    import math

    # Clamp to avoid log(0)
    p = max(1e-300, min(p, 1.0 - 1e-15))
    # One-sided tail probability
    t_val = p / 2.0
    # We need the upper-tail quantile, so work with 1 - t_val
    q = 1.0 - t_val
    # A&S 26.2.23 rational approximation for the inverse normal CDF
    # valid for q in (0.5, 1.0), i.e. for small t_val
    if q > 0.5:
        sign = 1.0
        arg = q
    else:
        sign = -1.0
        arg = 1.0 - q

    t = math.sqrt(-2.0 * math.log(1.0 - arg))
    # Rational approximation coefficients from A&S 26.2.23
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    z = sign * (t - numerator / denominator)

    return abs(z)


def compute_se_from_p(estimate: float, p_value: float) -> float | None:
    """Derive standard error from effect size and two-sided p-value.

    Formula: SE = |estimate| / z  where z = _z_from_p(p_value).

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
    if estimate == 0:
        return None
    z = _z_from_p(p_value)
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
            "trust_weighted_estimate": 0.0,
            "se": None,
        }

    original_weight = 1.0 / (se ** 2)
    score_fraction = trust_score / 100.0
    tw = original_weight * score_fraction

    return {
        "original_weight": original_weight,
        "trust_weight": tw,
        "trust_weighted_estimate": estimate * score_fraction,
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
    n_total_sig = len(sig_df)

    rows = []
    for t in thresholds:
        remaining = sig_df[sig_df["final_score"] >= t]
        excluded = sig_df[sig_df["final_score"] < t]
        n_remaining = len(remaining)
        n_excluded = len(excluded)

        n_weakened = 0
        for _, row in remaining.iterrows():
            se = compute_se_from_p(row["estimate"], row["p_value"])
            if se is None:
                # Cannot assess — count as weakened (conservative)
                n_weakened += 1
                continue
            original_weight = 1.0 / (se ** 2)
            score_fraction = row["final_score"] / 100.0
            trust_weight = original_weight * score_fraction
            # Trust-adjusted SE: se_adj = 1/sqrt(trust_weight)
            se_adj = 1.0 / (trust_weight ** 0.5)
            trust_z = abs(row["estimate"]) / se_adj
            if trust_z < 1.96:
                n_weakened += 1

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

    results: dict = {}
    needs_fetch = [d for d in dois if d not in cache]

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

        except Exception:
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
