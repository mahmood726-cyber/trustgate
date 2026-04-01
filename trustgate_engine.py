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

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

SCORES_CSV = r"C:\Models\EvidenceScore\results\scores.csv"
VERDICTS_CSV = r"C:\Models\ActionableEvidence\results\verdicts.csv"
DOIS_CSV = r"C:\Users\user\asreview_pairwise70_metadata.csv"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Trust score thresholds (final_score 0-100)
EROSION_THRESHOLDS = {
    "high":   80,   # final_score >= 80 → high trust
    "medium": 60,   # 60 <= final_score < 80 → medium trust
    "low":    40,   # 40 <= final_score < 60 → low trust
    # below 40 → very low trust
}

# Weights for each sub-score when computing a custom trust index
INFLUENCE_WEIGHTS = {
    "audit_score":        0.25,
    "consistency_score":  0.20,
    "robustness_score":   0.25,
    "stability_score":    0.15,
    "power_score":        0.15,
}

# Trust-adjusted significance cutoffs
TRUST_SIGNIFICANCE_THRESHOLD = 0.05   # nominal alpha
INFLUENCE_SCORE_THRESHOLD = 0.70      # minimum weighted influence to flag


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
    # Step 1: inner join on ma_id
    combined = pd.merge(scores_df, verdicts_df, on="ma_id", how="inner")

    # Step 2: left join DOIs — use review_id from scores side (_x after merge)
    # After inner join, scores' review_id is review_id_x; resolve appropriately
    review_id_col = "review_id_x" if "review_id_x" in combined.columns else "review_id"

    merged = pd.merge(
        combined,
        dois_df[["review_id_prefix", "doi", "n_studies", "years"]],
        left_on=review_id_col,
        right_on="review_id_prefix",
        how="left",
    )

    return merged
