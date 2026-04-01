"""
Tests T1-T3 for TrustGate data loaders.

T1: load_scores  — CSV load, 4 rows, required columns, numeric final_score
T2: load_verdicts — CSV load, bool coercion, numeric p_value
T3: merge_data   — inner join scores+verdicts, left join DOIs via prefix strip
"""
import io
import re
import pytest
import pandas as pd
from pathlib import Path
import sys

# Ensure the engine module is importable from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trustgate_engine import load_scores, load_verdicts, load_dois, merge_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCORES_CSV = """\
ma_id,review_id,audit_score,consistency_score,robustness_score,stability_score,power_score,final_score,grade,grade_label
CD001155_pub3_data__A5,CD001155,92.0,100.0,100.0,99.9,100.0,98,A+,Highly Trustworthy
CD000028_pub4_data__A1,CD000028,80.0,85.0,90.0,88.0,75.0,84,B+,Trustworthy
CD000028_pub4_data__A2,CD000028,78.0,82.0,88.0,86.0,73.0,81,B,Trustworthy
CD001909_pub4_data__A1,CD001909,92.0,100.0,100.0,99.6,100.0,98,A+,Highly Trustworthy
"""

VERDICTS_CSV = """\
ma_id,review_id,k,total_n,estimate,p_value,significant,pi_crosses_null,robustness,n_audit_fails,has_pub_bias,verdict,failed_criteria
CD000028_pub4_data__A1,CD000028,36,61632,-0.122319,0.00466,True,True,Fragile,2,False,NOT YET,prediction_interval;robustness
CD000028_pub4_data__A2,CD000028,42,85253,-0.383057,0.0,True,False,Fragile,0,False,NOT YET,robustness
CD001155_pub3_data__A5,CD001155,12,5400,0.35,0.032,True,False,Robust,0,False,ACTIONABLE,
CD001909_pub4_data__A1,CD001909,8,3200,0.12,0.210,False,True,Fragile,1,True,NOT YET,pub_bias
"""

DOIS_CSV = """\
review_id,n_studies,doi,years
CD000028_pub4,16,10.1002/14651858.CD000028.pub4,1970
CD000028_pub4,16,10.1002/14651858.CD000028.pub4,2008
CD001155_pub3,12,10.1002/14651858.CD001155.pub3,2005
"""


@pytest.fixture
def sample_scores():
    return pd.read_csv(io.StringIO(SCORES_CSV))


@pytest.fixture
def sample_verdicts():
    return pd.read_csv(io.StringIO(VERDICTS_CSV))


@pytest.fixture
def sample_dois():
    return pd.read_csv(io.StringIO(DOIS_CSV))


# ---------------------------------------------------------------------------
# T1: load_scores
# ---------------------------------------------------------------------------

def test_load_scores(tmp_path, sample_scores):
    """T1: load_scores returns 4 rows, required columns, numeric final_score."""
    csv_path = tmp_path / "scores.csv"
    sample_scores.to_csv(csv_path, index=False)

    df = load_scores(path=str(csv_path))

    # Row count
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    # Required columns present
    required = {"ma_id", "review_id", "final_score"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    # final_score is numeric
    assert pd.api.types.is_numeric_dtype(df["final_score"]), (
        f"final_score dtype is {df['final_score'].dtype}, expected numeric"
    )

    # No NaN in key columns
    assert df["ma_id"].notna().all(), "ma_id has NaN values"
    assert df["final_score"].notna().all(), "final_score has NaN values"


# ---------------------------------------------------------------------------
# T2: load_verdicts
# ---------------------------------------------------------------------------

def test_load_verdicts(tmp_path, sample_verdicts):
    """T2: load_verdicts parses True/False strings to bools, p_value numeric."""
    csv_path = tmp_path / "verdicts.csv"
    sample_verdicts.to_csv(csv_path, index=False)

    df = load_verdicts(path=str(csv_path))

    # Row count
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    # Bool columns are actual Python/numpy bool
    for col in ("significant", "pi_crosses_null", "has_pub_bias"):
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].dtype == bool, (
            f"Column '{col}' dtype is {df[col].dtype}, expected bool"
        )

    # Check specific values are correct bools (not strings).
    # Use == not `is`: numpy bool_ values compare equal to Python True/False
    # but are not the same singleton object.
    assert df.loc[df["ma_id"] == "CD000028_pub4_data__A1", "significant"].iloc[0] == True
    assert df.loc[df["ma_id"] == "CD001909_pub4_data__A1", "significant"].iloc[0] == False

    # p_value is numeric
    assert pd.api.types.is_numeric_dtype(df["p_value"]), (
        f"p_value dtype is {df['p_value'].dtype}, expected numeric"
    )

    # Numeric columns: k, total_n, estimate
    for col in ("k", "total_n", "estimate"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


# ---------------------------------------------------------------------------
# T3: merge_data
# ---------------------------------------------------------------------------

def test_merge_data(tmp_path, sample_scores, sample_verdicts, sample_dois):
    """T3: merge_data inner-joins scores+verdicts, left-joins DOIs via prefix strip."""
    # Write CSVs so we can pass through load functions for realistic test
    scores_path = tmp_path / "scores.csv"
    verdicts_path = tmp_path / "verdicts.csv"
    dois_path = tmp_path / "dois.csv"

    sample_scores.to_csv(scores_path, index=False)
    sample_verdicts.to_csv(verdicts_path, index=False)
    sample_dois.to_csv(dois_path, index=False)

    scores_df = load_scores(path=str(scores_path))
    verdicts_df = load_verdicts(path=str(verdicts_path))
    dois_df = load_dois(path=str(dois_path))

    merged = merge_data(scores_df, verdicts_df, dois_df)

    # Inner join on ma_id: all 4 MAs are present in both scores and verdicts
    assert len(merged) == 4, f"Expected 4 rows after inner join, got {len(merged)}"

    # Columns from both sides are present
    assert "final_score" in merged.columns, "final_score missing from merged"
    assert "p_value" in merged.columns, "p_value missing from merged"
    assert "significant" in merged.columns, "significant missing from merged"

    # DOI join: CD000028 rows should have a doi (prefix CD000028 matches)
    cd28_rows = merged[merged["review_id_x"].str.startswith("CD000028") |
                       merged["review_id_y"].str.startswith("CD000028")]
    assert len(cd28_rows) > 0, "No CD000028 rows in merged output"
    # DOI column should exist
    assert "doi" in merged.columns, "doi column missing from merged"
    # CD000028 has a matching DOI entry
    cd28_doi = merged[merged["ma_id"].str.startswith("CD000028")]["doi"]
    assert cd28_doi.notna().all(), "CD000028 rows should have non-null doi"

    # CD001909 has no entry in sample_dois — should have NaN doi (left join)
    cd1909_doi = merged[merged["ma_id"].str.startswith("CD001909")]["doi"]
    assert cd1909_doi.isna().all(), "CD001909 should have NaN doi (no DOI in fixture)"

    # DOI prefix stripping: verify dois_df has review_id_prefix column
    assert "review_id_prefix" in dois_df.columns, "review_id_prefix missing from dois_df"
    # Strip pattern: CD000028_pub4 -> CD000028
    sample_prefix = dois_df[dois_df["review_id"].str.startswith("CD000028")]["review_id_prefix"].iloc[0]
    assert sample_prefix == "CD000028", f"Expected 'CD000028', got '{sample_prefix}'"
