"""
Tests T1-T9 for TrustGate data loaders, trust-weighting, and erosion curve.

T1: load_scores  — CSV load, 4 rows, required columns, numeric final_score
T2: load_verdicts — CSV load, bool coercion, numeric p_value
T3: merge_data   — inner join scores+verdicts, left join DOIs via prefix strip
T4: compute_se_from_p — SE derivation from estimate + p-value
T5: trust_weight_ma — trust_weight = original_weight * score/100
T6: trust_weight_zero_score — trust_weight=0 when score=0
T7: erosion_curve_low_threshold — threshold 50 excludes 1 low-score MA
T8: erosion_curve_high_threshold — threshold 90 excludes 2 MAs
T9: erosion_rate_zero — all MAs score 100 → erosion_rate=0 at every threshold
"""
import io
import re
import pytest
import pandas as pd
from pathlib import Path
import sys

# Ensure the engine module is importable from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trustgate_engine import (
    load_scores, load_verdicts, load_dois, merge_data,
    z_from_p, compute_se_from_p, trust_weight_ma, compute_erosion_curve,
    load_review_groups, compute_domain_erosion,
    compute_citation_percentile, compute_influence_score,
    load_who_medicines, match_who_medicines, fetch_nice_guideline_counts,
    assign_quadrant, build_risk_register,
    run_pipeline, save_results,
)
from trustgate_engine import SCORES_CSV as ENGINE_SCORES_CSV


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
    assert "review_id" in merged.columns, "review_id column missing from merged"
    cd28_rows = merged[merged["review_id"] == "CD000028"]
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


# ---------------------------------------------------------------------------
# Fixtures for T7-T9  (scores 79, 96, 42 to match task description)
# ---------------------------------------------------------------------------

EROSION_SCORES_CSV = """\
ma_id,review_id,audit_score,consistency_score,robustness_score,stability_score,power_score,final_score,grade,grade_label
MA001,R001,80.0,78.0,79.0,77.0,81.0,79,B+,Trustworthy
MA002,R002,96.0,97.0,95.0,96.0,96.0,96,A+,Highly Trustworthy
MA003,R003,40.0,43.0,42.0,41.0,44.0,42,C,Low Trust
MA004,R004,60.0,63.0,62.0,61.0,64.0,62,B-,Moderate Trust
"""

EROSION_VERDICTS_CSV = """\
ma_id,review_id,k,total_n,estimate,p_value,significant,pi_crosses_null,robustness,n_audit_fails,has_pub_bias,verdict,failed_criteria
MA001,R001,10,2000,0.45,0.001,True,False,Robust,0,False,ACTIONABLE,
MA002,R002,15,5000,0.30,0.020,True,False,Robust,0,False,ACTIONABLE,
MA003,R003,8,1500,0.55,0.040,True,True,Fragile,1,False,NOT YET,prediction_interval
MA004,R004,6,800,0.10,0.210,False,True,Fragile,1,True,NOT YET,pub_bias
"""


@pytest.fixture
def erosion_scores():
    return pd.read_csv(io.StringIO(EROSION_SCORES_CSV))


@pytest.fixture
def erosion_verdicts():
    df = pd.read_csv(io.StringIO(EROSION_VERDICTS_CSV))
    bool_cols = ["significant", "pi_crosses_null", "has_pub_bias"]
    for col in bool_cols:
        df[col] = df[col].map(
            lambda x: True if str(x).strip() == "True"
            else (False if str(x).strip() == "False" else x)
        ).astype(bool)
    for col in ["p_value", "k", "total_n", "estimate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# T4: compute_se_from_p
# ---------------------------------------------------------------------------

def test_se_from_p_value():
    """T4: SE derived from estimate + p-value via z-score."""
    # estimate=0.5, p=0.001 → z≈3.29 → SE≈0.152
    se = compute_se_from_p(0.5, 0.001)
    assert se is not None, "SE should not be None for non-zero estimate"
    assert 0.14 < se < 0.16, f"Expected SE in (0.14, 0.16), got {se}"

    # estimate=0 → None
    se_zero = compute_se_from_p(0.0, 0.001)
    assert se_zero is None, "SE should be None when estimate=0"


# ---------------------------------------------------------------------------
# T4b: z_from_p accuracy against known reference values
# ---------------------------------------------------------------------------

def test_z_from_p_accuracy():
    """T4b: z_from_p matches known z-values within 0.1%."""
    assert abs(z_from_p(0.05) - 1.96) < 0.01, f"z(0.05) should be ~1.96, got {z_from_p(0.05)}"
    assert abs(z_from_p(0.01) - 2.576) < 0.01, f"z(0.01) should be ~2.576, got {z_from_p(0.01)}"
    assert abs(z_from_p(0.001) - 3.291) < 0.01, f"z(0.001) should be ~3.291, got {z_from_p(0.001)}"
    # NaN guard
    import math
    assert z_from_p(float('nan')) == 0.0, "z_from_p(NaN) should return 0.0"


# ---------------------------------------------------------------------------
# T4c: compute_se_from_p with negative estimate
# ---------------------------------------------------------------------------

def test_se_from_p_negative_estimate():
    """T4c: Negative estimates handled correctly via abs()."""
    se_neg = compute_se_from_p(-0.5, 0.001)
    se_pos = compute_se_from_p(0.5, 0.001)
    assert se_neg is not None
    assert abs(se_neg - se_pos) < 1e-10, "SE should be same for +/- estimate"
    # NaN estimates return None
    se_nan = compute_se_from_p(float('nan'), 0.05)
    assert se_nan is None, "NaN estimate should return None"


# ---------------------------------------------------------------------------
# T5: trust_weight_ma
# ---------------------------------------------------------------------------

def test_trust_weight():
    """T5: trust_weight = original_weight * score/100."""
    result = trust_weight_ma(0.5, 0.001, 80)
    assert result is not None
    orig = result["original_weight"]
    tw = result["trust_weight"]
    assert abs(tw - orig * 0.8) < 1e-9, (
        f"trust_weight {tw} != original_weight {orig} * 0.8"
    )


# ---------------------------------------------------------------------------
# T6: trust_weight_zero_score
# ---------------------------------------------------------------------------

def test_trust_weight_zero_score():
    """T6: score=0 → trust_weight=0."""
    result = trust_weight_ma(0.5, 0.001, 0)
    assert result["trust_weight"] == 0.0, (
        f"Expected trust_weight=0.0 for score=0, got {result['trust_weight']}"
    )


# ---------------------------------------------------------------------------
# T7: erosion_curve_low_threshold
# ---------------------------------------------------------------------------

def test_erosion_curve_low_threshold(erosion_scores, erosion_verdicts):
    """T7: threshold 50 — MA003 (score 42) is excluded, 2 MAs remain."""
    merged = pd.merge(erosion_scores, erosion_verdicts, on="ma_id", how="inner")
    curve = compute_erosion_curve(merged)

    row = curve[curve["threshold"] == 50].iloc[0]
    # 3 originally significant MAs total
    assert row["n_total_sig"] == 3, f"Expected 3 significant MAs, got {row['n_total_sig']}"
    # MA003 (score 42) is excluded at threshold 50
    assert row["n_excluded"] == 1, f"Expected n_excluded=1 at threshold 50, got {row['n_excluded']}"
    # Remaining = 3 - 1 = 2
    assert row["n_remaining"] == 2, f"Expected n_remaining=2 at threshold 50, got {row['n_remaining']}"


# ---------------------------------------------------------------------------
# T8: erosion_curve_high_threshold
# ---------------------------------------------------------------------------

def test_erosion_curve_high_threshold(erosion_scores, erosion_verdicts):
    """T8: threshold 90 — only MA002 (score 96) survives; 2 excluded."""
    merged = pd.merge(erosion_scores, erosion_verdicts, on="ma_id", how="inner")
    curve = compute_erosion_curve(merged)

    row = curve[curve["threshold"] == 90].iloc[0]
    # MA001 (79) and MA003 (42) are excluded; only MA002 (96) passes threshold
    assert row["n_excluded"] == 2, f"Expected n_excluded=2 at threshold 90, got {row['n_excluded']}"
    assert row["n_remaining"] == 1, f"Expected n_remaining=1 at threshold 90, got {row['n_remaining']}"


# ---------------------------------------------------------------------------
# T9: erosion_rate_zero
# ---------------------------------------------------------------------------

def test_erosion_rate_zero():
    """T9: all MAs score 100 → erosion_rate=0 at every threshold."""
    # Build a merged DataFrame with 3 significant MAs all scoring 100
    merged = pd.DataFrame({
        "ma_id":       ["M1", "M2", "M3"],
        "review_id":   ["R1", "R2", "R3"],
        "final_score": [100, 100, 100],
        "estimate":    [0.5, 0.4, 0.6],
        "p_value":     [0.001, 0.005, 0.010],
        "significant": [True, True, True],
    })
    curve = compute_erosion_curve(merged)

    for _, row in curve.iterrows():
        assert row["erosion_rate"] == 0.0, (
            f"Expected erosion_rate=0 at threshold {row['threshold']}, "
            f"got {row['erosion_rate']}"
        )


# ---------------------------------------------------------------------------
# T10: load_review_groups
# ---------------------------------------------------------------------------

REVIEW_GROUPS_CSV = """\
review_id_prefix,review_group
MA001,Cardiology
MA002,Oncology
MA003,Cardiology
"""


def test_load_review_groups(tmp_path):
    """T10: load_review_groups returns 3 rows and review_group column."""
    csv_path = tmp_path / "review_groups.csv"
    csv_path.write_text(REVIEW_GROUPS_CSV)

    df = load_review_groups(path=str(csv_path))

    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
    assert "review_group" in df.columns, "review_group column missing"
    assert "review_id_prefix" in df.columns, "review_id_prefix column missing"


def test_load_review_groups_missing_file(tmp_path):
    """T10b: load_review_groups returns empty DataFrame when file missing."""
    df = load_review_groups(path=str(tmp_path / "nonexistent.csv"))

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert len(df) == 0, f"Expected 0 rows for missing file, got {len(df)}"
    assert "review_group" in df.columns, "review_group column should be present even when empty"


# ---------------------------------------------------------------------------
# T11: compute_domain_erosion (grouping)
# ---------------------------------------------------------------------------

def test_domain_erosion_grouping(erosion_scores, erosion_verdicts):
    """T11: compute_domain_erosion at threshold 70 has entries for expected domains."""
    # Merge base fixtures
    merged = pd.merge(erosion_scores, erosion_verdicts, on="ma_id", how="inner")

    # Assign domain groups: MA001->Cardiology, MA002->Oncology, MA003->Cardiology, MA004->Oncology
    group_map = {
        "MA001": "Cardiology",
        "MA002": "Oncology",
        "MA003": "Cardiology",
        "MA004": "Oncology",
    }
    merged["review_group"] = merged["ma_id"].map(group_map)

    result = compute_domain_erosion(merged, thresholds=[70])

    # Should have one row per domain per threshold
    assert len(result) == 2, f"Expected 2 domain rows at threshold 70, got {len(result)}"

    domains_found = set(result["domain"].tolist())
    assert "Cardiology" in domains_found, f"'Cardiology' missing from domains: {domains_found}"
    assert "Oncology" in domains_found, f"'Oncology' missing from domains: {domains_found}"

    # Verify required columns
    for col in ("domain", "threshold", "n_total_sig", "n_excluded", "erosion_rate", "n_mas"):
        assert col in result.columns, f"Column '{col}' missing from domain erosion output"

    # Cardiology: MA001 (score 79, sig), MA003 (score 42, sig) → at threshold 70, MA003 excluded
    cardio = result[result["domain"] == "Cardiology"].iloc[0]
    assert cardio["n_total_sig"] == 2, f"Cardiology n_total_sig: expected 2, got {cardio['n_total_sig']}"
    assert cardio["n_excluded"] == 1, f"Cardiology n_excluded: expected 1 (MA003 score 42 < 70), got {cardio['n_excluded']}"

    # Oncology: MA002 (score 96, sig), MA004 (score 62, not sig) → at threshold 70, 0 sig excluded
    onco = result[result["domain"] == "Oncology"].iloc[0]
    assert onco["n_total_sig"] == 1, f"Oncology n_total_sig: expected 1, got {onco['n_total_sig']}"
    assert onco["n_excluded"] == 0, f"Oncology n_excluded: expected 0 (MA002 score 96 >= 70), got {onco['n_excluded']}"


# ---------------------------------------------------------------------------
# T12: compute_domain_erosion (no significant MAs → erosion_rate=0)
# ---------------------------------------------------------------------------

def test_domain_erosion_no_significant():
    """T12: domain with only non-significant MAs has erosion_rate=0."""
    merged = pd.DataFrame({
        "ma_id":        ["M1", "M2"],
        "review_id":    ["R1", "R2"],
        "final_score":  [30, 40],
        "estimate":     [0.1, 0.2],
        "p_value":      [0.3, 0.4],
        "significant":  [False, False],
        "review_group": ["Neurology", "Neurology"],
    })

    result = compute_domain_erosion(merged, thresholds=[50])

    assert len(result) == 1, f"Expected 1 row, got {len(result)}"
    row = result.iloc[0]
    assert row["domain"] == "Neurology"
    assert row["n_total_sig"] == 0, f"Expected n_total_sig=0, got {row['n_total_sig']}"
    assert row["erosion_rate"] == 0.0, f"Expected erosion_rate=0.0, got {row['erosion_rate']}"


# ---------------------------------------------------------------------------
# T13: compute_citation_percentile — basic normalization
# ---------------------------------------------------------------------------

def test_citation_percentile_basic():
    """T13: Series [0, 5, 10, 50, 100] → percentiles in [0,100], highest > lowest."""
    citations = pd.Series([0, 5, 10, 50, 100])
    result = compute_citation_percentile(citations)

    assert len(result) == 5, f"Expected 5 values, got {len(result)}"
    assert result.min() >= 0.0, f"Min percentile should be >= 0, got {result.min()}"
    assert result.max() <= 100.0, f"Max percentile should be <= 100, got {result.max()}"

    # The entry corresponding to count=100 must be higher than count=0
    idx_max = citations.idxmax()
    idx_min = citations.idxmin()
    assert result[idx_max] > result[idx_min], (
        f"Highest-cited entry percentile {result[idx_max]} should exceed "
        f"lowest-cited entry percentile {result[idx_min]}"
    )


# ---------------------------------------------------------------------------
# T14: compute_citation_percentile — all-same guard
# ---------------------------------------------------------------------------

def test_citation_percentile_all_same():
    """T14: Series [10, 10, 10, 10] → all percentiles equal 50.0."""
    citations = pd.Series([10, 10, 10, 10])
    result = compute_citation_percentile(citations)

    assert len(result) == 4, f"Expected 4 values, got {len(result)}"
    for val in result:
        assert val == 50.0, f"Expected 50.0 for all-same series, got {val}"


# ---------------------------------------------------------------------------
# T15: compute_influence_score — all sources contribute
# ---------------------------------------------------------------------------

def test_influence_score_all_sources():
    """T15: citation_pct=80, who=True, nice=2, group_pct=60 → 78.0."""
    # 80*0.4 + 20 + min(2*10, 30) + 60*0.1 = 32 + 20 + 20 + 6 = 78
    score = compute_influence_score(
        citation_percentile=80,
        who_essential=True,
        nice_guideline_count=2,
        group_size_percentile=60,
    )
    assert abs(score - 78.0) < 1e-9, f"Expected 78.0, got {score}"


# ---------------------------------------------------------------------------
# T16: compute_influence_score — minimal (no WHO, no NICE)
# ---------------------------------------------------------------------------

def test_influence_score_minimal():
    """T16: citation_pct=50, who=False, nice=0, group_pct=50 → 25.0."""
    # 50*0.4 + 0 + 0 + 50*0.1 = 20 + 0 + 0 + 5 = 25
    score = compute_influence_score(
        citation_percentile=50,
        who_essential=False,
        nice_guideline_count=0,
        group_size_percentile=50,
    )
    assert abs(score - 25.0) < 1e-9, f"Expected 25.0, got {score}"


# ---------------------------------------------------------------------------
# T17: match_who_medicines — exact match in text
# ---------------------------------------------------------------------------

def test_who_match_exact():
    """T17: 'metoprolol vs placebo for heart failure' matches WHO medicines list."""
    who_df = pd.DataFrame({
        "medicine_name": ["metoprolol", "aspirin", "amoxicillin"],
        "category": ["cardiovascular", "analgesic", "antibiotic"],
    })
    interventions = {"R1": "metoprolol vs placebo for heart failure"}
    results = match_who_medicines(interventions, who_df)
    assert results["R1"] == True, (
        f"Expected True for 'metoprolol vs placebo for heart failure', got {results['R1']}"
    )


# ---------------------------------------------------------------------------
# T18: match_who_medicines — no match in text
# ---------------------------------------------------------------------------

def test_who_match_none():
    """T18: 'cognitive behavioral therapy for depression' does not match WHO medicines."""
    who_df = pd.DataFrame({
        "medicine_name": ["metoprolol", "aspirin", "amoxicillin"],
        "category": ["cardiovascular", "analgesic", "antibiotic"],
    })
    interventions = {"R1": "cognitive behavioral therapy for depression"}
    results = match_who_medicines(interventions, who_df)
    assert results["R1"] == False, (
        f"Expected False for 'cognitive behavioral therapy for depression', got {results['R1']}"
    )


# ---------------------------------------------------------------------------
# T19: assign_quadrant — all five quadrants
# ---------------------------------------------------------------------------

def test_assign_quadrant():
    """T19: Each quadrant boundary maps to the correct label."""
    # Red Flag: trust < 60 AND influence > 50
    assert assign_quadrant(45, 55) == "Red Flag", (
        f"Expected 'Red Flag' for trust=45, influence=55, "
        f"got '{assign_quadrant(45, 55)}'"
    )
    # Hidden Gem: trust >= 80 AND influence < 30
    assert assign_quadrant(90, 20) == "Hidden Gem", (
        f"Expected 'Hidden Gem' for trust=90, influence=20, "
        f"got '{assign_quadrant(90, 20)}'"
    )
    # Safe: trust >= 80 AND influence > 50
    assert assign_quadrant(90, 55) == "Safe", (
        f"Expected 'Safe' for trust=90, influence=55, "
        f"got '{assign_quadrant(90, 55)}'"
    )
    # Low Stakes: trust < 60 AND influence < 30
    assert assign_quadrant(45, 20) == "Low Stakes", (
        f"Expected 'Low Stakes' for trust=45, influence=20, "
        f"got '{assign_quadrant(45, 20)}'"
    )
    # Moderate: middle zone (trust=70, influence=40)
    assert assign_quadrant(70, 40) == "Moderate", (
        f"Expected 'Moderate' for trust=70, influence=40, "
        f"got '{assign_quadrant(70, 40)}'"
    )


# ---------------------------------------------------------------------------
# T19b: boundary values at exact thresholds
# ---------------------------------------------------------------------------

def test_assign_quadrant_boundaries():
    """T19b: Exact boundary values for quadrant assignment."""
    # trust=60 is NOT < 60, so not Red Flag or Low Stakes → Moderate
    assert assign_quadrant(60, 55) == "Moderate"
    # trust=80 is >= 80 → could be Safe/Hidden Gem
    assert assign_quadrant(80, 55) == "Safe"
    assert assign_quadrant(80, 20) == "Hidden Gem"
    # influence=50 exactly: > 50 is False, so not Red Flag/Safe
    assert assign_quadrant(45, 50) == "Moderate"
    assert assign_quadrant(90, 50) == "Moderate"
    # influence=30 exactly: not < 30, so not Hidden Gem/Low Stakes
    assert assign_quadrant(90, 30) == "Moderate"
    assert assign_quadrant(45, 30) == "Moderate"


# ---------------------------------------------------------------------------
# T20: build_risk_register — quadrant column on full DataFrame
# ---------------------------------------------------------------------------

def test_build_risk_register():
    """T20: build_risk_register adds quadrant column, len=5, Red Flag count=1."""
    df = pd.DataFrame({
        "ma_id": ["MA_RF", "MA_HG", "MA_SA", "MA_LS", "MA_MO"],
        "final_score":    [45,  90,  90,  45,  70],   # trust proxy
        "influence_score":[55,  20,  55,  20,  40],   # influence proxy (threshold=50)
    })

    result = build_risk_register(df)

    # quadrant column must exist
    assert "quadrant" in result.columns, "quadrant column missing from result"

    # Length preserved
    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"

    # Original DataFrame is not mutated
    assert "quadrant" not in df.columns, "build_risk_register mutated the input DataFrame"

    # Verify each quadrant label
    assert result.loc[result["ma_id"] == "MA_RF", "quadrant"].iloc[0] == "Red Flag"
    assert result.loc[result["ma_id"] == "MA_HG", "quadrant"].iloc[0] == "Hidden Gem"
    assert result.loc[result["ma_id"] == "MA_SA", "quadrant"].iloc[0] == "Safe"
    assert result.loc[result["ma_id"] == "MA_LS", "quadrant"].iloc[0] == "Low Stakes"
    assert result.loc[result["ma_id"] == "MA_MO", "quadrant"].iloc[0] == "Moderate"

    # Exactly one Red Flag
    red_flag_count = (result["quadrant"] == "Red Flag").sum()
    assert red_flag_count == 1, f"Expected 1 Red Flag row, got {red_flag_count}"


# ---------------------------------------------------------------------------
# T21: test_pipeline_fixture
# ---------------------------------------------------------------------------

def test_pipeline_fixture(tmp_path):
    """T21: run_pipeline on fixture CSVs returns expected keys and 4 merged rows."""
    # Write fixture CSVs to tmp_path
    scores_path = tmp_path / "scores.csv"
    verdicts_path = tmp_path / "verdicts.csv"
    dois_path = tmp_path / "dois.csv"

    scores_path.write_text(SCORES_CSV, encoding="utf-8")
    verdicts_path.write_text(VERDICTS_CSV, encoding="utf-8")
    dois_path.write_text(DOIS_CSV, encoding="utf-8")

    results = run_pipeline(
        scores_path=str(scores_path),
        verdicts_path=str(verdicts_path),
        dois_path=str(dois_path),
        fetch_citations=False,
    )

    # Required keys present
    for key in ("erosion_curve", "risk_register", "merged", "summary", "domain_erosion"):
        assert key in results, f"Missing key '{key}' in pipeline results"

    merged = results["merged"]
    assert len(merged) == 4, f"Expected 4 rows in merged, got {len(merged)}"


# ---------------------------------------------------------------------------
# T22: test_score_clamping
# ---------------------------------------------------------------------------

def test_score_clamping(tmp_path):
    """T22: influence_score in [0,100] and erosion_rate in [0,100]."""
    scores_path = tmp_path / "scores.csv"
    verdicts_path = tmp_path / "verdicts.csv"
    dois_path = tmp_path / "dois.csv"

    scores_path.write_text(SCORES_CSV, encoding="utf-8")
    verdicts_path.write_text(VERDICTS_CSV, encoding="utf-8")
    dois_path.write_text(DOIS_CSV, encoding="utf-8")

    results = run_pipeline(
        scores_path=str(scores_path),
        verdicts_path=str(verdicts_path),
        dois_path=str(dois_path),
        fetch_citations=False,
    )

    rr = results["risk_register"]
    assert rr["influence_score"].min() >= 0, (
        f"influence_score min {rr['influence_score'].min()} < 0"
    )
    assert rr["influence_score"].max() <= 100, (
        f"influence_score max {rr['influence_score'].max()} > 100"
    )

    ec = results["erosion_curve"]
    assert ec["erosion_rate"].min() >= 0, (
        f"erosion_rate min {ec['erosion_rate'].min()} < 0"
    )
    assert ec["erosion_rate"].max() <= 100, (
        f"erosion_rate max {ec['erosion_rate'].max()} > 100"
    )


# ---------------------------------------------------------------------------
# T23: test_join_integrity
# ---------------------------------------------------------------------------

def test_join_integrity(tmp_path):
    """T23: merged.final_score and merged.p_value have no NaN after pipeline."""
    scores_path = tmp_path / "scores.csv"
    verdicts_path = tmp_path / "verdicts.csv"
    dois_path = tmp_path / "dois.csv"

    scores_path.write_text(SCORES_CSV, encoding="utf-8")
    verdicts_path.write_text(VERDICTS_CSV, encoding="utf-8")
    dois_path.write_text(DOIS_CSV, encoding="utf-8")

    results = run_pipeline(
        scores_path=str(scores_path),
        verdicts_path=str(verdicts_path),
        dois_path=str(dois_path),
        fetch_citations=False,
    )

    merged = results["merged"]
    assert merged["final_score"].notna().all(), (
        f"final_score has {merged['final_score'].isna().sum()} NaN value(s)"
    )
    assert merged["p_value"].notna().all(), (
        f"p_value has {merged['p_value'].isna().sum()} NaN value(s)"
    )


# ---------------------------------------------------------------------------
# T24: test_summary_consistency
# ---------------------------------------------------------------------------

def test_summary_consistency(tmp_path):
    """T24: summary totals match fixture data (4 MAs, 3 significant, valid rates)."""
    scores_path = tmp_path / "scores.csv"
    verdicts_path = tmp_path / "verdicts.csv"
    dois_path = tmp_path / "dois.csv"

    scores_path.write_text(SCORES_CSV, encoding="utf-8")
    verdicts_path.write_text(VERDICTS_CSV, encoding="utf-8")
    dois_path.write_text(DOIS_CSV, encoding="utf-8")

    results = run_pipeline(
        scores_path=str(scores_path),
        verdicts_path=str(verdicts_path),
        dois_path=str(dois_path),
        fetch_citations=False,
    )

    summary = results["summary"]

    assert summary["total_mas"] == 4, (
        f"Expected total_mas=4, got {summary['total_mas']}"
    )
    assert summary["total_significant"] == 3, (
        f"Expected total_significant=3, got {summary['total_significant']}"
    )
    assert 0 <= summary["erosion_rate_at_70"] <= 100, (
        f"erosion_rate_at_70 out of range: {summary['erosion_rate_at_70']}"
    )
    # quadrant_counts values must sum to total_mas (4)
    qc_total = sum(summary["quadrant_counts"].values())
    assert qc_total == 4, (
        f"quadrant_counts total {qc_total} != total_mas 4. "
        f"quadrant_counts={summary['quadrant_counts']}"
    )


# ---------------------------------------------------------------------------
# T25: test_integration_real_data
# ---------------------------------------------------------------------------

def test_integration_real_data():
    """T25: Full pipeline on real data (skip if SCORES_CSV not found)."""
    import os
    from pathlib import Path

    # Skip if the real scores file doesn't exist
    if not Path(ENGINE_SCORES_CSV).exists():
        pytest.skip(f"Real data not found: {ENGINE_SCORES_CSV}")

    results = run_pipeline(fetch_citations=False)
    merged = results["merged"]

    assert len(merged) > 0, "No rows returned from real-data pipeline"

    # All final_scores in valid range
    assert merged["final_score"].between(0, 100).all(), (
        "Some final_score values are outside [0, 100]"
    )

    # All grades are valid (non-null strings)
    valid_grades = {"A+", "A", "B+", "B", "B-", "C+", "C", "C-", "D", "F"}
    if "grade" in merged.columns:
        actual_grades = set(merged["grade"].dropna().unique())
        invalid_grades = actual_grades - valid_grades
        assert not invalid_grades, (
            f"Unexpected grade values: {invalid_grades}"
        )
