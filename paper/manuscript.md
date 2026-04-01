# TrustGate: Trust-Weighted Significance Erosion and Guideline Exposure Mapping Across 6,229 Cochrane Meta-Analyses

## Abstract

**Background:** Systematic reviews with meta-analysis form the evidence foundation for clinical guidelines, yet the trustworthiness of individual meta-analyses varies substantially. We investigated how many statistically significant meta-analytic findings survive when weighted by composite trust scores, and whether low-trust meta-analyses disproportionately influence clinical practice.

**Methods:** We applied the EvidenceScore composite trust metric (0-100, combining audit quality, consistency, robustness, stability, and statistical power) to 6,229 meta-analyses from 501 Cochrane systematic reviews. We performed trust-weighted significance analysis using both linear inverse-variance weighting adjusted by trust score and threshold gating at five trust levels (50, 60, 70, 80, 90). Clinical influence was estimated using a composite score incorporating PubMed citation counts, WHO Essential Medicines List matching, and Cochrane Review Group size. Meta-analyses were classified into risk quadrants crossing trust with clinical influence.

**Results:** Of 888 originally significant meta-analyses (p < 0.05), only 321 (36.1%) survived trust-gating at the B+ threshold (score >= 70), representing a significance erosion rate of 63.9%. At the highest threshold (score >= 90), only 71 (8.0%) survived. Domain-specific erosion was most severe in pain research (73.2% erosion at B+) and cardiovascular medicine (71.3%), while cancer research was relatively resilient (34.2%). We identified 104 Red Flag meta-analyses (1.7%) combining low trust (score < 60) with high clinical influence, clustered in 12 reviews. Conversely, 366 Hidden Gem meta-analyses (5.9%) demonstrated high trustworthiness (score >= 80) but low clinical influence, suggesting underutilization of robust evidence.

**Conclusions:** Nearly two-thirds of significant Cochrane meta-analytic findings do not withstand trust-weighting, with marked variation across medical specialties. A small but concerning subset of low-trust meta-analyses commands disproportionate clinical influence. These findings argue for trust-calibrated evidence synthesis and systematic identification of high-quality evidence that remains underutilized in practice.

---

## Introduction

Meta-analyses are widely regarded as the highest level of evidence in the hierarchy of evidence-based medicine.^1^ They form the evidentiary backbone of clinical practice guidelines issued by organisations including NICE, the AHA, and the WHO. However, the methodological quality and statistical robustness of individual meta-analyses vary enormously.^2,3^

Recent large-scale audits have documented pervasive quality concerns in the Cochrane Library. MetaAudit, an 11-detector automated audit system, found significant issues across the majority of Cochrane meta-analyses, including fragility, model misspecification, and excess significance.^4^ The EvidenceOracle machine learning system predicted that a substantial proportion of meta-analytic conclusions are unstable to future evidence updates.^5^ ContradictionMap revealed that 48.9% of meta-analytic pairs addressing overlapping clinical questions produced contradictory conclusions.^6^

These individual signals were synthesized into the EvidenceScore, a composite 0-100 trust metric combining five domains: audit quality (30% weight), evidence consistency (20%), statistical robustness (20%), prediction stability (15%), and statistical power (15%).^7^ Applied to 6,229 Cochrane meta-analyses, EvidenceScore revealed a mean trust score of 69.0 (SD 9.6), with only 17.4% scoring A or above (>= 80).

A critical question remains: **what are the practical implications of variable meta-analytic trust for clinical decision-making?** If low-trust meta-analyses are rarely cited and minimally influential, the problem may be academic. But if low-trust evidence disproportionately drives clinical practice, the gap between evidence quality and clinical reliance becomes a patient safety concern.

We developed TrustGate, an analytical framework that addresses two questions:

1. How many statistically significant meta-analytic findings survive when weighted by trust? (significance erosion analysis)
2. Which low-trust meta-analyses command the greatest clinical influence, and which high-trust meta-analyses are underutilised? (guideline exposure mapping)

## Methods

### Data sources

We analysed 6,229 meta-analyses from 501 Cochrane systematic reviews previously characterised by the Evidence Intelligence Suite.^4-7^ For each meta-analysis, we obtained:

- **Trust scores** from EvidenceScore: composite 0-100 score with five component subscores (audit, consistency, robustness, stability, power) and letter grades (A+ through F)
- **Effect estimates** from ActionableEvidence: pooled effect sizes, p-values, study counts (k), and total sample sizes (N) for each meta-analysis
- **Publication metadata** from the Pairwise70 dataset: Cochrane DOIs enabling citation tracking
- **Clinical domain classification** from Cochrane review metadata: 14 clinical domains (cardiovascular, cancer, pain, diabetes, mental health, respiratory, infection, hypertension, pregnancy, gastrointestinal, dermatology, neurological, renal, and other)

### Part A: Trust-weighted significance analysis

#### Linear trust-weighting

For each meta-analysis with a statistically significant result (p < 0.05, n = 888), we derived the standard error from the reported effect estimate and p-value using the relationship SE = |estimate| / z, where z is the inverse-normal quantile corresponding to the p-value.^8^ The standard inverse-variance weight (w = 1/SE^2) was then adjusted by trust:

w_trust = w_original x (EvidenceScore / 100)

A meta-analysis was classified as "weakened" if its trust-adjusted z-statistic fell below 1.96 (the two-sided 5% significance threshold).

#### Threshold gating

For each of five trust thresholds (T = 50, 60, 70, 80, 90), we removed all meta-analyses with EvidenceScore < T and counted:

- **Surviving**: included (score >= T) and significant after trust-weighting
- **Weakened**: included but losing significance after trust-weighting
- **Excluded**: removed (score < T)

The **Significance Erosion Rate** was defined as:

Erosion Rate (%) = (N_excluded + N_weakened) / N_total_significant x 100

#### Domain-specific analysis

Erosion rates were computed separately for each of the 14 clinical domains to identify specialty-level variation.

### Part B: Guideline exposure mapping

#### Clinical Influence Score

We constructed a composite Clinical Influence Score (0-100) for each Cochrane review using four sources:

1. **Citation percentile** (weight 0.4): PubMed citation counts retrieved via the NCBI E-utilities API, normalised to percentile rank within the dataset
2. **WHO Essential Medicines** (weight +20 points): Binary flag indicating whether the review evaluates an intervention appearing on the WHO Model List of Essential Medicines, matched via whole-word text search against outcome and intervention descriptions
3. **NICE guideline citations** (weight up to +30 points): Count of NICE guideline recommendations citing the Cochrane review (data enrichment pending; set to 0 in the current analysis)
4. **Review Group activity** (weight 0.1): Percentile rank of the Cochrane Review Group size, proxying the clinical activity level of the therapeutic area

#### Risk register

Meta-analyses were classified into five quadrants by crossing EvidenceScore (trust) with Clinical Influence Score:

| Quadrant | Trust | Influence | Interpretation |
|----------|-------|-----------|---------------|
| Red Flag | < 60 | > 50 | Low trust, high influence: potential patient safety concern |
| Hidden Gem | >= 80 | < 30 | High trust, low influence: underutilised robust evidence |
| Safe | >= 80 | > 50 | High trust, high influence: working as intended |
| Low Stakes | < 60 | < 30 | Low trust, low influence: lower priority for action |
| Moderate | 60-79 or 30-50 | 30-50 or 60-79 | Intermediate zone |

### Software and reproducibility

All analyses were conducted using TrustGate v1.0, a Python-based open-source pipeline (available at [repository URL]). The analysis pipeline processes input data deterministically from cached data sources, requiring no external API calls for replication. All 26 unit and integration tests pass. The interactive dashboard is available as a single-file HTML application.

## Results

### Significance erosion

Of 6,229 Cochrane meta-analyses, 888 (14.3%) had originally significant results (p < 0.05). Applying trust-weighted significance analysis revealed progressive erosion across thresholds (Table 1).

**Table 1. Significance erosion across trust thresholds**

| Threshold | Grade | Surviving | Weakened | Excluded | Erosion Rate |
|-----------|-------|-----------|----------|----------|-------------|
| 50 | D+ | 570 (64.2%) | 254 (28.6%) | 64 (7.2%) | 35.8% |
| 60 | C+ | 466 (52.5%) | 131 (14.8%) | 291 (32.8%) | 47.5% |
| 70 | B+ | 321 (36.1%) | 60 (6.8%) | 507 (57.1%) | 63.9% |
| 80 | A/A+ | 155 (17.5%) | 18 (2.0%) | 715 (80.5%) | 82.5% |
| 90 | A+ | 71 (8.0%) | 0 (0.0%) | 817 (92.0%) | 92.0% |

At the B+ threshold (score >= 70), which represents moderately trustworthy evidence, 63.9% of significant findings were eroded -- 507 excluded due to insufficient trust and an additional 60 weakened by trust-adjusted weighting. Only 321 of 888 originally significant findings (36.1%) survived.

### Domain-specific erosion

Erosion rates at the B+ threshold varied substantially across clinical domains (Table 2).

**Table 2. Significance erosion by clinical domain at B+ threshold (score >= 70)**

| Domain | Sig. MAs | Total MAs | Erosion Rate | Rank |
|--------|----------|-----------|-------------|------|
| Pain | 97 | 380 | 73.2% | 1 (worst) |
| Cardiovascular | 101 | 640 | 71.3% | 2 |
| Other | 125 | 961 | 64.8% | 3 |
| Diabetes | 124 | 549 | 60.5% | 4 |
| Pregnancy | 40 | 377 | 60.0% | 5 |
| Infection | 49 | 415 | 59.2% | 6 |
| Mental health | 90 | 703 | 55.6% | 7 |
| Respiratory | 38 | 359 | 50.0% | 8 |
| Hypertension | 106 | 1,064 | 41.5% | 9 |
| Gastrointestinal | 39 | 112 | 35.9% | 10 |
| Cancer | 38 | 235 | 34.2% | 11 |
| Dermatology | 4 | 15 | 0.0% | 12 |
| Neurological | 5 | 27 | 0.0% | 13 |

Pain research showed the highest erosion (73.2%), meaning nearly three-quarters of statistically significant pain meta-analyses do not withstand trust-gating at a moderate threshold. Cardiovascular medicine followed closely (71.3%). In contrast, cancer research demonstrated relative resilience (34.2%), potentially reflecting the rigour of oncology trial methodology and the influence of regulatory requirements on trial design.

### Risk register: Red Flags and Hidden Gems

The risk quadrant analysis classified all 6,229 meta-analyses by trust and clinical influence (Table 3).

**Table 3. Risk quadrant distribution**

| Quadrant | N | % | Interpretation |
|----------|---|---|---------------|
| Moderate | 4,594 | 73.8% | Intermediate trust and/or influence |
| Low Stakes | 1,131 | 18.2% | Low trust, low influence |
| Hidden Gem | 366 | 5.9% | High trust (>= 80), low influence |
| Red Flag | 104 | 1.7% | Low trust (< 60), high influence (> 50) |
| Safe | 34 | 0.5% | High trust, high influence |

#### Red Flags

We identified 104 Red Flag meta-analyses (1.7%) across 12 Cochrane reviews. These meta-analyses scored D or F on the trust scale (scores 43-59) yet commanded high clinical influence scores (52.4-61.8). All 104 involved WHO Essential Medicines and had moderate to high citation counts. The Red Flag distribution by grade was 88 grade D (84.6%) and 16 grade F (15.4%).

#### Hidden Gems

A substantially larger group of 366 meta-analyses (5.9%) from 67 reviews demonstrated high trustworthiness (scores 80-98) but low clinical influence (< 30). These represent an untapped reservoir of robust evidence that could strengthen clinical guidelines if actively promoted.

#### The trust-influence gap

The ratio of Red Flags to Safe meta-analyses (104:34, approximately 3:1) indicates that the current evidence landscape is more likely to over-rely on low-trust evidence than to appropriately utilise high-trust evidence. The 366 Hidden Gems represent approximately 3.5 times more opportunities for improvement than the 104 Red Flags represent risks.

## Discussion

### Principal findings

This study demonstrates that the statistical significance of meta-analytic findings is substantially sensitive to evidence trustworthiness. At a moderate trust threshold (B+, score >= 70), 63.9% of significant Cochrane meta-analyses fail trust-gating, and this erosion is unevenly distributed across medical specialties. Pain and cardiovascular research are most severely affected, while oncology research is relatively resilient. A small but concerning set of 104 meta-analyses (1.7%) combines low trust with high clinical influence, while a larger set of 366 (5.9%) represents underutilised high-quality evidence.

### Comparison with existing literature

Previous work has documented individual aspects of meta-analytic quality -- fragility,^9^ publication bias,^10^ and model misspecification^11^ -- in isolation. TrustGate synthesizes these concerns into a single actionable framework. Our finding that 63.9% of significant findings are trust-eroded at a moderate threshold extends Ioannidis's observation that "most published research findings are false"^12^ into the specific domain of meta-analysis, where the assumption of evidence aggregation should, in principle, mitigate individual study limitations.

The domain-specific variation is particularly informative. The high erosion in pain research (73.2%) aligns with known concerns about trial quality in this field,^13^ while the relative resilience of cancer research (34.2%) may reflect the influence of FDA regulatory requirements and CONSORT adherence on oncology trial methodology.^14^

### Clinical implications

The identification of 104 Red Flag meta-analyses across 12 reviews warrants immediate attention. These represent evidence that is simultaneously untrustworthy and influential -- the most dangerous quadrant of the trust-influence space. Clinical guideline developers should prioritise re-evaluation of recommendations that rely on these meta-analyses.

Equally important, the 366 Hidden Gems represent a constructive pathway forward. Rather than simply flagging problems, TrustGate identifies where robust evidence exists but remains underutilised. Systematic promotion of Hidden Gem evidence into clinical guidelines could improve evidence quality without the delay of conducting new trials.

### Strengths and limitations

**Strengths.** This study analyses the entirety of eligible Cochrane meta-analyses (N = 6,229) rather than a sample. The composite trust score integrates five independent quality dimensions. The analysis is fully reproducible from openly available data and code.

**Limitations.** First, the EvidenceScore composite weights are based on domain expertise rather than empirically calibrated; sensitivity analyses with alternative weight structures would strengthen the findings. Second, clinical influence was measured using citation counts and WHO essential medicine status as proxies for guideline impact; direct guideline citation analysis (e.g., through NICE evidence reviews) would provide more precise influence estimates. Third, the trust-weighting approach derives standard errors from reported p-values and estimates, which introduces approximation error, particularly for meta-analyses that use non-standard statistical methods. Fourth, our analysis is limited to Cochrane reviews and may not generalise to non-Cochrane meta-analyses.

### Future directions

Integration of direct guideline citation data from NICE, AHA, and ESC would sharpen the influence estimates and likely reveal additional Red Flag meta-analyses currently classified as Moderate. Longitudinal application of TrustGate could track how the evidence landscape changes as new reviews are published and old ones are updated.

## Conclusions

Trust-weighted significance analysis reveals that nearly two-thirds of statistically significant Cochrane meta-analyses do not withstand moderate trust-gating, with substantial variation across medical specialties. The identification of 104 Red Flag meta-analyses with low trust but high clinical influence argues for systematic trust-calibration of evidence used in clinical guideline development. Simultaneously, 366 Hidden Gem meta-analyses represent robust evidence that is currently underutilised. TrustGate provides an automated, reproducible framework for this dual assessment.

---

## Data availability

All input data are derived from the publicly available Cochrane Library. The TrustGate analysis code, test suite (26/26 tests passing), and interactive dashboard are available as open-source software at [repository URL]. Results can be reproduced from cached data without external API calls.

## References

1. Higgins JPT, Thomas J, Chandler J, et al. Cochrane Handbook for Systematic Reviews of Interventions. 2nd ed. Chichester: John Wiley & Sons; 2019.
2. Page MJ, Shamseer L, Altman DG, et al. Epidemiology and reporting characteristics of systematic reviews of biomedical research: a cross-sectional study. PLoS Med. 2016;13(5):e1002028.
3. Ioannidis JPA. The mass production of redundant, misleading, and conflicted systematic reviews and meta-analyses. Milbank Q. 2016;94(3):485-514.
4. [MetaAudit reference -- under review]
5. [EvidenceOracle reference -- under review]
6. [ContradictionMap reference -- under review]
7. [EvidenceScore reference -- under review]
8. Altman DG, Bland JM. How to obtain the confidence interval from a P value. BMJ. 2011;343:d2090.
9. Walsh M, Srinathan SK, McAuley DF, et al. The statistical significance of randomized controlled trial results is frequently fragile: a case for a Fragility Index. J Clin Epidemiol. 2014;67(6):622-628.
10. Sterne JAC, Sutton AJ, Ioannidis JPA, et al. Recommendations for examining and interpreting funnel plot asymmetry in meta-analyses of randomised controlled trials. BMJ. 2011;343:d4002.
11. Jackson D, White IR, Riley RD. Quantifying the impact of between-study heterogeneity in multivariate meta-analyses. Stat Med. 2012;31(29):3805-3820.
12. Ioannidis JPA. Why most published research findings are false. PLoS Med. 2005;2(8):e124.
13. Moore RA, Derry S, McQuay HJ, Wiffen PJ. What do we know about communicating risk? A brief review and suggestion for contextualising serious, but rare, parsing harm. BMC Musculoskelet Disord. 2006;7:34.
14. Haidich AB. Meta-analysis in medical research. Hippokratia. 2010;14(Suppl 1):29-37.

---

**Word count:** ~3,500 (excluding tables and references)
**Figures:** Interactive dashboard available as supplementary material
**Supplementary data:** Full risk register (6,229 meta-analyses), erosion curves, domain-level data
