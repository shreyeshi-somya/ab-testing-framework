# Intelligent A/B Testing Framework — Project Plan

> **Stack:** Python · PySpark · SciPy · StatsModels · Anthropic API · Gradio  
> **Goal:** An end-to-end experimentation framework that automates hypothesis generation, statistical testing (frequentist + Bayesian), SRM detection, novelty effect checks, and executive narrative creation.

---

## Overview

| Phase | Name | Output |
|-------|------|--------|
| 1 | Data Foundation | 3 synthetic datasets (1M rows each), Parquet + CSV |
| 2 | Core Stats Engine | Pure Python stats module (power analysis, frequentist, Bayesian, SRM, novelty) |
| 3 | Claude Integration | AI layer for hypothesis generation, executive summaries, segmentation insights |
| 4 | Gradio UI | Interactive dashboard with visualizations and drill-downs |
| 5 | Polish & Packaging | README, architecture diagram, setup instructions |

---

## Phase 1 — Data Foundation

**Goal:** Statistically grounded synthetic datasets that power all downstream modules.

### Step 1.1 — Define Shared Schema

All three datasets share a common base schema:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | string | Unique user identifier |
| `variant` | string | `control` or `treatment` |
| `experiment_day` | int | Day of experiment (1–30) |
| `user_type` | string | `new` or `returning` |
| `country` | string | `US`, `UK`, `IN`, `DE`, `BR` |
| `device` | string | `mobile`, `desktop`, `tablet` |
| `platform` | string | `iOS`, `Android`, `Web` |

Each dataset adds outcome-specific columns on top of this base.

---

### Step 1.2 — Dataset 1: Conversion Rate (Clearly Significant + Novelty Effect)

**Scenario:** A new checkout UI is tested. Treatment shows a clearly significant lift in conversion (p < 0.01), but the lift is stronger in the first 10 days and fades — a classic novelty effect.

**Additional columns:**

| Column | Type | Description |
|--------|------|-------------|
| `converted` | int | Binary outcome: 1 = converted, 0 = did not |
| `conversion_value` | float | Revenue if converted, else 0 |

**Statistical parameters:**
- Control conversion rate: 10%
- Treatment conversion rate: 13% (days 1–10), 10.5% (days 11–30) — novelty effect
- Sample size: 1,000,000 rows (500K control / 500K treatment)
- Expected result: p < 0.01 overall; novelty detectable via day-split analysis

**Segmentation nuances to bake in:**
- Mobile users convert at 7% (control) vs 13.5% (treatment) — strong segment effect
- Returning users show weaker lift than new users
- IN and BR markets show no significant lift (noise)

---

### Step 1.3 — Dataset 2: Revenue / AOV (Borderline / Ambiguous)

**Scenario:** A personalised product recommendation widget is tested for impact on average order value (AOV). The result is ambiguous — statistically marginal.

**Additional columns:**

| Column | Type | Description |
|--------|------|-------------|
| `purchased` | int | Binary: 1 = made a purchase |
| `revenue` | float | Order value if purchased, else 0.0 |
| `num_items` | int | Items in order (0 if no purchase) |

**Statistical parameters:**
- Control AOV (among purchasers): $85, std dev $40
- Treatment AOV: $89 (+$4 lift), std dev $42
- Purchase rate: 35% control, 36% treatment
- Sample size: 1,000,000 rows
- Expected result: p ~ 0.06–0.09 (borderline, confidence interval crosses 0 at 95%)

**Segmentation nuances:**
- Desktop users show cleaner lift ($85 → $92)
- Mobile users show negligible effect ($85 → $85.50)
- US market drives most of the observed lift; other markets near-zero

---

### Step 1.4 — Dataset 3: Subscription Renewal (No Effect / Null Result)

**Scenario:** A redesigned renewal reminder email sequence is tested. There is genuinely no effect — this dataset teaches how to correctly interpret and communicate a null result.

**Additional columns:**

| Column | Type | Description |
|--------|------|-------------|
| `renewed` | int | Binary: 1 = renewed subscription |
| `days_to_renewal_decision` | int | Days until user renewed or churned (1–30) |
| `plan_type` | string | `monthly`, `annual` |
| `tenure_months` | int | How long user has been a subscriber |

**Statistical parameters:**
- Control renewal rate: 72%
- Treatment renewal rate: 72.3% (within noise margin)
- Sample size: 1,000,000 rows
- Expected result: p > 0.3, confidence interval clearly spans 0 in both directions

**Segmentation nuances:**
- Annual plan holders renew at 88% regardless of variant (both groups)
- Monthly churners churn at the same rate in both variants
- No segment shows a meaningful lift — clean null across all cuts

---

### Step 1.5 — PySpark Generation Script

**File:** `generate_datasets.py`

**Script responsibilities:**
- Accept a config dict per dataset (rates, effect sizes, segment distributions)
- Use `pyspark.sql` with a fixed random seed for reproducibility
- Apply segment-specific conversion rates using conditional logic / UDFs
- Add `experiment_day` and apply novelty effect decay on Dataset 1
- Write output to:
  - `data/parquet/dataset_1_conversion/`
  - `data/parquet/dataset_2_revenue/`
  - `data/parquet/dataset_3_renewal/`
  - `data/csv/dataset_1_sample_10k.csv` (10K row samples for quick inspection)

**Key libraries:** `pyspark`, `numpy`, `random`

**Validation checks to include:**
- Print actual vs expected conversion rates per variant
- Print variant split (should be ~50/50)
- Print row count per dataset
- Print segment distribution sanity check

---

## Phase 2 — Core Stats Engine

**Goal:** A standalone Python module that takes a dataset and returns all statistical outputs. No UI dependencies — fully testable in isolation.

**File structure:**
```
stats/
├── __init__.py
├── power_analysis.py
├── frequentist.py
├── bayesian.py
├── srm_detection.py
├── novelty_effect.py
└── segmentation.py
```

---

### Step 2.1 — Power Analysis (`power_analysis.py`)

Pre-experiment sample size calculator. Answers: *"How many users do I need?"*

**Inputs:** baseline rate, minimum detectable effect (MDE), alpha, power  
**Outputs:** required sample size per variant, estimated experiment duration  
**Library:** `statsmodels.stats.power`

**Functions:**
- `calc_sample_size_proportion(baseline, mde, alpha, power)` — for conversion rate experiments
- `calc_sample_size_ttest(baseline_mean, baseline_std, mde, alpha, power)` — for revenue experiments
- `estimate_experiment_duration(sample_size, daily_traffic)` — given traffic volume

---

### Step 2.2 — Frequentist Testing (`frequentist.py`)

**Inputs:** control and treatment arrays (or summary stats)  
**Outputs:** test statistic, p-value, confidence interval, effect size, recommendation

**Functions:**
- `z_test_proportions(control_conversions, control_n, treatment_conversions, treatment_n)` — for conversion rate
- `welch_ttest(control_values, treatment_values)` — for revenue / AOV (handles unequal variance)
- `log_rank_test(control_events, control_times, treatment_events, treatment_times)` — for renewal / survival
- `compute_relative_lift(control_mean, treatment_mean)` — % lift with confidence interval
- `interpret_result(p_value, alpha=0.05)` → `"significant"` / `"not significant"` / `"borderline"`

---

### Step 2.3 — Bayesian Analysis (`bayesian.py`)

**Inputs:** same as frequentist  
**Outputs:** posterior distributions, credible intervals, P(treatment > control), expected loss

**Functions:**
- `beta_binomial_posterior(control_conversions, control_n, treatment_conversions, treatment_n)` — returns Beta distribution params
- `sample_posterior(alpha, beta, n_samples=100000)` — draws samples for simulation
- `probability_of_improvement(control_posterior, treatment_posterior)` — P(B > A)
- `expected_loss(control_posterior, treatment_posterior)` — cost of wrong decision
- `credible_interval(posterior_samples, credibility=0.95)` — HDI interval

---

### Step 2.4 — SRM Detection (`srm_detection.py`)

**Goal:** Detect if the control/treatment split deviates significantly from the intended ratio.

**Functions:**
- `detect_srm(control_n, treatment_n, expected_ratio=0.5)` — chi-square test on assignment counts
- `srm_severity(p_value)` → `"none"` / `"mild"` / `"severe"` — actionable classification
- `srm_diagnosis_hints()` — returns list of common SRM causes (logging bug, bot traffic, cache issue, etc.)

---

### Step 2.5 — Novelty Effect Detection (`novelty_effect.py`)

**Goal:** Detect if the treatment lift is driven by early-period novelty that fades over time.

**Functions:**
- `split_early_late(df, experiment_day_col, split_day=10)` — returns early/late DataFrames
- `compare_lift_over_time(df, variant_col, outcome_col, experiment_day_col)` — returns daily lift series
- `detect_novelty(early_lift, late_lift, threshold=0.3)` — flags novelty if early lift > late lift by threshold
- `plot_cumulative_metric(df)` — returns data for time-series visualization

---

### Step 2.6 — Segmentation Engine (`segmentation.py`)

**Goal:** Automatically slice results by segment columns and surface heterogeneous treatment effects.

**Functions:**
- `run_segmented_analysis(df, segment_cols, variant_col, outcome_col, test_fn)` — runs the chosen test on every segment value
- `build_segment_summary(results)` — returns a DataFrame of segment → lift, p-value, sample size
- `flag_significant_segments(summary, alpha=0.05)` — highlights segments that differ from overall result
- `detect_heterogeneous_effects(summary)` — flags if treatment helps one segment but hurts another

---

## Phase 3 — Claude Integration

**Goal:** Use the Anthropic API to transform raw statistical output into analyst-grade narratives and actionable recommendations.

**File:** `claude_integration.py`

---

### Step 3.1 — Hypothesis Generation

Called at experiment setup time, before results are known.

**Input:** dataset metadata (columns, experiment type, business context string)  
**Output:** 3–5 candidate hypotheses ranked by testability  
**Prompt strategy:** Provide column names, outcome variable, and segment columns. Ask Claude to generate hypotheses about *what* might drive a treatment effect and *which segments* to watch.

---

### Step 3.2 — Executive Summary Generation

Called after statistical analysis is complete.

**Input:** structured results dict (p-value, lift, CI, Bayesian probability, SRM flag, novelty flag, segment summary)  
**Output:** 3-paragraph executive summary:
1. What was tested and for how long
2. What the data shows (plain English, no jargon)
3. Recommended action with rationale

**Prompt strategy:** Pass full results as JSON. Instruct Claude to write for a non-technical VP audience. Specify: no p-values in the output, use % lift and confidence language instead.

---

### Step 3.3 — Segmentation Insight Narration

**Input:** segment summary DataFrame  
**Output:** 2–4 bullet points surfacing the most surprising or actionable segment findings  
**Example output:** *"Mobile users in India showed no lift, while desktop users in the US drove the majority of the observed conversion improvement. Consider a mobile-specific redesign before a full rollout."*

---

### Step 3.4 — Ship / Iterate / Kill Recommendation

**Input:** full results dict + business context (e.g., implementation cost, strategic priority)  
**Output:** One of three recommendations with a 2-sentence rationale:
- **Ship** — statistically significant positive result, no data quality issues
- **Iterate** — borderline result or segment heterogeneity detected; specific next test suggested
- **Kill** — null result or negative segments outweigh positive ones

---

## Phase 4 — Gradio UI

**Goal:** An interactive, demo-ready dashboard that showcases all framework capabilities.

**File:** `app.py`

---

### Step 4.1 — App Layout

```
┌─────────────────────────────────────────────────────┐
│  Sidebar: Dataset selector + Experiment config       │
├─────────────┬───────────────────────────────────────┤
│  Tab 1      │  Results Overview                      │
│  Tab 2      │  Statistical Tests (Freq + Bayesian)   │
│  Tab 3      │  Visualizations                        │
│  Tab 4      │  Segmentation Drill-Down               │
│  Tab 5      │  Executive Summary (Claude)            │
└─────────────┴───────────────────────────────────────┘
```

---

### Step 4.2 — Sidebar: Configuration Panel

- Dataset selector: Dropdown (Conversion Rate / Revenue AOV / Subscription Renewal)
- Significance level: Slider (α = 0.01, 0.05, 0.10)
- Analysis method: Radio (Frequentist / Bayesian / Both)
- Segment to drill into: Dropdown (user_type / country / device)
- Run Analysis button

---

### Step 4.3 — Tab 1: Results Overview

- KPI cards: Control rate, Treatment rate, Observed lift %, p-value, Decision badge (Ship / Iterate / Kill)
- SRM warning banner (if SRM detected)
- Novelty effect warning banner (if novelty detected)
- Data quality summary: row counts, variant split, missing value report

---

### Step 4.4 — Tab 2: Statistical Tests

**Frequentist panel:**
- Test used (auto-selected based on experiment type)
- Test statistic, p-value, effect size (Cohen's d or relative risk)
- 95% confidence interval visualised as a horizontal bar
- Power analysis retrospective: was the experiment adequately powered?

**Bayesian panel:**
- P(treatment > control) displayed as a gauge
- Posterior distribution plot (Beta distributions for conversion; KDE for revenue)
- 95% credible interval
- Expected loss metric

---

### Step 4.5 — Tab 3: Visualizations

- **Cumulative metric over time** — line chart of daily conversion/revenue per variant (reveals novelty effect)
- **Lift distribution** — histogram of bootstrapped lift estimates with CI bands
- **Confidence interval plot** — forest plot style, overall + per segment
- **Posterior distribution** — shaded area chart (Bayesian only)

---

### Step 4.6 — Tab 4: Segmentation Drill-Down

- Segment selector (user_type / country / device / platform)
- Segment results table: segment value → control rate, treatment rate, lift %, p-value, sample size
- Colour-coded significance indicators
- Bar chart: lift per segment value with error bars
- Claude-generated segment insight (auto-generated on tab load)

---

### Step 4.7 — Tab 5: Executive Summary

- Run Summary button triggers Claude API call
- Displays three-paragraph narrative (non-technical)
- Ship / Iterate / Kill recommendation badge with rationale
- Copy to clipboard button
- Export as PDF button (stretch goal)

---

## Phase 5 — Polish & Portfolio Packaging

### Step 5.1 — Project Structure

```
ab-testing-framework/
├── README.md
├── requirements.txt
├── generate_datasets.py          # Phase 1
├── stats/                        # Phase 2
│   ├── power_analysis.py
│   ├── frequentist.py
│   ├── bayesian.py
│   ├── srm_detection.py
│   ├── novelty_effect.py
│   └── segmentation.py
├── claude_integration.py         # Phase 3
├── app.py                        # Phase 4
├── data/
│   ├── parquet/
│   └── csv/
├── tests/
│   ├── test_frequentist.py
│   ├── test_bayesian.py
│   └── test_srm.py
└── assets/
    └── architecture_diagram.png
```

---

### Step 5.2 — README Sections

1. Project overview and motivation
2. Architecture diagram
3. Dataset descriptions (with statistical scenarios)
4. Setup instructions (`pip install -r requirements.txt`)
5. How to run the data generator
6. How to run the Gradio app
7. Module-level documentation for the stats engine
8. Sample screenshots of the UI

---

### Step 5.3 — requirements.txt

```
pyspark>=3.5.0
scipy>=1.11.0
statsmodels>=0.14.0
numpy>=1.24.0
pandas>=2.0.0
anthropic>=0.20.0
gradio>=4.0.0
matplotlib>=3.7.0
plotly>=5.15.0
pyarrow>=12.0.0
```

---

### Step 5.4 — Tests

Write unit tests for the stats engine using `pytest`:
- `test_frequentist.py` — test z-test, t-test, log-rank with known inputs and expected outputs
- `test_bayesian.py` — test posterior calculations, P(B>A) with synthetic data
- `test_srm.py` — test SRM detection with a known imbalanced split

---

### Step 5.5 — Stretch Goals (Post-v1)

| Feature | Description |
|---------|-------------|
| SRM as 4th dataset | Add a 4th synthetic dataset with a baked-in 55/45 split to demo SRM detection end-to-end |
| PDF export | Export executive summary tab as a formatted PDF report |
| Multi-metric testing | Support simultaneous guardrail metrics (e.g., test conversion rate but also check session duration) |
| Sequential testing | Add alpha-spending functions for early stopping decisions |
| dbt integration | Connect to real experiment tables via a dbt model adapter |

---

## Build Order Summary

```
Phase 1 → generate_datasets.py (PySpark, 1M rows x 3 datasets)
Phase 2 → stats/ module (power_analysis, frequentist, bayesian, srm, novelty, segmentation)
Phase 3 → claude_integration.py (hypothesis gen, exec summary, segment insights, recommendation)
Phase 4 → app.py (Gradio dashboard, 5 tabs)
Phase 5 → README, tests, packaging
```

> Start with Phase 1. Validate the data before writing a single line of stats code.
