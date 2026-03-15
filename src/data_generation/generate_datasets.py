"""
Phase 1 — Data Foundation
Generates 3 synthetic A/B testing datasets (1M rows each) using pandas + numpy.

Datasets:
  1. Conversion Rate — significant lift + novelty effect
  2. Revenue / AOV — borderline / ambiguous result
  3. Subscription Renewal — null result

Usage:
    python3 -m src.data_generation.generate_datasets
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
ROWS_PER_DATASET = 1_000_000
CSV_SAMPLE_SIZE = 10_000

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
CSV_DIR = os.path.join(DATA_DIR, "csv")

# Segment distributions
COUNTRIES = ["US", "UK", "IN", "DE", "BR"]
COUNTRY_WEIGHTS = [0.35, 0.20, 0.20, 0.15, 0.10]

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.50, 0.35, 0.15]

PLATFORMS = ["iOS", "Android", "Web"]
PLATFORM_WEIGHTS = [0.30, 0.35, 0.35]

USER_TYPES = ["new", "returning"]
USER_TYPE_WEIGHTS = [0.45, 0.55]


def _build_base_columns(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the shared base columns for all datasets."""
    n = ROWS_PER_DATASET
    return pd.DataFrame({
        "user_id": [f"user_{i:07d}" for i in range(n)],
        "variant": rng.choice(["control", "treatment"], size=n),
        "experiment_day": rng.integers(1, 31, size=n),
        "user_type": rng.choice(USER_TYPES, size=n, p=USER_TYPE_WEIGHTS),
        "country": rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS),
        "device": rng.choice(DEVICES, size=n, p=DEVICE_WEIGHTS),
        "platform": rng.choice(PLATFORMS, size=n, p=PLATFORM_WEIGHTS),
    })


# ---------------------------------------------------------------------------
# Dataset 1: Conversion Rate (Significant + Novelty Effect)
# ---------------------------------------------------------------------------

def _conversion_rates_vectorized(df: pd.DataFrame) -> np.ndarray:
    """Compute conversion probability for each row (vectorized)."""
    probs = np.full(len(df), 0.10)

    is_control = df["variant"].values == "control"
    is_treatment = ~is_control
    is_mobile = df["device"].values == "mobile"
    is_tablet = df["device"].values == "tablet"
    is_desktop = ~is_mobile & ~is_tablet
    is_early = df["experiment_day"].values <= 10
    is_late = ~is_early
    is_returning = df["user_type"].values == "returning"
    is_in_br = np.isin(df["country"].values, ["IN", "BR"])

    # Control base rates by device
    probs[is_control & is_mobile] = 0.07
    probs[is_control & is_tablet] = 0.09
    probs[is_control & is_desktop] = 0.10

    # Treatment rates — early (novelty)
    probs[is_treatment & is_early & is_mobile] = 0.135
    probs[is_treatment & is_early & is_tablet] = 0.12
    probs[is_treatment & is_early & is_desktop] = 0.13

    # Treatment rates — late (novelty faded)
    probs[is_treatment & is_late & is_mobile] = 0.08
    probs[is_treatment & is_late & is_tablet] = 0.095
    probs[is_treatment & is_late & is_desktop] = 0.105

    # Returning users show weaker lift (treatment only)
    probs[is_treatment & is_returning] *= 0.92

    # IN and BR markets — pull treatment back toward control
    treatment_in_br = is_treatment & is_in_br
    if treatment_in_br.any():
        control_rates = np.where(is_mobile, 0.07, 0.10)
        probs[treatment_in_br] = (
            control_rates[treatment_in_br]
            + (probs[treatment_in_br] - control_rates[treatment_in_br]) * 0.15
        )

    return probs


def generate_dataset_1(rng: np.random.Generator) -> None:
    """Conversion Rate dataset — significant lift with novelty effect."""
    print("\n=== Dataset 1: Conversion Rate ===")
    df = _build_base_columns(rng)

    conv_probs = _conversion_rates_vectorized(df)
    df["converted"] = (rng.random(ROWS_PER_DATASET) < conv_probs).astype(np.int8)

    # Revenue for converters: log-normal distribution
    lognormal_values = rng.lognormal(mean=3.5, sigma=0.8, size=ROWS_PER_DATASET)
    df["conversion_value"] = np.where(df["converted"] == 1, lognormal_values.round(2), 0.0)

    _write_and_validate(df, "dataset_1_conversion", "converted")


# ---------------------------------------------------------------------------
# Dataset 2: Revenue / AOV (Borderline)
# ---------------------------------------------------------------------------

def generate_dataset_2(rng: np.random.Generator) -> None:
    """Revenue / AOV dataset — borderline result."""
    print("\n=== Dataset 2: Revenue / AOV ===")
    df = _build_base_columns(rng)

    # Purchase decision
    purchase_probs = np.where(df["variant"].values == "treatment", 0.36, 0.35)
    df["purchased"] = (rng.random(ROWS_PER_DATASET) < purchase_probs).astype(np.int8)

    # Revenue for purchasers — vectorized
    is_purchased = df["purchased"].values == 1
    is_treatment = df["variant"].values == "treatment"
    is_desktop = df["device"].values == "desktop"
    is_us = df["country"].values == "US"

    means = np.full(ROWS_PER_DATASET, 85.0)
    stds = np.full(ROWS_PER_DATASET, 40.0)

    # Treatment desktop: $92 +/- $42
    means[is_treatment & is_desktop] = 92.0
    stds[is_treatment & is_desktop] = 42.0
    # Treatment US (non-desktop): $90 +/- $41
    means[is_treatment & is_us & ~is_desktop] = 90.0
    stds[is_treatment & is_us & ~is_desktop] = 41.0
    # Treatment other: $85.50 +/- $40.50
    means[is_treatment & ~is_desktop & ~is_us] = 85.5
    stds[is_treatment & ~is_desktop & ~is_us] = 40.5

    normal_draws = rng.standard_normal(ROWS_PER_DATASET)
    revenue = np.maximum(1.0, means + stds * normal_draws)
    df["revenue"] = np.where(is_purchased, revenue.round(2), 0.0)

    # Number of items
    items = rng.poisson(3, size=ROWS_PER_DATASET) + 1
    df["num_items"] = np.where(is_purchased, items, 0).astype(np.int32)

    _write_and_validate(df, "dataset_2_revenue", "purchased")


# ---------------------------------------------------------------------------
# Dataset 3: Subscription Renewal (Null Result)
# ---------------------------------------------------------------------------

def generate_dataset_3(rng: np.random.Generator) -> None:
    """Subscription Renewal dataset — null result."""
    print("\n=== Dataset 3: Subscription Renewal ===")
    df = _build_base_columns(rng)

    # Plan type: 40% annual, 60% monthly
    df["plan_type"] = rng.choice(["annual", "monthly"], size=ROWS_PER_DATASET, p=[0.40, 0.60])

    # Tenure: exponential-ish distribution, 1-60 months
    df["tenure_months"] = np.clip(
        rng.exponential(scale=18, size=ROWS_PER_DATASET).astype(int), 1, 60
    )

    # Renewal rates
    is_annual = df["plan_type"].values == "annual"
    is_treatment = df["variant"].values == "treatment"

    renewal_probs = np.full(ROWS_PER_DATASET, 0.62)
    renewal_probs[is_annual] = 0.88
    renewal_probs[~is_annual & is_treatment] = 0.623  # within noise

    df["renewed"] = (rng.random(ROWS_PER_DATASET) < renewal_probs).astype(np.int8)

    # Days to renewal decision
    df["days_to_renewal_decision"] = np.where(
        df["renewed"].values == 1,
        np.clip(rng.exponential(scale=8, size=ROWS_PER_DATASET).astype(int), 1, 30),
        np.clip(rng.exponential(scale=18, size=ROWS_PER_DATASET).astype(int), 1, 30),
    ).astype(np.int32)

    # Reorder columns to match plan
    df = df[["user_id", "variant", "experiment_day", "user_type", "country",
             "device", "platform", "renewed", "days_to_renewal_decision",
             "plan_type", "tenure_months"]]

    _write_and_validate(df, "dataset_3_renewal", "renewed")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _write_and_validate(df: pd.DataFrame, name: str, outcome_col: str) -> None:
    """Write dataset to parquet and CSV sample, then print validation stats."""
    parquet_path = os.path.join(PARQUET_DIR, name)
    csv_path = os.path.join(CSV_DIR, f"{name}_sample_10k.csv")

    os.makedirs(parquet_path, exist_ok=True)

    # Write parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, os.path.join(parquet_path, "data.parquet"))
    print(f"  Written parquet -> {parquet_path}")

    # Write CSV sample
    sample = df.sample(n=CSV_SAMPLE_SIZE, random_state=SEED)
    sample.to_csv(csv_path, index=False)
    print(f"  Written CSV sample -> {csv_path}")

    # --- Validation ---
    total = len(df)
    print(f"\n  Row count: {total:,}")

    # Variant split
    variant_counts = df["variant"].value_counts()
    for variant, count in variant_counts.items():
        pct = count / total * 100
        print(f"  {variant}: {count:,} ({pct:.1f}%)")

    # Outcome rate by variant
    rates = df.groupby("variant")[outcome_col].mean()
    for variant, rate in rates.items():
        print(f"  {variant} {outcome_col} rate: {rate:.4f}")

    # Segment distribution check
    for seg_col in ["user_type", "country", "device"]:
        print(f"\n  {seg_col} distribution:")
        seg_counts = df[seg_col].value_counts().sort_index()
        for val, count in seg_counts.items():
            pct = count / total * 100
            print(f"    {val}: {pct:.1f}%")


def main() -> None:
    rng = np.random.default_rng(SEED)

    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    generate_dataset_1(rng)
    generate_dataset_2(rng)
    generate_dataset_3(rng)

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
