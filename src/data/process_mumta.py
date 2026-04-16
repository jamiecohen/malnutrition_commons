"""
Process raw MUMTA (Pakistan) data into 6 clean analysis-ready CSV files.

Raw source: data/raw/MUMTA/Pakistan/MumtaPW_outcomes.xlsx (sheet "Data")
Outputs:    data/processed/mumta/
  1. mumta_cohort_summary.csv      — 1 row per participant
  2. mumta_infant_growth.csv       — long format (study_id x month)
  3. mumta_maternal_anemia.csv     — long format (study_id x timepoint)
  4. mumta_binfantis.csv           — long format qPCR results
  5. mumta_gut_inflammation.csv    — long format MPO/LCN
  6. mumta_microbiome_top_genera.csv — top 20 genera relative abundance

Usage:
    python src/data/process_mumta.py          # skip if outputs exist
    python src/data/process_mumta.py --force  # overwrite existing outputs
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "MUMTA" / "Pakistan"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mumta"

OUTCOMES_FILE = RAW_DIR / "MumtaPW_outcomes.xlsx"
ABUNDANCE_FILE = RAW_DIR / "MUMTA-PW_relative_abundance.tsv"
METADATA_FILE = RAW_DIR / "MUMTA-PW_Metadata.xlsx"

ARM_LABELS = {
    "A": "Control",
    "B": "Maamta",
    "C": "Maamta+Azithromycin",
    "D": "Maamta+Choline+Nicotinamide",
}

OUTPUT_FILES = [
    "mumta_cohort_summary.csv",
    "mumta_infant_growth.csv",
    "mumta_maternal_anemia.csv",
    "mumta_binfantis.csv",
    "mumta_gut_inflammation.csv",
    "mumta_microbiome_top_genera.csv",
]

# Follow-up suffixes: F0 = birth, F1-F8 = follow-ups
FOLLOWUP_SUFFIXES = ["6a_F0"] + [f"7c_F{i}" for i in range(1, 9)]
FOLLOWUP_MONTHS = list(range(9))  # 0 through 8

# Gut inflammation / qPCR timepoints with specimen classification
BIOMARKER_TIMEPOINTS = {
    "19wks":  ("19wk",  "maternal"),
    "32wks":  ("32wk",  "maternal"),
    "1-2MS":  ("1-2mo", "maternal"),
    "1-2IS":  ("1-2mo", "infant"),
    "3-4MS":  ("3-4mo", "maternal"),
    "3-4IS":  ("3-4mo", "infant"),
    "5-6MS":  ("5-6mo", "maternal"),
    "5-6IS":  ("5-6mo", "infant"),
    "12MS":   ("12mo",  "maternal"),
    "12IS":   ("12mo",  "infant"),
}

# Chronological order for timepoints
TIMEPOINT_ORDER = ["19wk", "32wk", "1-2mo", "3-4mo", "5-6mo", "12mo"]


def coerce_numeric(series):
    """Coerce a series to numeric, handling below-detection-limit strings like '<<4.00'."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none", "n/a", "na", "."):
            return np.nan
        # Handle below-detection-limit patterns like <<4.00, <4.00
        m = re.match(r'^[<]+\s*([\d.]+)$', s)
        if m:
            return float(m.group(1))
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan
    return series.apply(_parse)


def load_outcomes():
    """Load the main outcomes Excel file."""
    print(f"Reading {OUTCOMES_FILE} ...")
    df = pd.read_excel(OUTCOMES_FILE, sheet_name="Data")
    print(f"  Raw shape: {df.shape}")
    # Normalise ARM column: extract single letter (A/B/C/D) from full label
    df["ARM"] = df["ARM"].str.extract(r"Arm\s+([A-D])", expand=False).fillna(df["ARM"])
    return df


# ---------------------------------------------------------------------------
# 1. Cohort summary
# ---------------------------------------------------------------------------
def build_cohort_summary(df):
    """One row per participant with demographics, birth outcomes, and anemia flags."""
    out = pd.DataFrame()
    out["study_id"] = df["Study ID"]
    out["arm"] = df["ARM"]
    out["arm_label"] = df["ARM"].map(ARM_LABELS)
    out["age"] = coerce_numeric(df["Age of the woman in years (3b)"])
    out["birth_outcome"] = df["Birth outcome type (6a)"]
    # Handle curly apostrophe in column name
    _gender_col = [c for c in df.columns if "gender" in c.lower() and "6b" in c]
    out["child_gender"] = df[_gender_col[0]] if _gender_col else np.nan
    out["delivery_mode"] = df["Mode of delivery (6b)"]
    out["gestational_age_weeks"] = coerce_numeric(df["Gestational age at the time of outcome (6a)"])
    out["birth_weight_g"] = coerce_numeric(df["weight_gram_6a_F0"])

    # Binary flags
    out["lbw"] = out["birth_weight_g"] < 2500
    out["preterm"] = out["gestational_age_weeks"] < 37
    out["stunted_at_birth"] = coerce_numeric(df["zlen_6a_F0"]) < -2
    out["wasted_at_birth"] = coerce_numeric(df["zwfl_6a_F0"]) < -2

    # Micronutrients
    out["hb_19wk"] = coerce_numeric(df["TP_1_PW_Hb_g_dL"])
    out["ferritin_19wk"] = coerce_numeric(df["TP_1_Ferritin_ng_mL"])
    out["hb_32wk"] = coerce_numeric(df["32wksPW_Hb_g_dL"])
    out["ferritin_32wk"] = coerce_numeric(df["32wksFerritin_ng_mL"])

    # Anemia / iron deficiency flags
    out["anaemic_19wk"] = out["hb_19wk"] < 11
    out["anaemic_32wk"] = out["hb_32wk"] < 11
    out["iron_deficient_19wk"] = out["ferritin_19wk"] < 15
    out["iron_deficient_32wk"] = out["ferritin_32wk"] < 15

    return out


# ---------------------------------------------------------------------------
# 2. Infant growth (long format)
# ---------------------------------------------------------------------------
def build_infant_growth(df):
    """Long format: study_id x month with anthropometrics and z-scores."""
    records = []
    for suffix, month in zip(FOLLOWUP_SUFFIXES, FOLLOWUP_MONTHS):
        sub = pd.DataFrame()
        sub["study_id"] = df["Study ID"]
        sub["arm"] = df["ARM"]
        sub["month"] = month
        sub["weight_g"] = coerce_numeric(df.get(f"weight_gram_{suffix}", pd.Series(dtype=float)))
        sub["length_cm"] = coerce_numeric(df.get(f"length_cm_{suffix}", pd.Series(dtype=float)))
        sub["muac_cm"] = coerce_numeric(df.get(f"muac_cm_{suffix}", pd.Series(dtype=float)))
        sub["head_circ_cm"] = coerce_numeric(df.get(f"head_circum_cm_{suffix}", pd.Series(dtype=float)))
        sub["laz"] = coerce_numeric(df.get(f"zlen_{suffix}", pd.Series(dtype=float)))
        sub["waz"] = coerce_numeric(df.get(f"zwei_{suffix}", pd.Series(dtype=float)))
        sub["wlz"] = coerce_numeric(df.get(f"zwfl_{suffix}", pd.Series(dtype=float)))

        # Binary flags
        sub["stunted"] = sub["laz"] < -2
        sub["underweight"] = sub["waz"] < -2
        sub["wasted"] = sub["wlz"] < -2

        records.append(sub)

    long = pd.concat(records, ignore_index=True)

    # Keep only rows where at least one measurement is non-null
    meas_cols = ["weight_g", "length_cm", "muac_cm", "head_circ_cm", "laz", "waz", "wlz"]
    long = long.dropna(subset=meas_cols, how="all").reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# 3. Maternal anemia (long format)
# ---------------------------------------------------------------------------
def build_maternal_anemia(df):
    """Long format: study_id x timepoint (19wk, 32wk) with Hb, ferritin, vitamin D."""
    records = []

    # 19-week timepoint
    tp19 = pd.DataFrame()
    tp19["study_id"] = df["Study ID"]
    tp19["arm"] = df["ARM"]
    tp19["timepoint"] = "19wk"
    tp19["hemoglobin"] = coerce_numeric(df["TP_1_PW_Hb_g_dL"])
    tp19["ferritin"] = coerce_numeric(df["TP_1_Ferritin_ng_mL"])
    tp19["vitamin_d"] = coerce_numeric(df["TP_1_Vitamin_D_ng_mL"])
    records.append(tp19)

    # 32-week timepoint
    tp32 = pd.DataFrame()
    tp32["study_id"] = df["Study ID"]
    tp32["arm"] = df["ARM"]
    tp32["timepoint"] = "32wk"
    tp32["hemoglobin"] = coerce_numeric(df["32wksPW_Hb_g_dL"])
    tp32["ferritin"] = coerce_numeric(df["32wksFerritin_ng_mL"])
    tp32["vitamin_d"] = coerce_numeric(df["32wksVitamin_D_ng_mL"])
    records.append(tp32)

    long = pd.concat(records, ignore_index=True)
    long["anaemic"] = long["hemoglobin"] < 11
    long["iron_deficient"] = long["ferritin"] < 15
    return long


# ---------------------------------------------------------------------------
# 4. B. infantis / B. longum qPCR (long format)
# ---------------------------------------------------------------------------
def build_binfantis(df):
    """Long format qPCR results for B. infantis and B. longum."""
    records = []
    for raw_tp, (clean_tp, specimen) in BIOMARKER_TIMEPOINTS.items():
        prefix = f"{raw_tp} BInfBLong"
        binf_result_col = f"{prefix} - BInf Result"
        binf_ct_col = f"{prefix} - BInf ct value"
        blong_result_col = f"{prefix} - BLong Result"
        blong_ct_col = f"{prefix} - BLong ct value"

        # Skip if columns don't exist
        if binf_result_col not in df.columns:
            continue

        sub = pd.DataFrame()
        sub["study_id"] = df["Study ID"]
        sub["arm"] = df["ARM"]
        sub["timepoint"] = clean_tp
        sub["specimen_type"] = specimen
        sub["b_infantis_positive"] = df[binf_result_col].astype(str).str.strip().str.lower() == "positive"
        sub["b_infantis_ct"] = coerce_numeric(df.get(binf_ct_col, pd.Series(dtype=float)))
        sub["b_longum_positive"] = df[blong_result_col].astype(str).str.strip().str.lower() == "positive"
        sub["b_longum_ct"] = coerce_numeric(df.get(blong_ct_col, pd.Series(dtype=float)))
        records.append(sub)

    long = pd.concat(records, ignore_index=True)
    # Sort by chronological order
    long["timepoint"] = pd.Categorical(long["timepoint"], categories=TIMEPOINT_ORDER, ordered=True)
    long = long.sort_values(["study_id", "timepoint", "specimen_type"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# 5. Gut inflammation MPO/LCN (long format)
# ---------------------------------------------------------------------------
def build_gut_inflammation(df):
    """Long format MPO and LCN2 biomarkers."""
    records = []
    for raw_tp, (clean_tp, specimen) in BIOMARKER_TIMEPOINTS.items():
        mpo_col = f"{raw_tp} MPO"
        lcn_col = f"{raw_tp} LCN"

        if mpo_col not in df.columns and lcn_col not in df.columns:
            continue

        sub = pd.DataFrame()
        sub["study_id"] = df["Study ID"]
        sub["arm"] = df["ARM"]
        sub["timepoint"] = clean_tp
        sub["specimen_type"] = specimen
        sub["mpo"] = coerce_numeric(df.get(mpo_col, pd.Series(dtype=float)))
        sub["lcn2"] = coerce_numeric(df.get(lcn_col, pd.Series(dtype=float)))
        records.append(sub)

    long = pd.concat(records, ignore_index=True)
    long["timepoint"] = pd.Categorical(long["timepoint"], categories=TIMEPOINT_ORDER, ordered=True)
    long = long.sort_values(["study_id", "timepoint", "specimen_type"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# 6. Microbiome top genera
# ---------------------------------------------------------------------------
def build_microbiome_top_genera():
    """Top 20 genera by mean relative abundance from 16S data."""
    print(f"Reading {ABUNDANCE_FILE} ...")
    abund = pd.read_csv(ABUNDANCE_FILE, sep="\t", index_col=0, comment="#")
    # Drop the 'clade_name' header row if present
    if "clade_name" in abund.index:
        abund = abund.drop("clade_name")
    print(f"  Abundance shape: {abund.shape}")

    print(f"Reading {METADATA_FILE} ...")
    meta = pd.read_excel(METADATA_FILE)
    print(f"  Metadata shape: {meta.shape}")

    # Filter to genus-level rows: contains g__ but NOT s__
    genus_mask = abund.index.str.contains("g__", na=False) & ~abund.index.str.contains("s__", na=False)
    genus_df = abund.loc[genus_mask].copy()
    print(f"  Genus-level rows: {genus_df.shape[0]}")

    # Coerce all abundance values to numeric (TSV may parse as strings)
    genus_df = genus_df.apply(pd.to_numeric, errors="coerce")

    # Extract genus name from lineage
    def extract_genus(lineage):
        # MetaPhlAn uses | as taxonomy separator
        parts = str(lineage).split("|")
        for p in reversed(parts):
            p = p.strip()
            if p.startswith("g__"):
                return p.replace("g__", "")
        return lineage
    genus_df.index = genus_df.index.map(extract_genus)

    # If duplicate genera, sum them
    genus_df = genus_df.groupby(genus_df.index).sum()

    # Rank by mean relative abundance across all samples, keep top 20
    mean_abund = genus_df.mean(axis=1, numeric_only=True).sort_values(ascending=False)
    top20 = mean_abund.head(20).index.tolist()
    print(f"  Top 20 genera: {top20[:5]} ... (showing first 5)")

    # Transpose: samples as rows, genera as columns
    top_df = genus_df.loc[top20].T
    top_df.index.name = "sample_id"
    top_df = top_df.reset_index()

    # Merge with metadata to get study_id, timepoint, arm
    # Identify the sample ID column in metadata
    sample_col = None
    for candidate in ["stool_Ids", "stool_ids", "Stool_Ids", "sample_id", "SampleID"]:
        if candidate in meta.columns:
            sample_col = candidate
            break
    if sample_col is None:
        # Fallback: use first column
        sample_col = meta.columns[0]
        print(f"  Warning: using '{sample_col}' as sample ID column in metadata")

    meta_slim = meta[[sample_col, "study_id", "TimePoint", "ARM"]].copy()
    meta_slim.columns = ["sample_id", "study_id", "timepoint", "arm"]

    # Clean timepoint
    def clean_timepoint(tp):
        s = str(tp).strip().lower()
        if "19" in s:
            return "19wk"
        elif "32" in s:
            return "32wk"
        return s
    meta_slim["timepoint"] = meta_slim["timepoint"].apply(clean_timepoint)

    merged = top_df.merge(meta_slim, on="sample_id", how="inner")

    # Reorder columns: sample_id, study_id, timepoint, arm, then genera
    id_cols = ["sample_id", "study_id", "timepoint", "arm"]
    genus_cols = [c for c in merged.columns if c not in id_cols]
    merged = merged[id_cols + genus_cols]

    print(f"  Merged microbiome shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_summary(name, df):
    """Print shape and key stats for a dataframe."""
    print(f"\n{'='*60}")
    print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Process MUMTA raw data into clean CSVs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if outputs already exist
    if not args.force:
        existing = [f for f in OUTPUT_FILES if (OUTPUT_DIR / f).exists()]
        if len(existing) == len(OUTPUT_FILES):
            print("All output files already exist. Use --force to overwrite.")
            print("  " + "\n  ".join(str(OUTPUT_DIR / f) for f in existing))
            return
        elif existing:
            print(f"{len(existing)}/{len(OUTPUT_FILES)} outputs exist. Processing missing files...")

    # Load main data
    df = load_outcomes()

    # 1. Cohort summary
    print("\n--- Building cohort summary ---")
    cohort = build_cohort_summary(df)
    cohort.to_csv(OUTPUT_DIR / "mumta_cohort_summary.csv", index=False)
    print_summary("mumta_cohort_summary.csv", cohort)
    print(f"  ARM distribution:\n{cohort['arm'].value_counts().to_string()}")
    print(f"  LBW prevalence: {cohort['lbw'].mean():.1%} (n={cohort['lbw'].sum()})")
    print(f"  Preterm prevalence: {cohort['preterm'].mean():.1%} (n={cohort['preterm'].sum()})")
    print(f"  Stunted at birth: {cohort['stunted_at_birth'].mean():.1%}")
    print(f"  Anaemic at 19wk: {cohort['anaemic_19wk'].mean():.1%}")
    print(f"  Anaemic at 32wk: {cohort['anaemic_32wk'].mean():.1%}")
    print(f"  Iron deficient 19wk: {cohort['iron_deficient_19wk'].mean():.1%}")

    # 2. Infant growth
    print("\n--- Building infant growth ---")
    growth = build_infant_growth(df)
    growth.to_csv(OUTPUT_DIR / "mumta_infant_growth.csv", index=False)
    print_summary("mumta_infant_growth.csv", growth)
    print(f"  Months represented: {sorted(growth['month'].unique())}")
    print(f"  Rows per month:\n{growth['month'].value_counts().sort_index().to_string()}")

    # 3. Maternal anemia
    print("\n--- Building maternal anemia ---")
    anemia = build_maternal_anemia(df)
    anemia.to_csv(OUTPUT_DIR / "mumta_maternal_anemia.csv", index=False)
    print_summary("mumta_maternal_anemia.csv", anemia)
    for tp in ["19wk", "32wk"]:
        sub = anemia[anemia["timepoint"] == tp]
        hb_valid = sub["hemoglobin"].notna().sum()
        print(f"  {tp}: {hb_valid} Hb measurements, anaemic={sub['anaemic'].mean():.1%}")

    # 4. B. infantis
    print("\n--- Building B. infantis qPCR ---")
    binfantis = build_binfantis(df)
    binfantis.to_csv(OUTPUT_DIR / "mumta_binfantis.csv", index=False)
    print_summary("mumta_binfantis.csv", binfantis)
    print(f"  Timepoints: {binfantis['timepoint'].unique().tolist()}")
    print(f"  B. infantis positive rate: {binfantis['b_infantis_positive'].mean():.1%}")
    print(f"  B. longum positive rate: {binfantis['b_longum_positive'].mean():.1%}")

    # 5. Gut inflammation
    print("\n--- Building gut inflammation ---")
    gut = build_gut_inflammation(df)
    gut.to_csv(OUTPUT_DIR / "mumta_gut_inflammation.csv", index=False)
    print_summary("mumta_gut_inflammation.csv", gut)
    print(f"  Timepoints: {gut['timepoint'].unique().tolist()}")
    mpo_valid = gut["mpo"].notna().sum()
    lcn_valid = gut["lcn2"].notna().sum()
    print(f"  MPO non-null: {mpo_valid}, LCN2 non-null: {lcn_valid}")

    # 6. Microbiome top genera
    print("\n--- Building microbiome top genera ---")
    microbiome = build_microbiome_top_genera()
    microbiome.to_csv(OUTPUT_DIR / "mumta_microbiome_top_genera.csv", index=False)
    print_summary("mumta_microbiome_top_genera.csv", microbiome)

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL 6 FILES WRITTEN SUCCESSFULLY")
    print("=" * 60)
    for f in OUTPUT_FILES:
        path = OUTPUT_DIR / f
        size_kb = path.stat().st_size / 1024
        print(f"  {f}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
