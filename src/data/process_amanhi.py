"""
Process raw AMANHI (WHO Alliance for Maternal and Newborn Health Improvement)
data into clean analysis-ready CSV files.

AMANHI is a multi-country observational cohort (Pakistan, Bangladesh, Tanzania)
studying the maternal gut microbiome and neonatal outcomes.

Raw sources (all in data/raw/AMANHI/):
  - Bioanalytes_AMANHIP.xlsx                          — serum biomarkers (Pakistan)
  - qPCR Data/WHO_AMANHI_Neonates_PCR_Metadata.xlsx   — B. infantis/longum + outcomes (3 countries)
  - qPCR Data/AMANHI-WHO_B.Infantis_&_Blongum_...xlsx — raw PCR results (3 countries)
  - AMANHI_Pak_Maternal_TAC ... (Cat CTs).xlsx         — maternal enteropathogens (Pakistan)
  - Metadata/WHO_AMANHI_Maternal_Metadata.xlsx         — maternal metadata (3 countries)

Outputs (data/processed/amanhi/):
  1. amanhi_neonatal.csv       — 1 row per neonate: B.inf/B.long status + outcomes + growth
  2. amanhi_bioanalytes.csv    — 1 row per mother (Pakistan): CRP, ferritin, etc.
  3. amanhi_maternal_tac.csv   — long format: mother × pathogen (Pakistan, 108 mothers)

Usage:
    python src/data/process_amanhi.py          # skip if outputs exist
    python src/data/process_amanhi.py --force  # overwrite existing outputs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "AMANHI"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "amanhi"

# Output file list for existence check
OUTPUT_FILES = [
    "amanhi_neonatal.csv",
    "amanhi_bioanalytes.csv",
    "amanhi_maternal_tac.csv",
]

# ── Clinical pathogen columns in the TAC Cat CTs file ────────────────────────
# These are the actual enteropathogens (excluding metadata, QC markers, and
# outcome columns that happen to be binary 0/1).
# We exclude: fev_or_psdbd, fev_or_psdad, sga_bin, aga_bin, nutri_prg,
#   preterm_new, age0, phhv_1, phhv_2, hs99999901_s1, mpha, ctx_m_1_2_9,
#   ctx_m_8_25, ms2_1, ms2_2 (QC/AMR markers, not clinical pathogens)
TAC_PATHOGEN_COLS = {
    # Viruses
    "adenovirus_40_41":           "Adenovirus 40/41",
    "astrovirus":                 "Astrovirus",
    "norovirus_gi":               "Norovirus GI",
    "norovirus_gii":              "Norovirus GII",
    "rotavirus":                  "Rotavirus",
    "sapovirus_i_ii_iv":          "Sapovirus I/II/IV",
    "sapovirus_v":                "Sapovirus V",
    "sars_cov_2":                 "SARS-CoV-2",
    # Bacteria – Campylobacter
    "campy16s":                   "Campylobacter (16S)",
    "campy23s2075a":              "Campylobacter (23S-A)",
    "campy23s2075g":              "Campylobacter (23S-G)",
    "campylobacter_coli":         "Campylobacter coli",
    "campylobacter_jejuni":       "Campylobacter jejuni",
    "campylobacter_jejuni_coli":  "C. jejuni/coli",
    "campylobacter_pan":          "Campylobacter (pan)",
    # Bacteria – diarrhoeal E. coli
    "eaec_aaic":                  "EAEC (aaiC)",
    "eaec_aata":                  "EAEC (aatA)",
    "epec_bfpa":                  "EPEC (bfpA)",
    "epec_eae":                   "EPEC (eae)",
    "etec_cfa_i":                 "ETEC (CFA/I)",
    "etec_cs1":                   "ETEC (CS1)",
    "etec_cs2":                   "ETEC (CS2)",
    "etec_cs3":                   "ETEC (CS3)",
    "etec_cs5":                   "ETEC (CS5)",
    "etec_cs6":                   "ETEC (CS6)",
    "etec_lt":                    "ETEC (LT)",
    "etec_sth":                   "ETEC (STh)",
    "etec_stp":                   "ETEC (STp)",
    "stec_stx1":                  "STEC (Stx1)",
    "stec_stx2":                  "STEC (Stx2)",
    # Bacteria – other
    "aeromonas":                  "Aeromonas",
    "c_difficile":                "C. difficile",
    "h_pylori":                   "H. pylori",
    "m_tuberculosis":             "M. tuberculosis",
    "plesiomonas":                "Plesiomonas",
    "salmonella":                 "Salmonella",
    "shigella_clade_1":           "Shigella clade 1",
    "shigella_eiec":              "Shigella/EIEC",
    "shigella_flexneri_6":        "S. flexneri 6",
    "shigella_flexneri_non6":     "S. flexneri (non-6)",
    "shigella_sonnei":            "S. sonnei",
    "v_cholerae":                 "V. cholerae",
    # Parasites
    "ancylostoma":                "Ancylostoma (hookworm)",
    "ascaris":                    "Ascaris",
    "cryptosporidium":            "Cryptosporidium",
    "cyclospora":                 "Cyclospora",
    "e_bieneusi":                 "E. bieneusi",
    "e_histolytica":              "E. histolytica",
    "e_histolytica_h":            "E. histolytica (H)",
    "e_intestinalis":             "E. intestinalis",
    "giardia":                    "Giardia",
    "isospora":                   "Isospora",
    "necator":                    "Necator",
    "strongyloides":              "Strongyloides",
    "trichuris":                  "Trichuris",
}

# Shigella marker columns (used in TAC for gene-level detection, not clinical pathogens)
SHIGELLA_MARKER_COLS = ["shegyra83l", "shegyra83s", "sheparc80i", "sheparc80s"]


def coerce_numeric(series):
    """Coerce a series to numeric, handling string quirks."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none", "n/a", "na", "."):
            return np.nan
        # Handle below-detection-limit values like "<<4.00" or "<0.01"
        s = s.lstrip("<>")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(_parse).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Neonatal dataset: B. infantis / B. longum + outcomes + growth
# ─────────────────────────────────────────────────────────────────────────────

def build_neonatal(force=False):
    """Build the neonatal analysis file from the PCR metadata.

    One row per neonate across all three countries with:
      - B. infantis and B. longum detection status + Ct values
      - Birth outcomes (weight, GA, preterm, SGA)
      - Growth z-scores at birth and 6 months
      - Maternal/demographic covariates
    """
    outpath = OUTPUT_DIR / "amanhi_neonatal.csv"
    if outpath.exists() and not force:
        print(f"  ✓ {outpath.name} exists, skipping")
        return pd.read_csv(outpath)

    pcr_file = RAW_DIR / "qPCR Data" / "WHO_AMANHI_Neonates_PCR_Metadata.xlsx"
    df = pd.read_excel(pcr_file, sheet_name="ALL_DATA")
    print(f"  Read {len(df)} neonatal records from PCR metadata")

    # Standardise site codes
    site_map = {"Pakistan": "PAK", "Bangladesh": "BGD", "Tanzania": "TZA"}
    df["site"] = df["SITE_CODE"].map(site_map)

    # Clean B. infantis / B. longum — classify from Ct values directly
    # (not the pre-classified POS/NEG field) for consistency and to catch errors.
    # Ct < 35 = positive; Ct >= 35 or NaN-with-result = negative; no result = not tested.
    CT_THRESHOLD = 35
    df.rename(columns={"Ct Value": "binfantis_ct", "Ct Value.1": "blongum_ct"}, inplace=True)
    df["binfantis_ct"] = pd.to_numeric(df["binfantis_ct"], errors="coerce")
    df["blongum_ct"] = pd.to_numeric(df["blongum_ct"], errors="coerce")

    # B. infantis: positive if Ct < threshold
    _bi_tested = df["B.Inf."].notna() | df["binfantis_ct"].notna()
    df["binfantis_positive"] = np.where(
        ~_bi_tested, np.nan,
        np.where(df["binfantis_ct"].notna() & (df["binfantis_ct"] < CT_THRESHOLD), True, False),
    )
    df["binfantis_positive"] = df["binfantis_positive"].astype(float)  # NaN-safe

    # B. longum: positive if Ct < threshold
    _bl_tested = df["B.Long."].notna() | df["blongum_ct"].notna()
    df["blongum_positive"] = np.where(
        ~_bl_tested, np.nan,
        np.where(df["blongum_ct"].notna() & (df["blongum_ct"] < CT_THRESHOLD), True, False),
    )
    df["blongum_positive"] = df["blongum_positive"].astype(float)

    # Log reclassification vs original
    _orig_bi = df["B.Inf."].map({"POS": True, "NEG": False})
    _new_bi = df["binfantis_positive"].map({1.0: True, 0.0: False})
    _diff = (_orig_bi != _new_bi) & _orig_bi.notna() & _new_bi.notna()
    if _diff.sum() > 0:
        print(f"  ⚠ B. infantis: {_diff.sum()} reclassified vs original POS/NEG field")
    else:
        print(f"  ✓ B. infantis: Ct-based classification matches original POS/NEG")

    # Clean birth outcomes
    df["birth_weight_g"] = pd.to_numeric(df["BIRTH_WEIGHT"], errors="coerce")
    df.loc[df["birth_weight_g"] < 0, "birth_weight_g"] = np.nan  # -88 = missing code
    df["ga_days"] = pd.to_numeric(df["GAGEBRTH_NEW"], errors="coerce")
    df["ga_weeks"] = df["ga_days"] / 7
    df["preterm"] = df["PTB_NEW"].map({1: True, 0: False})
    df["sga"] = df["SGA_10_NEW"].map({1: True, 0: False})
    df["lbw"] = df["birth_weight_g"] < 2500

    # Delivery
    df["delivery_place"] = df["DELIVERY_PLACE"].map({1: "home", 2: "facility"})
    df["delivery_mode"] = df["DEL_MODE"].map({1: "vaginal", 2: "cesarean"})

    # Growth z-scores
    for col in ["haz1", "waz1", "whz1", "haz6", "waz6", "whz6"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Replace sentinel values (99, -99)
        df.loc[df[col].abs() > 10, col] = np.nan

    # Maternal / demographic
    df["maternal_age"] = pd.to_numeric(df["PW_AGE"], errors="coerce")
    df["gravidity"] = pd.to_numeric(df["GRAVIDITY"], errors="coerce")
    df["parity"] = pd.to_numeric(df["PARITY"], errors="coerce")
    df["wealth_index"] = pd.to_numeric(df["WEALTH_INDEX"], errors="coerce")

    # Select and rename
    out = df[[
        "site", "PARTICIPANT_ID", "infant_vial_ID",
        "binfantis_positive", "binfantis_ct", "blongum_positive", "blongum_ct",
        "birth_weight_g", "ga_days", "ga_weeks", "preterm", "sga", "lbw",
        "delivery_place", "delivery_mode",
        "haz1", "waz1", "whz1", "haz6", "waz6", "whz6",
        "maternal_age", "gravidity", "parity", "wealth_index",
    ]].copy()
    out.rename(columns={"PARTICIPANT_ID": "participant_id"}, inplace=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(outpath, index=False)
    print(f"  → Wrote {outpath.name}: {out.shape[0]} rows × {out.shape[1]} cols")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bioanalytes (Pakistan only): CRP, ferritin, albumin, etc.
# ─────────────────────────────────────────────────────────────────────────────

def build_bioanalytes(force=False):
    """Build the bioanalytes file from the Pakistan-specific Excel.

    One row per mother with serum markers: CRP, ferritin, albumin, calcium,
    BUN, SCR, AST, TSH, syphilis, PlGF, sFlt-1, PAPP-A.
    """
    outpath = OUTPUT_DIR / "amanhi_bioanalytes.csv"
    if outpath.exists() and not force:
        print(f"  ✓ {outpath.name} exists, skipping")
        return pd.read_csv(outpath)

    bio_file = RAW_DIR / "Bioanalytes_AMANHIP.xlsx"
    df = pd.read_excel(bio_file, sheet_name="Sheet1")
    print(f"  Read {len(df)} bioanalyte records")

    # Coerce string numeric columns
    for col in ["CRP", "FER", "TSH"]:
        df[col] = coerce_numeric(df[col])

    # Rename for clarity
    rename_map = {
        "Participant_id": "participant_id",
        "baby_id": "baby_id",
        "whowid": "whowid",
        "CRP": "crp_mg_dl",
        "FER": "ferritin_ng_ml",
        "CA": "calcium_mg_dl",
        "BUN": "bun",
        "SCR": "creatinine",
        "AST": "ast",
        "ALB": "albumin_g_dl",
        "TSH": "tsh",
        "SRPR": "syphilis_rpr",
        "SFLT": "sflt1_pg_ml",
        "PLGF": "plgf_pg_ml",
        "PAPPA": "pappa_iu_ml",
    }
    out = df.rename(columns=rename_map)

    # Add derived columns
    # Elevated CRP (common threshold: > 5 mg/L = 0.5 mg/dL for pregnancy)
    out["crp_elevated"] = out["crp_mg_dl"] > 0.5
    # Iron deficiency proxy: ferritin < 15 ng/mL (WHO definition)
    out["iron_deficient"] = out["ferritin_ng_ml"] < 15
    # Adjusted ferritin: if CRP elevated, ferritin may be falsely high (acute phase)
    # Flag for interpretation
    out["ferritin_adjusted_flag"] = out["crp_elevated"] & (out["ferritin_ng_ml"] >= 15)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(outpath, index=False)
    print(f"  → Wrote {outpath.name}: {out.shape[0]} rows × {out.shape[1]} cols")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Maternal TAC pathogens (Pakistan only, 108 mothers)
# ─────────────────────────────────────────────────────────────────────────────

def build_maternal_tac(force=False):
    """Build long-format maternal TAC pathogen detection file.

    Reads the Cat CTs (binary 0/1) file and pivots to long format:
    one row per mother × pathogen with detection status.
    Also preserves key metadata (anthropometry, outcomes) for cross-analysis.
    """
    outpath = OUTPUT_DIR / "amanhi_maternal_tac.csv"
    if outpath.exists() and not force:
        print(f"  ✓ {outpath.name} exists, skipping")
        return pd.read_csv(outpath)

    tac_file = list(RAW_DIR.glob("AMANHI_Pak_Maternal_TAC*Cat CTs*"))[0]
    df = pd.read_excel(tac_file)
    print(f"  Read {len(df)} maternal TAC records from {tac_file.name}")

    # Identify which pathogen columns are actually in the file
    available_pathogens = {
        col: label for col, label in TAC_PATHOGEN_COLS.items()
        if col in df.columns
    }
    missing = set(TAC_PATHOGEN_COLS) - set(available_pathogens)
    if missing:
        print(f"  ⚠ {len(missing)} expected pathogen columns not found: {sorted(missing)[:5]}...")

    print(f"  Found {len(available_pathogens)} pathogen columns")

    # Metadata columns to keep
    meta_cols = [
        "whowid", "site", "bmi", "bmi_cat", "matmuac", "muac_cat",
        "ga_outcome", "preterm_new", "sga_bin", "wt0", "len0",
        "haz0", "waz0", "whz0",
        "wt12", "len12", "haz12", "waz12", "whz12",
        "wt18", "len18", "haz18", "waz18", "whz18",
        "wealth_quantile", "place_of_delivery", "motherage",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    # Melt pathogen columns to long format
    pathogen_col_names = list(available_pathogens.keys())
    long = df[meta_cols + pathogen_col_names].melt(
        id_vars=meta_cols,
        value_vars=pathogen_col_names,
        var_name="pathogen_raw",
        value_name="detected",
    )
    long["pathogen"] = long["pathogen_raw"].map(available_pathogens)
    long["detected"] = long["detected"].astype(float).fillna(0).astype(int)

    # Add pathogen category
    virus_pathogens = {
        "adenovirus_40_41", "astrovirus", "norovirus_gi", "norovirus_gii",
        "rotavirus", "sapovirus_i_ii_iv", "sapovirus_v", "sars_cov_2",
    }
    parasite_pathogens = {
        "ancylostoma", "ascaris", "cryptosporidium", "cyclospora",
        "e_bieneusi", "e_histolytica", "e_histolytica_h", "e_intestinalis",
        "giardia", "isospora", "necator", "strongyloides", "trichuris",
    }
    long["pathogen_category"] = "bacteria"
    long.loc[long["pathogen_raw"].isin(virus_pathogens), "pathogen_category"] = "virus"
    long.loc[long["pathogen_raw"].isin(parasite_pathogens), "pathogen_category"] = "parasite"

    # Also compute per-mother total pathogen burden (wide summary)
    long["total_pathogens"] = long.groupby("whowid")["detected"].transform("sum")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(outpath, index=False)
    print(f"  → Wrote {outpath.name}: {long.shape[0]} rows × {long.shape[1]} cols")
    return long


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process AMANHI data")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    print("Processing AMANHI data...")
    print(f"  Raw dir:    {RAW_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    build_neonatal(force=args.force)
    print()
    build_bioanalytes(force=args.force)
    print()
    build_maternal_tac(force=args.force)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
