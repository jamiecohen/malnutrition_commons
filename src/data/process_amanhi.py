"""
Process raw AMANHI (WHO Alliance for Maternal and Newborn Health Improvement)
data into clean analysis-ready CSV files.

AMANHI is a multi-country observational cohort (Pakistan, Bangladesh, Tanzania)
studying the maternal gut microbiome and neonatal outcomes.

Raw sources (all in data/raw/AMANHI/):
  - Bioanalytes_AMANHIP.xlsx                           — serum biomarkers (Pakistan, n=1,937)
  - qPCR Data/WHO_AMANHI_Neonates_PCR_Metadata.xlsx    — B. infantis/longum + outcomes (3 countries)
  - AMANHI_Pak_Maternal_TAC ... (Raw CTs).xlsx         — maternal enteropathogens, raw Ct (Pakistan)
  - Metadata/WHO_AMANHI_Maternal_Metadata.xlsx         — maternal metadata (3 countries)

Key assumptions / decisions:
  - B. infantis / B. longum positive: Ct < 35 (applied directly to raw Ct values, not the
    pre-classified POS/NEG field). Note: B. infantis is taxonomically B. longum subsp. infantis —
    the assays detect different gene targets for each subspecies.
  - TAC pathogen detection: Ct < 35 = detected; Ct = 40 = undetected (instrument sentinel).
    Applied to the Raw CTs file for full reproducibility.
  - CRP elevated threshold: > 19 mg/L (1.9 mg/dL) — appropriate for pregnant women
    (WHO 2020 BRINDA framework for pregnancy; normal range <15-19 mg/L).
  - Iron deficiency: ferritin < 30 ng/mL at delivery — WHO recommends this higher threshold
    near delivery as ferritin is an acute-phase protein that is falsely elevated when CRP
    is elevated. True iron deficiency rate is likely underestimated even at this threshold.
  - Maternal TAC stool: collected POSTPARTUM (mean 29 days after delivery, range 1-245 days),
    not during pregnancy. The `days_postpartum` column reflects this.
  - Bioanalytes: 1,937 unique mothers — a broader Pakistan pregnancy cohort than the 266
    qPCR neonates. Biomarkers were collected at delivery (assumed; timing variable not available).
  - Sentinel z-scores (|z| > 6 per WHO guidance) set to NaN. Using |z| > 10 as our threshold
    to catch only clear coding errors (99, -99) while preserving biologically plausible outliers.
  - SGA uses the pre-computed SGA_10_NEW column (assumed INTERGROWTH-21st 10th percentile).
    Sex is not available in the neonatal PCR metadata, and birth length is not recorded,
    preventing independent recomputation of weight+length-for-GA z-scores.
  - Pathogen organism collapsing: multiple gene targets per organism (e.g. 7 Campylobacter
    targets) are collapsed to "any detection" at the organism level for summary analyses.

Outputs (data/processed/amanhi/):
  1. amanhi_neonatal.csv       — 1 row per neonate: B.inf/B.long status + outcomes + growth
  2. amanhi_bioanalytes.csv    — 1 row per mother (Pakistan): CRP, ferritin, etc.
  3. amanhi_maternal_tac.csv   — long format: mother × pathogen (Pakistan, 107 mothers)

Usage:
    python src/data/process_amanhi.py          # skip if outputs exist
    python src/data/process_amanhi.py --force  # overwrite existing outputs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "AMANHI"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "amanhi"

OUTPUT_FILES = [
    "amanhi_neonatal.csv",
    "amanhi_bioanalytes.csv",
    "amanhi_maternal_tac.csv",
]

# qPCR positive threshold — consistent with MUMTA and TAC standard
QPCR_CT_POSITIVE_THRESHOLD = 35

# ── CRP and ferritin thresholds (pregnancy / delivery) ───────────────────────
# CRP: WHO/BRINDA 2020 — >19 mg/L (1.9 mg/dL) for pregnant women
# Ferritin: WHO recommends <30 ng/mL at delivery (higher than the standard <15
# because ferritin is an acute-phase reactant falsely elevated by inflammation).
CRP_ELEVATED_MG_DL = 1.9        # = 19 mg/L
FERRITIN_DEFICIENT_NG_ML = 30   # at delivery (first trimester threshold is 15)

# ── Clinical pathogen columns in the TAC Raw CTs file ────────────────────────
# Gene-target → organism display label.
# Excluded: phhv_1/phhv_2 (process controls), hs99999901_s1 (fecal marker),
#   mpha / ctx_m_1_2_9 / ctx_m_8_25 (AMR markers), ms2_1/ms2_2 (phage QC),
#   shegyra83l/shegyra83s/sheparc80i/sheparc80s (Shigella subtype markers —
#   rolled into shigella_eiec / shigella_flexneri / shigella_sonnei),
#   plus binary outcome/metadata columns.
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
    # Bacteria – Campylobacter (7 gene targets → collapsed below)
    "campy16s":                   "Campylobacter (16S)",
    "campy23s2075a":              "Campylobacter (23S-A)",
    "campy23s2075g":              "Campylobacter (23S-G)",
    "campylobacter_coli":         "Campylobacter coli",
    "campylobacter_jejuni":       "Campylobacter jejuni",
    "campylobacter_jejuni_coli":  "C. jejuni/coli",
    "campylobacter_pan":          "Campylobacter (pan)",
    # Bacteria – diarrhoeagenic E. coli (collapsed below)
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
    # Bacteria – Shigella (4 species targets → collapsed below)
    "shigella_clade_1":           "Shigella clade 1",
    "shigella_eiec":              "Shigella/EIEC",
    "shigella_flexneri_6":        "S. flexneri 6",
    "shigella_flexneri_non6":     "S. flexneri (non-6)",
    "shigella_sonnei":            "S. sonnei",
    # Bacteria – other
    "aeromonas":                  "Aeromonas",
    "c_difficile":                "C. difficile",
    "h_pylori":                   "H. pylori",
    "m_tuberculosis":             "M. tuberculosis",
    "plesiomonas":                "Plesiomonas",
    "salmonella":                 "Salmonella",
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

# Organism-level collapse groups: any positive gene target = organism detected
COLLAPSE_GROUPS = {
    "Any Campylobacter": [
        "campy16s", "campy23s2075a", "campy23s2075g",
        "campylobacter_coli", "campylobacter_jejuni",
        "campylobacter_jejuni_coli", "campylobacter_pan",
    ],
    "Any EAEC": ["eaec_aaic", "eaec_aata"],
    "Any EPEC": ["epec_bfpa", "epec_eae"],
    "Any ETEC": [
        "etec_cfa_i", "etec_cs1", "etec_cs2", "etec_cs3",
        "etec_cs5", "etec_cs6", "etec_lt", "etec_sth", "etec_stp",
    ],
    "Any STEC": ["stec_stx1", "stec_stx2"],
    "Any Shigella": [
        "shigella_clade_1", "shigella_eiec",
        "shigella_flexneri_6", "shigella_flexneri_non6", "shigella_sonnei",
    ],
}

VIRUS_PATHOGENS = {
    "adenovirus_40_41", "astrovirus", "norovirus_gi", "norovirus_gii",
    "rotavirus", "sapovirus_i_ii_iv", "sapovirus_v", "sars_cov_2",
}
PARASITE_PATHOGENS = {
    "ancylostoma", "ascaris", "cryptosporidium", "cyclospora",
    "e_bieneusi", "e_histolytica", "e_histolytica_h", "e_intestinalis",
    "giardia", "isospora", "necator", "strongyloides", "trichuris",
}


def coerce_numeric(series):
    """Coerce a series to numeric, stripping below-detection-limit prefixes."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none", "n/a", "na", "."):
            return np.nan
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
      - B. infantis and B. longum detection status + Ct values (Ct < 35 = positive)
      - Birth outcomes (weight, GA, preterm, SGA, LBW)
      - Growth z-scores at birth and 6 months (pre-computed in dataset)
      - Maternal/demographic covariates

    Limitations: no infant sex or birth length in this file, so INTERGROWTH-21st
    weight+length-for-GA z-scores cannot be independently recomputed. The pre-computed
    SGA_10_NEW column is used for SGA (assumed INTERGROWTH-21st 10th percentile).
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

    # ── B. infantis / B. longum: classify from Ct values ─────────────────────
    # B. infantis is taxonomically B. longum subsp. infantis — these assays
    # target different gene regions to distinguish the subspecies.
    # Positive = Ct < QPCR_CT_POSITIVE_THRESHOLD; no result column = not tested.
    df.rename(columns={"Ct Value": "binfantis_ct", "Ct Value.1": "blongum_ct"}, inplace=True)
    df["binfantis_ct"] = pd.to_numeric(df["binfantis_ct"], errors="coerce")
    df["blongum_ct"] = pd.to_numeric(df["blongum_ct"], errors="coerce")

    for label, ct_col, result_col in [
        ("binfantis", "binfantis_ct", "B.Inf."),
        ("blongum", "blongum_ct", "B.Long."),
    ]:
        tested_mask = df[result_col].notna() | df[ct_col].notna()
        df[f"{label}_positive"] = np.where(
            ~tested_mask, np.nan,
            np.where(df[ct_col].notna() & (df[ct_col] < QPCR_CT_POSITIVE_THRESHOLD),
                     True, False),
        ).astype(float)

        # Verify vs pre-classified field
        _orig = df[result_col].map({"POS": True, "NEG": False})
        _new = df[f"{label}_positive"].map({1.0: True, 0.0: False})
        _diff = (_orig != _new) & _orig.notna() & _new.notna()
        if _diff.sum() > 0:
            print(f"  ⚠ {label}: {_diff.sum()} reclassified vs original POS/NEG field")
        else:
            print(f"  ✓ {label}: Ct-based classification matches original POS/NEG")

    # ── Birth outcomes ────────────────────────────────────────────────────────
    df["birth_weight_g"] = pd.to_numeric(df["BIRTH_WEIGHT"], errors="coerce")
    df.loc[df["birth_weight_g"] < 0, "birth_weight_g"] = np.nan  # -88 = missing
    df["ga_days"] = pd.to_numeric(df["GAGEBRTH_NEW"], errors="coerce")
    df["ga_weeks"] = df["ga_days"] / 7
    df["preterm"] = df["PTB_NEW"].map({1: True, 0: False})
    # SGA from pre-computed column (assumed INTERGROWTH-21st 10th percentile)
    df["sga"] = df["SGA_10_NEW"].map({1: True, 0: False})
    df["lbw"] = df["birth_weight_g"] < 2500

    df["delivery_place"] = df["DELIVERY_PLACE"].map({1: "home", 2: "facility"})
    df["delivery_mode"] = df["DEL_MODE"].map({1: "vaginal", 2: "cesarean"})

    # ── Growth z-scores ───────────────────────────────────────────────────────
    # Pre-computed in dataset; clean sentinel values (99, -99 etc).
    # Threshold: |z| > 10 catches clear coding errors without removing
    # biologically plausible (but extreme) values up to ±6 SD.
    for col in ["haz1", "waz1", "whz1", "haz6", "waz6", "whz6"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].abs() > 10, col] = np.nan

    # ── Maternal / demographic ────────────────────────────────────────────────
    df["maternal_age"] = pd.to_numeric(df["PW_AGE"], errors="coerce")
    df["gravidity"] = pd.to_numeric(df["GRAVIDITY"], errors="coerce")
    df["parity"] = pd.to_numeric(df["PARITY"], errors="coerce")
    df.loc[df["parity"] < 0, "parity"] = np.nan  # -88 missing code
    df["wealth_index"] = pd.to_numeric(df["WEALTH_INDEX"], errors="coerce")

    out = df[[
        "site", "PARTICIPANT_ID", "infant_vial_ID",
        "binfantis_positive", "binfantis_ct",
        "blongum_positive", "blongum_ct",
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
# 2. Bioanalytes (Pakistan, n=1,937 unique mothers at delivery)
# ─────────────────────────────────────────────────────────────────────────────

def build_bioanalytes(force=False):
    """Build the bioanalytes file from the Pakistan-specific Excel.

    One row per mother (n=1,937 — a broader Pakistan pregnancy cohort than the
    266 qPCR neonates). Biomarkers assumed collected at delivery.

    CRP threshold: >19 mg/L (1.9 mg/dL) — WHO/BRINDA 2020 for pregnant women.
    Ferritin threshold: <30 ng/mL — appropriate at delivery where inflammation
    is common and falsely elevates ferritin. The standard <15 ng/mL threshold
    is for non-pregnant / early pregnancy.
    """
    outpath = OUTPUT_DIR / "amanhi_bioanalytes.csv"
    if outpath.exists() and not force:
        print(f"  ✓ {outpath.name} exists, skipping")
        return pd.read_csv(outpath)

    bio_file = RAW_DIR / "Bioanalytes_AMANHIP.xlsx"
    df = pd.read_excel(bio_file, sheet_name="Sheet1")
    print(f"  Read {len(df)} bioanalyte records ({df['Participant_id'].nunique()} unique mothers)")

    for col in ["CRP", "FER", "TSH"]:
        df[col] = coerce_numeric(df[col])

    out = df.rename(columns={
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
    })

    # ── Derived classification columns ────────────────────────────────────────
    # CRP: WHO/BRINDA 2020 threshold for pregnant women = 19 mg/L = 1.9 mg/dL
    out["crp_mg_l"] = out["crp_mg_dl"] * 10  # convert for display
    out["crp_elevated"] = out["crp_mg_dl"] > CRP_ELEVATED_MG_DL

    # Iron deficiency: ferritin < 30 ng/mL at delivery (WHO at-delivery threshold)
    out["iron_deficient"] = out["ferritin_ng_ml"] < FERRITIN_DEFICIENT_NG_ML

    # Flag: CRP elevated AND ferritin in deficient range — highest concern
    # (inflammation confirmed AND iron stores low even with acute-phase boost)
    out["iron_deficient_with_inflammation"] = (
        out["crp_elevated"] & out["iron_deficient"]
    )

    # Flag: CRP elevated AND ferritin >= threshold — may be falsely replete
    # (inflammation could be masking iron deficiency)
    out["ferritin_may_be_falsely_normal"] = (
        out["crp_elevated"] & ~out["iron_deficient"]
    )

    print(
        f"  CRP elevated (>{CRP_ELEVATED_MG_DL:.1f} mg/dL): "
        f"{out['crp_elevated'].mean():.1%}"
    )
    print(
        f"  Iron deficient (<{FERRITIN_DEFICIENT_NG_ML} ng/mL at delivery): "
        f"{out['iron_deficient'].mean():.1%}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(outpath, index=False)
    print(f"  → Wrote {outpath.name}: {out.shape[0]} rows × {out.shape[1]} cols")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Maternal TAC pathogens (Pakistan, 107 mothers, postpartum stool)
# ─────────────────────────────────────────────────────────────────────────────

def build_maternal_tac(force=False):
    """Build long-format maternal TAC pathogen detection file.

    Uses the Raw CTs file and applies Ct < 35 threshold for consistency
    across cohorts. Ct = 40 is the instrument-assigned sentinel for undetected.

    IMPORTANT: Stool was collected POSTPARTUM (mean 29 days after birth,
    range 1–245 days), not during pregnancy. The `days_postpartum` column
    reflects this.

    Also produces organism-level collapsed detection (e.g. "Any Campylobacter")
    by grouping multiple gene targets per organism.
    """
    outpath = OUTPUT_DIR / "amanhi_maternal_tac.csv"
    if outpath.exists() and not force:
        print(f"  ✓ {outpath.name} exists, skipping")
        return pd.read_csv(outpath)

    # Use Raw CTs file — apply our own Ct < 35 threshold
    tac_file_raw = list(RAW_DIR.glob("AMANHI_Pak_Maternal_TAC*Raw CTs*"))
    if not tac_file_raw:
        # Fall back to Cat CTs if raw not available
        tac_file_raw = list(RAW_DIR.glob("AMANHI_Pak_Maternal_TAC*Cat CTs*"))
        print("  ⚠ Raw CTs file not found, falling back to Cat CTs")
        use_raw_ct = False
    else:
        use_raw_ct = True
    tac_file = tac_file_raw[0]

    df = pd.read_excel(tac_file)
    # Drop extra header rows if present (raw file may have 108 rows)
    if len(df) == 108 and df.iloc[0]["whowid"] != df.iloc[1]["whowid"]:
        # Check if first row is a duplicate/header artefact
        pass  # both rows appear to be real data
    df = df[df["whowid"].notna()].copy()
    print(f"  Read {len(df)} maternal TAC records from {tac_file.name}")
    print(f"  Stool timing: mean {df['delta_dob_date_ofsample_collecti'].mean():.0f} days postpartum "
          f"(range {df['delta_dob_date_ofsample_collecti'].min():.0f}–"
          f"{df['delta_dob_date_ofsample_collecti'].max():.0f})")

    # Metadata columns
    meta_cols = [
        "whowid", "site", "bmi", "bmi_cat", "matmuac", "muac_cat",
        "ga_outcome", "preterm_new", "sga_bin", "gender",
        "wt0", "len0", "haz0", "waz0", "whz0",
        "wt12", "len12", "haz12", "waz12", "whz12",
        "wt18", "len18", "haz18", "waz18", "whz18",
        "wealth_quantile", "place_of_delivery", "motherage",
        "delta_dob_date_ofsample_collecti",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    # ── Pathogen detection from Raw Ct values ─────────────────────────────────
    available_pathogens = {
        col: label for col, label in TAC_PATHOGEN_COLS.items()
        if col in df.columns
    }
    print(f"  Found {len(available_pathogens)} pathogen target columns")

    # Build wide detection matrix (Ct < 35 = 1, else 0)
    detect_wide = pd.DataFrame(index=df.index)
    if use_raw_ct:
        for col in available_pathogens:
            ct_vals = pd.to_numeric(df[col], errors="coerce")
            detect_wide[col] = (ct_vals < QPCR_CT_POSITIVE_THRESHOLD).astype(int)
    else:
        for col in available_pathogens:
            detect_wide[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── Organism-level collapse ───────────────────────────────────────────────
    for organism, gene_targets in COLLAPSE_GROUPS.items():
        present = [g for g in gene_targets if g in detect_wide.columns]
        if present:
            detect_wide[f"__collapsed__{organism}"] = (
                detect_wide[present].max(axis=1)
            )

    # ── Melt to long format ───────────────────────────────────────────────────
    pathogen_cols = list(available_pathogens.keys())
    collapsed_cols = [c for c in detect_wide.columns if c.startswith("__collapsed__")]

    meta_df = df[meta_cols].copy()
    meta_df["days_postpartum"] = pd.to_numeric(
        df["delta_dob_date_ofsample_collecti"], errors="coerce"
    )

    long_parts = []

    # Gene-target level
    long_gt = pd.concat([meta_df, detect_wide[pathogen_cols]], axis=1).melt(
        id_vars=list(meta_df.columns),
        value_vars=pathogen_cols,
        var_name="pathogen_raw",
        value_name="detected",
    )
    long_gt["pathogen"] = long_gt["pathogen_raw"].map(available_pathogens)
    long_gt["level"] = "gene_target"
    long_parts.append(long_gt)

    # Organism-collapsed level
    collapsed_labels = {c: c.replace("__collapsed__", "") for c in collapsed_cols}
    long_org = pd.concat([meta_df, detect_wide[collapsed_cols]], axis=1).melt(
        id_vars=list(meta_df.columns),
        value_vars=collapsed_cols,
        var_name="pathogen_raw",
        value_name="detected",
    )
    long_org["pathogen"] = long_org["pathogen_raw"].map(collapsed_labels)
    long_org["level"] = "organism"
    long_parts.append(long_org)

    long = pd.concat(long_parts, ignore_index=True)

    # Pathogen category
    long["pathogen_category"] = "bacteria"
    long.loc[long["pathogen_raw"].isin(VIRUS_PATHOGENS), "pathogen_category"] = "virus"
    long.loc[long["pathogen_raw"].isin(PARASITE_PATHOGENS), "pathogen_category"] = "parasite"
    long.loc[long["level"] == "organism", "pathogen_category"] = "bacteria"  # all collapsed are bacteria

    # Per-mother total burden (gene-target level)
    burden = (
        long[long["level"] == "gene_target"]
        .groupby("whowid")["detected"]
        .sum()
        .rename("total_pathogens")
    )
    long = long.merge(burden, on="whowid", how="left")

    # Rename timing column
    long.rename(columns={"delta_dob_date_ofsample_collecti": "days_postpartum"}, inplace=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(outpath, index=False)
    print(f"  → Wrote {outpath.name}: {long.shape[0]} rows × {long.shape[1]} cols")

    # Summary
    gene_level = long[long["level"] == "gene_target"]
    n_mothers = gene_level["whowid"].nunique()
    top5 = (
        gene_level[gene_level["detected"] == 1]
        .groupby("pathogen").size()
        .sort_values(ascending=False)
        .head(5)
    )
    print(f"  Top 5 detected (gene-target level, n={n_mothers} mothers):")
    for p, n in top5.items():
        print(f"    {p}: {n/n_mothers:.0%}")
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
