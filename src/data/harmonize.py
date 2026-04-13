"""
Harmonize all raw data sources into a single country-level indicator table.

Output: data/processed/commons_indicators.csv
Columns:
  iso3, country_name, region, year,
  anaemia_children_pct, anaemia_pregnant_women_pct, anaemia_women_repro_pct,
  stunting_pct, wasting_pct,
  tb_incidence_per100k,
  undernourishment_pct  (where available)

Logic:
- Standardize to ISO3 country codes
- Filter to country-level records only (exclude aggregates)
- For most-recent-year views: take the most recent value per country within 2015–2023
- For time-series views: keep all years 2000–2023
"""

import pandas as pd
from pathlib import Path

RAW = Path(__file__).parents[2] / "data" / "raw"
PROCESSED = Path(__file__).parents[2] / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ISO3 codes that are aggregates / regions — exclude from country-level analysis
AGGREGATE_ISO3 = {
    "AFR", "AMR", "SEAR", "EUR", "EMR", "WPR",  # WHO regions
    "WLD", "LMY", "UMC", "LMC", "LIC", "HIC",   # World Bank income groups
    "EAP", "ECA", "LAC", "MNA", "NAC", "SAS", "SSA",  # WB regions
    "AFE", "AFW", "ARB", "CEB", "CSS", "EAR", "EAS",
    "ECS", "FCS", "HPC", "IBD", "IBT", "IDA", "IDB",
    "IDX", "LAC", "LCN", "LDC", "MIC", "OED", "OSS",
    "PRE", "PSS", "PST", "SAS", "SSF", "SST", "TEA",
    "TEC", "TLA", "TMN", "TSA", "TSS", "UMC",
}


# ── WHO GHO helpers ──────────────────────────────────────────────────────────

def load_gho(filename: str, col_name: str, sex_filter: str | None = "SEX_BTSX") -> pd.DataFrame:
    """Load a WHO GHO CSV and return a clean (iso3, year, col_name) frame."""
    path = RAW / "who_gho" / filename
    df = pd.read_csv(path)
    # Filter to both-sexes if a sex dimension exists
    if sex_filter and "Dim1" in df.columns:
        df = df[df["Dim1"] == sex_filter]
    df = df.rename(columns={"SpatialDim": "iso3", "TimeDim": "year", "NumericValue": col_name})
    df = df[["iso3", "year", col_name]].dropna(subset=[col_name])
    df["year"] = df["year"].astype(int)
    df = df[~df["iso3"].isin(AGGREGATE_ISO3)]
    return df


def load_gho_any_sex(filename: str, col_name: str) -> pd.DataFrame:
    """For indicators with no sex dimension, load without filtering."""
    return load_gho(filename, col_name, sex_filter=None)


# ── Load each source ─────────────────────────────────────────────────────────

def load_all_sources() -> dict[str, pd.DataFrame]:
    sources = {}

    # Anaemia in children <5
    sources["anaemia_children"] = load_gho("anaemia_children.csv", "anaemia_children_pct")

    # Anaemia in pregnant women — dimension is SEVERITY_TOTAL, not SEX
    preg = pd.read_csv(RAW / "who_gho" / "anaemia_pregnant_women.csv")
    preg = preg[preg["Dim1"] == "SEVERITY_TOTAL"]
    preg = preg.rename(columns={"SpatialDim": "iso3", "TimeDim": "year", "NumericValue": "anaemia_pregnant_women_pct"})
    preg = preg[~preg["iso3"].isin(AGGREGATE_ISO3)]
    preg = preg[["iso3", "year", "anaemia_pregnant_women_pct"]].dropna(subset=["anaemia_pregnant_women_pct"])
    preg["year"] = preg["year"].astype(int)
    sources["anaemia_pregnant"] = preg

    # Anaemia in women of reproductive age
    sources["anaemia_women"] = load_gho_any_sex("anaemia_women_repro_age.csv", "anaemia_women_repro_pct")

    # Stunting (model-based)
    sources["stunting"] = load_gho_any_sex("stunting_prev.csv", "stunting_pct_who")

    # Wasting — survey-based, take both sexes or all
    wast = pd.read_csv(RAW / "who_gho" / "wasting_prev.csv")
    wast = wast.rename(columns={"SpatialDim": "iso3", "TimeDim": "year", "NumericValue": "wasting_pct"})
    wast = wast[~wast["iso3"].isin(AGGREGATE_ISO3)]
    # This indicator has many sub-dimensions; take overall (Dim1 is null or BTSX)
    wast = wast[wast["Dim1"].isna() | (wast["Dim1"] == "SEX_BTSX")]
    wast = wast[["iso3", "year", "wasting_pct"]].dropna(subset=["wasting_pct"])
    wast["year"] = wast["year"].astype(int)
    # Keep the most disaggregated (per-country) rows
    wast = wast.sort_values("wasting_pct").drop_duplicates(subset=["iso3", "year"], keep="first")
    sources["wasting"] = wast

    # TB incidence per 100k — no sex dimension in MDG_0000000020
    sources["tb"] = load_gho_any_sex("tb_incidence.csv", "tb_incidence_per100k")

    # HIV prevalence adults 15–49 (%) — no sex dimension
    sources["hiv"] = load_gho_any_sex("hiv_prevalence.csv", "hiv_prevalence_pct")

    # Malaria estimated incidence per 1000 population at risk
    sources["malaria"] = load_gho_any_sex("malaria_incidence.csv", "malaria_incidence_per1000")

    # Birth outcomes — WHO GHO (no sex/age dimension; load any-dim)
    sources["low_birthweight"]    = load_gho_any_sex("low_birthweight.csv",    "low_birthweight_pct")
    sources["preterm_birth_rate"] = load_gho_any_sex("preterm_birth_rate.csv", "preterm_birth_rate_pct")

    # Healthcare coverage — WHO GHO modelled national estimates (no sub-dimensions)
    sources["anc4_coverage"]  = load_gho_any_sex("anc4_coverage.csv",  "anc4_coverage_pct")
    sources["mcv1_coverage"]  = load_gho_any_sex("mcv1_coverage.csv",  "mcv1_coverage_pct")
    sources["mcv2_coverage"]  = load_gho_any_sex("mcv2_coverage.csv",  "mcv2_coverage_pct")
    sources["dtp3_coverage"]  = load_gho_any_sex("dtp3_coverage.csv",  "dtp3_coverage_pct")
    sources["pcv3_coverage"]  = load_gho_any_sex("pcv3_coverage.csv",  "pcv3_coverage_pct")
    sources["rotac_coverage"] = load_gho_any_sex("rotac_coverage.csv", "rotac_coverage_pct")
    sources["ors_coverage"]   = load_gho_any_sex("ors_coverage.csv",   "ors_coverage_pct")

    # Measles reported cases (raw count, not a rate — keep as integer)
    if (RAW / "who_gho" / "measles_reported_cases.csv").exists():
        msl = pd.read_csv(RAW / "who_gho" / "measles_reported_cases.csv")
        msl = msl.rename(columns={"SpatialDim": "iso3", "TimeDim": "year",
                                   "NumericValue": "measles_reported_cases"})
        msl = msl[~msl["iso3"].isin(AGGREGATE_ISO3)]
        msl = msl[msl["iso3"].str.len() == 3]
        msl = msl[["iso3", "year", "measles_reported_cases"]].dropna(subset=["measles_reported_cases"])
        msl["year"] = msl["year"].astype(int)
        msl["measles_reported_cases"] = pd.to_numeric(msl["measles_reported_cases"], errors="coerce")
        msl = msl.dropna(subset=["measles_reported_cases"])
        msl = msl.drop_duplicates(subset=["iso3", "year"])
        sources["measles_cases"] = msl
    else:
        print("  [skip] measles_reported_cases.csv not found")

    # Outcome and food-systems context indicators (World Bank)
    OUTCOMES_RAW = RAW / "outcomes"
    outcomes_map = {
        "u5_mortality_rate.csv":       "u5_mortality_per1000",
        "neonatal_mortality_rate.csv": "neonatal_mortality_per1000",
        "maternal_mortality_ratio.csv":"maternal_mortality_per100k",
        "human_capital_index.csv":     "hci_score",
        "hci_learning_years.csv":      "hci_learning_years",
        "gdp_per_capita_ppp.csv":      "gdp_per_capita_ppp",
        "severe_food_insecurity.csv":  "severe_food_insecurity_pct",
        "food_insecurity_mod_sev.csv": "food_insecurity_mod_sev_pct",
    }
    for fname, col in outcomes_map.items():
        fpath = OUTCOMES_RAW / fname
        if fpath.exists():
            out = pd.read_csv(fpath)
            out = out[~out["iso3"].isin(AGGREGATE_ISO3)]
            out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
            out = out.dropna(subset=["iso3", "year", col])
            out["year"] = out["year"].astype(int)
            key = fname.replace(".csv", "")
            sources[key] = out[["iso3", "year", col]]
            print(f"  {key}: {len(out):,} rows")
        else:
            print(f"  [skip] {fname} not found — run pull_outcomes.py first")

    # UNICEF/WB child malnutrition
    uni = pd.read_csv(RAW / "unicef" / "child_malnutrition_wb.csv")
    uni = uni[~uni["iso3"].isin(AGGREGATE_ISO3)]
    sources["unicef"] = uni[["iso3", "year", "stunting_pct", "underweight_pct"]].dropna(
        subset=["stunting_pct"], how="all"
    )

    # LSFF — FFI 2023 curated dataset (no time dimension: applies to all years)
    lsff_path = RAW / "lsff" / "ffi_country_status.csv"
    if lsff_path.exists():
        lsff = pd.read_csv(lsff_path)
        lsff = lsff[~lsff["iso3"].isin(AGGREGATE_ISO3)]
        # Broadcast to all years in the panel by merging without year key later
        sources["lsff"] = lsff[[
            "iso3", "wheat_flour_legislation", "maize_flour_legislation",
            "lsff_any_mandatory", "lsff_any_program", "lsff_coverage_proxy_pct",
        ]]
    else:
        print("  [skip] LSFF data not found — run pull_lsff.py first")

    # GBD micronutrient deficiencies (OWID + optional manual GBD export)
    GBD_RAW = RAW / "gbd"
    gbd_indicators = {
        "vitamin_a_deficiency.csv": "vitamin_a_deficiency_pct",
        "zinc_deficiency.csv":      "zinc_deficiency_pct",
        "iron_deficiency.csv":      "iron_deficiency_pct",
        "iodine_deficiency.csv":    "iodine_deficiency_pct",
        "sga_prevalence.csv":       "sga_prevalence_pct",
    }
    for fname, col in gbd_indicators.items():
        fpath = GBD_RAW / fname
        if fpath.exists():
            gbd = pd.read_csv(fpath)
            gbd = gbd[~gbd["iso3"].isin(AGGREGATE_ISO3)]
            gbd["year"] = pd.to_numeric(gbd["year"], errors="coerce").astype("Int64")
            gbd = gbd.dropna(subset=["iso3", "year", col])
            gbd["year"] = gbd["year"].astype(int)
            key = col.replace("_pct", "")
            sources[key] = gbd[["iso3", "year", col]]
            print(f"  Loaded {col}: {len(gbd):,} rows")
        else:
            print(f"  [skip] {fname} not found — run pull_gbd.py first")

    return sources


# ── Merge into country-year panel ────────────────────────────────────────────

def build_panel(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all sources on (iso3, year). LSFF has no year column — merged separately."""
    # Sources with a year dimension
    time_series_keys = [k for k in sources if k != "lsff"]
    base = sources["anaemia_children"]
    for name in time_series_keys:
        if name == "anaemia_children":
            continue
        base = base.merge(sources[name], on=["iso3", "year"], how="outer")

    # LSFF: broadcast across all years by merging on iso3 only
    if "lsff" in sources:
        base = base.merge(sources["lsff"], on="iso3", how="left")

    return base


# ── Country metadata ─────────────────────────────────────────────────────────

# WHO region mapping from the GHO data (ParentLocation). We'll rebuild from raw.
def build_country_metadata() -> pd.DataFrame:
    """Extract country name and WHO region from WHO GHO raw files."""
    df = pd.read_csv(RAW / "who_gho" / "tb_incidence.csv")
    meta = df[["SpatialDim", "ParentLocation", "ParentLocationCode"]].drop_duplicates()
    meta = meta.rename(columns={
        "SpatialDim": "iso3",
        "ParentLocation": "who_region",
        "ParentLocationCode": "who_region_code",
    })
    meta = meta[~meta["iso3"].isin(AGGREGATE_ISO3)]
    return meta


# Country names from World Bank API metadata (returns ISO3 + name for all countries)
def get_country_names() -> pd.DataFrame:
    try:
        import requests
        resp = requests.get(
            "https://api.worldbank.org/v2/country?format=json&per_page=300",
            timeout=20,
        )
        data = resp.json()
        if len(data) < 2:
            raise ValueError("No data")
        rows = []
        for c in data[1]:
            if c.get("iso2Code") and c.get("capitalCity"):  # filter to real countries
                rows.append({
                    "iso3": c["id"],
                    "country_name": c["name"],
                    "region": c.get("region", {}).get("value", ""),
                    "income_level": c.get("incomeLevel", {}).get("value", ""),
                })
        return pd.DataFrame(rows)
    except Exception:
        # Fallback: use WHO GHO indicator metadata for names
        return pd.DataFrame(columns=["iso3", "country_name", "region", "income_level"])


# ── Most-recent-year snapshot ─────────────────────────────────────────────────

def most_recent(panel: pd.DataFrame, window: tuple = (2010, 2023)) -> pd.DataFrame:
    """For each country, take the most recent value within the recency window per indicator.

    If a country has NO value for a given indicator within the window, fall back to
    the best available year from the full panel (handles OWID reference estimates that
    pre-date the window, e.g. Vitamin A 2005, Zinc 1990–2005).

    LSFF columns (no time dimension) are carried through without a _year suffix.
    """
    lsff_cols = [c for c in panel.columns if c.startswith("lsff_") or
                 c in ("wheat_flour_legislation", "maize_flour_legislation")]

    filtered = panel[(panel["year"] >= window[0]) & (panel["year"] <= window[1])].copy()
    filtered = filtered.sort_values("year", ascending=False)

    # Full panel sorted newest-first (for fallback)
    full_sorted = panel.sort_values("year", ascending=False)

    # Time-varying indicators only
    indicator_cols = [c for c in filtered.columns
                      if c not in ("iso3", "year") and c not in lsff_cols]
    result = filtered[["iso3", "year"]].drop_duplicates("iso3", keep="first").copy()

    for col in indicator_cols:
        # Primary: most recent within window
        col_data = filtered[["iso3", "year", col]].dropna(subset=[col])
        col_data = col_data.drop_duplicates("iso3", keep="first")[["iso3", col, "year"]].rename(
            columns={"year": f"{col}_year"}
        )

        # Fallback: best available year from full panel (for countries with no in-window value)
        fallback = full_sorted[["iso3", "year", col]].dropna(subset=[col])
        fallback = fallback.drop_duplicates("iso3", keep="first")[["iso3", col, "year"]].rename(
            columns={"year": f"{col}_year"}
        )
        # Merge fallback only for rows missing in primary
        all_countries = result[["iso3"]].copy()
        merged_primary  = all_countries.merge(col_data,  on="iso3", how="left")
        merged_fallback = all_countries.merge(fallback,  on="iso3", how="left")
        # Fill NaN values from fallback
        merged_primary[col] = merged_primary[col].fillna(merged_fallback[col])
        merged_primary[f"{col}_year"] = merged_primary[f"{col}_year"].fillna(
            merged_fallback[f"{col}_year"]
        )
        col_data = merged_primary

        result = result.merge(col_data, on="iso3", how="left")

    # Re-attach LSFF columns (non-time-varying) directly
    if lsff_cols:
        lsff_data = filtered[["iso3"] + lsff_cols].drop_duplicates("iso3")
        result = result.merge(lsff_data, on="iso3", how="left")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading source data...")
    sources = load_all_sources()
    for name, df in sources.items():
        print(f"  {name}: {len(df):,} rows")

    print("Building country-year panel...")
    panel = build_panel(sources)
    print(f"  Panel: {len(panel):,} rows × {len(panel.columns)} columns")

    print("Adding country metadata...")
    meta = build_country_metadata()
    names = get_country_names()
    panel = panel.merge(meta, on="iso3", how="left")
    panel = panel.merge(names[["iso3", "country_name", "region", "income_level"]], on="iso3", how="left")

    # Save full panel
    out_panel = PROCESSED / "commons_panel.csv"
    panel.to_csv(out_panel, index=False)
    print(f"  Saved panel → {out_panel}")

    # Save most-recent-year snapshot
    print("Building most-recent-year snapshot...")
    snapshot = most_recent(panel)
    # Drop all metadata cols before cleanly re-attaching — avoids _x/_y duplication
    meta_prefixes = ["who_region", "who_region_code", "country_name", "region", "income_level"]
    drop_cols = [c for c in snapshot.columns if any(c.startswith(p) for p in meta_prefixes)]
    snapshot = snapshot.drop(columns=drop_cols, errors="ignore")
    snapshot = snapshot.merge(meta, on="iso3", how="left")
    snapshot = snapshot.merge(names[["iso3", "country_name", "region", "income_level"]], on="iso3", how="left")
    out_snap = PROCESSED / "commons_snapshot.csv"
    snapshot.to_csv(out_snap, index=False)
    print(f"  Saved snapshot → {out_snap} ({len(snapshot)} countries)")

    print("Done.")
    return panel, snapshot


if __name__ == "__main__":
    run()
