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

    # UNICEF/WB child malnutrition
    uni = pd.read_csv(RAW / "unicef" / "child_malnutrition_wb.csv")
    uni = uni[~uni["iso3"].isin(AGGREGATE_ISO3)]
    sources["unicef"] = uni[["iso3", "year", "stunting_pct", "underweight_pct"]].dropna(
        subset=["stunting_pct"], how="all"
    )

    return sources


# ── Merge into country-year panel ────────────────────────────────────────────

def build_panel(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all sources on (iso3, year)."""
    base = sources["anaemia_children"]
    for name, df in sources.items():
        if name == "anaemia_children":
            continue
        base = base.merge(df, on=["iso3", "year"], how="outer")
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
    """For each country, take the most recent value within the recency window per indicator."""
    filtered = panel[(panel["year"] >= window[0]) & (panel["year"] <= window[1])].copy()
    # Sort descending by year so first row per country is most recent
    filtered = filtered.sort_values("year", ascending=False)

    indicator_cols = [c for c in filtered.columns if c not in ("iso3", "year")]
    result = filtered[["iso3", "year"]].drop_duplicates("iso3", keep="first").copy()

    for col in indicator_cols:
        col_data = filtered[["iso3", "year", col]].dropna(subset=[col])
        col_data = col_data.drop_duplicates("iso3", keep="first")[["iso3", col, "year"]].rename(
            columns={"year": f"{col}_year"}
        )
        result = result.merge(col_data, on="iso3", how="left")

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
