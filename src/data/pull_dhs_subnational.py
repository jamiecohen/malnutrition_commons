"""
Pull Nigeria state-level nutrition and health indicators from the DHS Program API.

Source: DHS Program public REST API (no registration required)
  https://api.dhsprogram.com/

Survey: Nigeria DHS 2018 (NDHS 2018), state-level breakdown (37 areas: 36 states + FCT)

Indicators pulled:
  CN_NUTS_C_HA2  — Stunting (<-2 SD height-for-age), children <5 (%)
  CN_NUTS_C_WH2  — Wasting (<-2 SD weight-for-height), children <5 (%)
  CN_NUTS_C_WA2  — Underweight (<-2 SD weight-for-age), children <5 (%)
  AN_ANEM_C_ANY  — Anaemia (any), children 6–59 months (%)
  RH_ANCP_W_N4P  — ANC4+ visits, women with recent live birth (%)
  CH_VACC_C_DP3  — DTP3 vaccination coverage, children 12–23 months (%)
  CH_VACC_C_MS1  — MCV1 vaccination coverage, children 12–23 months (%)
  CN_BIRT_C_LBW  — Low birthweight (<2.5 kg) (%)

Also downloads Nigeria ADM1 (state) boundary GeoJSON from geoBoundaries.

Outputs:
  data/raw/subnational/nga_dhs_2018_states.csv   — long-format indicator data
  data/processed/subnational/nga_states_wide.csv — wide-format, one row per state
  data/raw/geo/nga_adm1.geojson                  — state boundary polygons
"""

import sys
import json
from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR   = PROJECT_ROOT / "data" / "raw" / "subnational"
GEO_DIR   = PROJECT_ROOT / "data" / "raw" / "geo"
PROC_DIR  = PROJECT_ROOT / "data" / "processed" / "subnational"

for d in (RAW_DIR, GEO_DIR, PROC_DIR):
    d.mkdir(parents=True, exist_ok=True)

DHS_BASE = "https://api.dhsprogram.com/rest/dhs/data"

# DHS indicator codes → output column names
# Codes verified against Nigeria 2018 DHS survey (NG2018DHS) indicator catalog
INDICATORS = {
    "CN_NUTS_C_HA2": "stunting_pct",       # Stunting (<-2 SD HAZ), children <5
    "CN_NUTS_C_WH2": "wasting_pct",        # Wasting (<-2 SD WHZ), children <5
    "CN_NUTS_C_WA2": "underweight_pct",    # Underweight (<-2 SD WAZ), children <5
    "CN_ANMC_C_ANY": "anaemia_children_pct", # Any anaemia, children 6-59 months
    "RH_ANCN_W_N4P": "anc4_coverage_pct", # ANC 4+ visits, women with recent birth
    "CH_VACC_C_DP3": "dtp3_coverage_pct", # DPT3 vaccination, children 12-23 months
    "CH_VACC_C_MSL": "mcv1_coverage_pct", # Measles vaccination received
    "CH_SZWT_C_L25": "low_birthweight_pct", # Birth weight <2.5 kg
}

# DHS CharacteristicLabel → geoBoundaries shapeName
# Only entries that differ from the DHS name (after stripping leading "..")
STATE_NAME_MAP = {
    "FCT Abuja": "Abuja Federal Capital Territory",
}

GEOBOUNDARIES_META = "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/"


def fetch_dhs_indicators(force: bool = False) -> pd.DataFrame:
    """Fetch all indicators in one call and return a long-format DataFrame."""
    out_path = RAW_DIR / "nga_dhs_2018_states.csv"
    if out_path.exists() and not force:
        print(f"  [skip] DHS data already downloaded ({out_path.name})")
        return pd.read_csv(out_path)

    indicator_str = ",".join(INDICATORS.keys())
    url = (
        f"{DHS_BASE}?f=json"
        f"&countryIds=NG"
        f"&surveyYear=2018"
        f"&indicatorIds={indicator_str}"
        f"&breakdown=subnational"
        f"&perpage=5000"
    )
    print(f"  [pull] DHS API → Nigeria 2018 subnational ({len(INDICATORS)} indicators)...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()

    records = []
    for row in payload.get("Data", []):
        label = row.get("CharacteristicLabel", "")
        # State-level rows have a ".." prefix; zone-level rows do not
        if not label.startswith(".."):
            continue
        state_raw = label.lstrip(".")
        state_name = STATE_NAME_MAP.get(state_raw, state_raw)
        records.append({
            "state_name":   state_name,
            "state_raw":    state_raw,
            "indicator_id": row["IndicatorId"],
            "indicator":    INDICATORS.get(row["IndicatorId"], row["IndicatorId"]),
            "value":        row.get("Value"),
            "ci_low":       row.get("CILow"),
            "ci_high":      row.get("CIHigh"),
            "denom_weighted": row.get("DenominatorWeighted"),
            "survey_year":  row.get("SurveyYear"),
        })

    df = pd.DataFrame(records)
    print(f"         {len(df)} rows — {df['state_name'].nunique()} states, "
          f"{df['indicator_id'].nunique()} indicators")

    df.to_csv(out_path, index=False)
    print(f"         saved → {out_path.relative_to(PROJECT_ROOT)}")
    return df


def pivot_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per state, one column per indicator."""
    wide = df_long.pivot_table(
        index="state_name", columns="indicator", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    # Add CI columns for stunting (the headline indicator)
    ci = df_long[df_long["indicator"] == "stunting_pct"][
        ["state_name", "ci_low", "ci_high", "denom_weighted"]
    ].rename(columns={"ci_low": "stunting_ci_low", "ci_high": "stunting_ci_high",
                       "denom_weighted": "stunting_n"})
    wide = wide.merge(ci, on="state_name", how="left")

    # Add Nigeria zone classification
    zone_map = _state_zone_map()
    wide["zone"] = wide["state_name"].map(zone_map)

    return wide


def fetch_geojson(force: bool = False) -> dict:
    """Download Nigeria ADM1 boundary GeoJSON from geoBoundaries."""
    out_path = GEO_DIR / "nga_adm1.geojson"
    if out_path.exists() and not force:
        print(f"  [skip] GeoJSON already downloaded ({out_path.name})")
        with open(out_path) as f:
            return json.load(f)

    print("  [pull] geoBoundaries → Nigeria ADM1 boundary GeoJSON...")
    meta = requests.get(GEOBOUNDARIES_META, timeout=20).json()
    gj_url = meta["gjDownloadURL"]
    r = requests.get(gj_url, timeout=60)
    r.raise_for_status()
    gj = r.json()

    with open(out_path, "w") as f:
        json.dump(gj, f)
    print(f"         {len(gj['features'])} state polygons → "
          f"{out_path.relative_to(PROJECT_ROOT)}")
    return gj


def _state_zone_map() -> dict:
    """Map each state to its geopolitical zone (for color grouping in maps)."""
    return {
        # North Central
        "Benue": "North Central", "Kogi": "North Central", "Kwara": "North Central",
        "Nasarawa": "North Central", "Niger": "North Central", "Plateau": "North Central",
        "Abuja Federal Capital Territory": "North Central",
        # North East
        "Adamawa": "North East", "Bauchi": "North East", "Borno": "North East",
        "Gombe": "North East", "Taraba": "North East", "Yobe": "North East",
        # North West
        "Jigawa": "North West", "Kaduna": "North West", "Kano": "North West",
        "Katsina": "North West", "Kebbi": "North West", "Sokoto": "North West",
        "Zamfara": "North West",
        # South East
        "Abia": "South East", "Anambra": "South East", "Ebonyi": "South East",
        "Enugu": "South East", "Imo": "South East",
        # South South
        "Akwa Ibom": "South South", "Bayelsa": "South South", "Cross River": "South South",
        "Delta": "South South", "Edo": "South South", "Rivers": "South South",
        # South West
        "Ekiti": "South West", "Lagos": "South West", "Ogun": "South West",
        "Ondo": "South West", "Osun": "South West", "Oyo": "South West",
    }


def pull_all(force: bool = False):
    df_long = fetch_dhs_indicators(force=force)
    if df_long.empty:
        print("  No data retrieved; aborting.")
        return

    wide = pivot_wide(df_long)
    out_wide = PROC_DIR / "nga_states_wide.csv"
    wide.to_csv(out_wide, index=False)
    print(f"  Wide dataset: {len(wide)} states × "
          f"{len([c for c in wide.columns if c not in ('state_name','zone')])} indicators"
          f" → {out_wide.relative_to(PROJECT_ROOT)}")

    fetch_geojson(force=force)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pull_all(force=args.force)
    print("Done.")
