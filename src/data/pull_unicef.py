"""
Pull UNICEF malnutrition data.

UNICEF publishes a flat CSV of child malnutrition indicators (stunting, wasting,
underweight, overweight) aggregated from national surveys. We download it directly
and save to data/raw/unicef/.
"""

import requests
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "unicef"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# UNICEF SOWC / global malnutrition estimates — JME (Joint Malnutrition Estimates)
# This is the UNICEF/WHO/World Bank joint dataset published annually.
UNICEF_JME_URL = (
    "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/data/"
    "UNICEF,NUTRITION,1.0/."
    "MNCH_STUNT+MNCH_WAST+MNCH_OVERWT+MNCH_UNDERWT"
    "._T._T._T.?format=csv&startPeriod=2000&endPeriod=2023"
)

# Fallback: pre-formatted CSV from UNICEF data warehouse
UNICEF_FALLBACK_URLS = {
    "stunting": "https://data.unicef.org/wp-content/uploads/2023/06/JME-2023-Country-Dataset-Stunting.csv",
    "wasting":  "https://data.unicef.org/wp-content/uploads/2023/06/JME-2023-Country-Dataset-Wasting.csv",
}

# More reliable: UNICEF Data Warehouse API
UNICEF_DW_BASE = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest"


def pull_jme_sdmx() -> pd.DataFrame | None:
    """Try to pull Joint Malnutrition Estimates via UNICEF SDMX API."""
    try:
        resp = requests.get(UNICEF_JME_URL, timeout=60)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        return df
    except Exception as e:
        print(f"  SDMX pull failed: {e}")
        return None


def pull_world_bank_malnutrition() -> pd.DataFrame:
    """
    Pull child malnutrition indicators from World Bank API as a reliable fallback.
    These are the same JME estimates republished by the World Bank.
    """
    indicators = {
        "SH.STA.STNT.ZS": "stunting_pct",
        "SH.STA.WAST.ZS": "wasting_pct",
        "SH.STA.MALN.ZS": "underweight_pct",
    }
    all_dfs = []
    for wb_code, col_name in indicators.items():
        rows = []
        page = 1
        while True:
            url = (
                f"https://api.worldbank.org/v2/country/all/indicator/{wb_code}"
                f"?format=json&per_page=1000&mrv=10&page={page}"
            )
            try:
                resp = requests.get(url, timeout=90)
            except requests.exceptions.Timeout:
                print(f"    TIMEOUT on {wb_code} page {page}, skipping")
                break
            resp.raise_for_status()
            data = resp.json()
            if len(data) < 2 or not data[1]:
                break
            for record in data[1]:
                if record.get("value") is None:
                    continue
                rows.append({
                    "iso3": record.get("countryiso3code", record["country"]["id"]),
                    "country_name": record["country"]["value"],
                    "year": int(record["date"]),
                    col_name: float(record["value"]),
                })
            meta = data[0]
            if page >= meta.get("pages", 1):
                break
            page += 1
        if rows:
            all_dfs.append(pd.DataFrame(rows))

    if not all_dfs:
        return pd.DataFrame()

    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = merged.merge(df, on=["iso3", "country_name", "year"], how="outer")
    return merged


def pull_all():
    out_path = OUTPUT_DIR / "child_malnutrition_wb.csv"
    if out_path.exists():
        print("  [skip] UNICEF/JME malnutrition data already downloaded")
        return

    print("  [pull] Child malnutrition indicators (World Bank/JME)...")
    df = pull_world_bank_malnutrition()
    if not df.empty:
        df.to_csv(out_path, index=False)
        print(f"         saved {len(df):,} rows → {out_path.name}")
    else:
        print("         ERROR: no data returned")


if __name__ == "__main__":
    print("Pulling UNICEF/JME malnutrition data...")
    pull_all()
    print("Done.")
