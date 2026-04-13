"""
Pull downstream outcome and food-systems context indicators from World Bank API.

These fill the gaps in the Integrated Nutrition Impact Framework:
  - Under-5 mortality rate      (SH.DYN.MORT)   — the headline child survival outcome
  - Neonatal mortality rate     (SH.DYN.NMRT)   — captures newborn period specifically
  - Maternal mortality ratio    (SH.STA.MMRT)   — links maternal anaemia → maternal death
  - Human Capital Index         (HD.HCI.OVRL)   — composite nutrition→development outcome
  - HCI learning-adjusted yrs   (HD.HCI.LAYS)   — cognition/schooling pathway
  - GDP per capita PPP          (NY.GDP.PCAP.PP.KD)  — economic productivity pathway
  - Severe food insecurity      (SN.ITK.SVFI.ZS) — upstream food-systems constraint
  - Mod.+severe food insecurity (SN.ITK.MSFI.ZS) — broader food access constraint

All written to data/raw/outcomes/ as ISO3-year-value CSVs.
"""

import sys
from pathlib import Path
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR   = PROJECT_ROOT / "data" / "raw" / "outcomes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WB_BASE = "https://api.worldbank.org/v2/country/all/indicator"

INDICATORS = {
    "u5_mortality_rate":      ("SH.DYN.MORT",       "u5_mortality_per1000"),
    "neonatal_mortality_rate":("SH.DYN.NMRT",       "neonatal_mortality_per1000"),
    "maternal_mortality_ratio":("SH.STA.MMRT",      "maternal_mortality_per100k"),
    "human_capital_index":    ("HD.HCI.OVRL",        "hci_score"),
    "hci_learning_years":     ("HD.HCI.LAYS",        "hci_learning_years"),
    "gdp_per_capita_ppp":     ("NY.GDP.PCAP.PP.KD", "gdp_per_capita_ppp"),
    "severe_food_insecurity": ("SN.ITK.SVFI.ZS",    "severe_food_insecurity_pct"),
    "food_insecurity_mod_sev":("SN.ITK.MSFI.ZS",    "food_insecurity_mod_sev_pct"),
}

AGGREGATE_ISO3 = {
    "WLD","LMY","UMC","LMC","LIC","HIC","AFR","AMR","SEAR","EUR","EMR","WPR",
    "EAP","ECA","LAC","MNA","NAC","SAS","SSA","EAS","TEA","TSA","TLA","TMN",
    "TSS","IBT","IBD","IDB","IDX","ARB","CSS","CEB","ECS","TEC","MEA",
}


def fetch_wb(wb_code: str, value_col: str) -> pd.DataFrame:
    """Paginate through World Bank API and return clean (iso3, year, value) frame."""
    records = []
    page = 1
    while True:
        r = requests.get(
            f"{WB_BASE}/{wb_code}",
            params={"format": "json", "per_page": "1000", "page": str(page)},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            break
        for row in data[1]:
            if row.get("value") is None:
                continue
            iso3 = row.get("countryiso3code", "")
            if len(iso3) != 3 or iso3 in AGGREGATE_ISO3:
                continue
            records.append({
                "iso3": iso3,
                "year": int(row["date"]),
                value_col: float(row["value"]),
            })
        if len(data[1]) < 1000:
            break
        page += 1

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(["iso3", "year"]).drop_duplicates(
        subset=["iso3", "year"]
    ).reset_index(drop=True)
    return df


def pull_all(force: bool = False):
    for name, (wb_code, value_col) in INDICATORS.items():
        out_path = OUTPUT_DIR / f"{name}.csv"
        if out_path.exists() and not force:
            print(f"  [skip] {name} already downloaded")
            continue
        print(f"  [pull] {name} ({wb_code})...")
        try:
            df = fetch_wb(wb_code, value_col)
            if df.empty:
                print(f"         no data returned")
                continue
            df.to_csv(out_path, index=False)
            print(f"         saved {len(df):,} rows → {out_path.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"         ERROR: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pull_all(force=args.force)
    print("Done.")
