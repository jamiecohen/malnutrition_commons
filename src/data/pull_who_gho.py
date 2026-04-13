"""
Pull indicators from the WHO Global Health Observatory (GHO) OData API.
Saves one CSV per indicator to data/raw/who_gho/.
Run idempotently — skips download if file already exists.
"""

import os
import requests
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "who_gho"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://ghoapi.azureedge.net/api"

INDICATORS = {
    "anaemia_children":         "NUTRITION_ANAEMIA_CHILDREN_PREV",
    "anaemia_women_repro_age":  "NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV",
    "anaemia_pregnant_women":   "NUTRITION_ANAEMIA_PREGNANT_PREV",
    "stunting_prev":            "NUTSTUNTINGPREV",
    "wasting_prev":             "NUTRITION_WH_2",
    "tb_incidence":             "MDG_0000000020",
    "hiv_prevalence":           "MDG_0000000029",   # HIV prevalence adults 15–49 (%)
    "malaria_incidence":        "MALARIA_EST_INCIDENCE",  # Estimated malaria incidence (per 1000 at risk)
    # Birth outcomes
    "low_birthweight":          "LBW_PREVALENCE",         # Low birthweight prevalence (%) — 158 countries, 2000–2020
    "preterm_birth_rate":       "PRETERMBIRTH_RATE",      # Preterm birth rate (%) — 108 countries, 2010–2020
    # Healthcare coverage
    "anc4_coverage":            "WHS4_154",               # ANC 4+ visits (%) — 164 countries, 1990–2024
    "mcv1_coverage":            "WHS8_110",               # Measles vaccine 1st dose coverage (%) — 182 countries, 2000–2024
    "mcv2_coverage":            "MCV2",                   # Measles vaccine 2nd dose coverage (%) — 188 countries, 2000–2024
    "dtp3_coverage":            "WHS4_100",               # DTP3 coverage (%) — 126 countries, 2000–2024
    "pcv3_coverage":            "PCV3",                   # Pneumococcal PCV3 coverage (%) — 176 countries, 2008–2024
    "rotac_coverage":           "ROTAC",                  # Rotavirus vaccine coverage (%) — 179 countries, 2006–2024
    # Childhood infectious disease
    "measles_reported_cases":   "WHS3_62",                # Measles reported cases (count) — 189 countries, 1974–2024
    "ors_coverage":             "UNICEF_ORS",             # ORS treatment for child diarrhoea (%) — 97 countries, 1990–2017
}


def fetch_indicator(code: str) -> pd.DataFrame:
    """Fetch all country-level records for a GHO indicator, handling pagination."""
    url = f"{BASE_URL}/{code}"
    params = {
        "$filter": "SpatialDimType eq 'COUNTRY'",
        "$select": "SpatialDim,TimeDim,Dim1,NumericValue,Low,High,ParentLocation,ParentLocationCode",
    }
    records = []
    while url:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("value", []))
        # OData pagination
        url = data.get("@odata.nextLink")
        params = {}  # next link already has query params embedded
    return pd.DataFrame(records)


def pull_all():
    for name, code in INDICATORS.items():
        out_path = OUTPUT_DIR / f"{name}.csv"
        if out_path.exists():
            print(f"  [skip] {name} already downloaded")
            continue
        print(f"  [pull] {name} ({code})...")
        try:
            df = fetch_indicator(code)
            df.to_csv(out_path, index=False)
            print(f"         saved {len(df):,} rows → {out_path.name}")
        except Exception as e:
            print(f"         ERROR: {e}")


if __name__ == "__main__":
    print("Pulling WHO GHO indicators...")
    pull_all()
    print("Done.")
