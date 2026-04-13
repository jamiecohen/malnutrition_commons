"""
Pull food security indicators from FAO FAOSTAT and World Bank.

Key indicators:
- Prevalence of undernourishment (PoU) — from FAO via World Bank API
- Cost of Healthy Diet (CoHD) — affordability indicator
"""

import requests
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "fao"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# World Bank republishes key FAO food security indicators with stable codes
WB_FOOD_SECURITY = {
    "SN.ITK.DEFC.ZS": "undernourishment_pct",   # Prevalence of undernourishment
}

# FAO FAOSTAT bulk CSV for food security suite
FAO_FOOD_SECURITY_URL = (
    "https://fenixservices.fao.org/faostat/static/bulkdownloads/"
    "Food_Security_Data_E_All_Data_(Normalized).zip"
)

# Simpler: use World Bank API for FAO-derived indicators
def pull_wb_food_security() -> pd.DataFrame:
    all_dfs = []
    for wb_code, col_name in WB_FOOD_SECURITY.items():
        print(f"    fetching {col_name}...")
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


def pull_fao_faostat_undernourishment() -> pd.DataFrame:
    """
    Pull PoU directly from FAOSTAT API (suite of food security indicators).
    Element code 21010 = Prevalence of undernourishment (%)
    Item code 210011 = Prevalence of undernourishment (3-year average)
    """
    url = "https://fenixservices.fao.org/faostat/api/v1/en/data/FS"
    params = {
        "area_cs": "ISO3",
        "element": "21010",   # Prevalence of undernourishment
        "item": "210011",
        "year": ",".join(str(y) for y in range(2005, 2024)),
        "output_type": "csv",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        return df
    except Exception as e:
        print(f"    FAOSTAT direct pull failed: {e}")
        return pd.DataFrame()


def pull_all():
    # World Bank / FAO food security indicators
    out_wb = OUTPUT_DIR / "food_security_wb.csv"
    if out_wb.exists():
        print("  [skip] FAO/World Bank food security data already downloaded")
    else:
        print("  [pull] FAO food security indicators (via World Bank API)...")
        df = pull_wb_food_security()
        if not df.empty:
            df.to_csv(out_wb, index=False)
            print(f"         saved {len(df):,} rows → {out_wb.name}")
        else:
            print("         ERROR: no data returned")

    # FAOSTAT PoU direct
    out_fao = OUTPUT_DIR / "faostat_undernourishment.csv"
    if out_fao.exists():
        print("  [skip] FAOSTAT undernourishment data already downloaded")
    else:
        print("  [pull] FAOSTAT prevalence of undernourishment...")
        df = pull_fao_faostat_undernourishment()
        if not df.empty:
            df.to_csv(out_fao, index=False)
            print(f"         saved {len(df):,} rows → {out_fao.name}")
        else:
            print("         WARNING: FAOSTAT direct pull returned no data (will use WB fallback)")


if __name__ == "__main__":
    print("Pulling FAO food security data...")
    pull_all()
    print("Done.")
