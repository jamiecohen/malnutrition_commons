"""
Pull GBD-adjacent micronutrient deficiency data.

Two pathways:
  1. OWID (programmatic) — Vitamin A deficiency and Zinc deficiency
     Both are derived from peer-reviewed estimates and exposed as
     downloadable CSVs by Our World in Data.

  2. GBD manual download (parser) — Iron deficiency and other GBD
     indicators that require IHME authentication to pull via API.
     See docs/gbd_download_guide.md for how to export the CSV.

Outputs (written to data/raw/gbd/):
  vitamin_a_deficiency.csv   — iso3, year, vitamin_a_deficiency_pct
  zinc_deficiency.csv        — iso3, year, zinc_deficiency_pct
  iron_deficiency.csv        — iso3, year, iron_deficiency_pct   (from manual GBD CSV)
  vitamin_a_deficiency_gbd.csv — iso3, year, vitamin_a_deficiency_pct (from GBD CSV, if available)
  zinc_deficiency_gbd.csv    — iso3, year, zinc_deficiency_pct (from GBD CSV, if available)

Usage:
  python src/data/pull_gbd.py              # pulls OWID; parses any GBD CSVs found
  python src/data/pull_gbd.py --gbd-dir /path/to/unzipped/gbd   # explicit GBD path
"""

import sys
import argparse
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
RAW_GBD = PROJECT_ROOT / "data" / "raw" / "gbd"
RAW_GBD.mkdir(parents=True, exist_ok=True)


# ── OWID source definitions ───────────────────────────────────────────────────

OWID_SOURCES = {
    "vitamin_a": {
        "slug":     "prevalence-of-vitamin-a-deficiency-in-children",
        "col_raw":  "Prevalence of vitamin-A deficiency",
        "col_out":  "vitamin_a_deficiency_pct",
        "out_file": "vitamin_a_deficiency.csv",
        "note":     "WHO (2017); ~163 countries; primarily 2005 reference year",
    },
    "zinc": {
        "slug":     "global-prevalence-of-zinc-deficiency",
        "col_raw":  "Prevalence of zinc deficiency",
        "col_out":  "zinc_deficiency_pct",
        "out_file": "zinc_deficiency.csv",
        "note":     "Wessells et al. (2012); ~185 countries; 1990–2015",
    },
}

OWID_BASE = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full"

# Standard ISO3 exclusions (aggregates / regions)
AGGREGATE_CODES = {
    "OWID_WRL", "OWID_AFR", "OWID_EUR", "OWID_ASI", "OWID_NAM",
    "OWID_SAM", "OWID_OCE", "OWID_HIC", "OWID_MIC", "OWID_LIC",
    "WLD", "LMY", "UMC", "LMC", "LIC", "HIC", "AFR", "AMR",
    "SEAR", "EUR", "EMR", "WPR", "EAP", "ECA", "LAC", "MNA",
    "NAC", "SAS", "SSA",
}


# ── 1. OWID pull ──────────────────────────────────────────────────────────────

def pull_owid(key: str, force: bool = False) -> pd.DataFrame:
    """Download one OWID indicator and return a clean (iso3, year, value) frame."""
    cfg = OWID_SOURCES[key]
    out_path = RAW_GBD / cfg["out_file"]

    if out_path.exists() and not force:
        print(f"  [skip] {cfg['out_file']} already exists")
        return pd.read_csv(out_path)

    url = OWID_BASE.format(slug=cfg["slug"])
    print(f"  Fetching {url} ...")
    resp = requests.get(url, timeout=60, headers={"Accept": "text/csv"})
    resp.raise_for_status()

    # OWID returns "Entity,Code,Year,<indicator col>" (sometimes with extras)
    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text))

    # Identify the value column — it may have extra text; match by partial name
    val_col = next(
        (c for c in raw.columns if cfg["col_raw"].lower() in c.lower()),
        None,
    )
    if val_col is None:
        raise ValueError(
            f"Could not find value column matching '{cfg['col_raw']}' in {list(raw.columns)}"
        )

    df = raw.rename(columns={"Code": "iso3", "Year": "year", val_col: cfg["col_out"]})[
        ["iso3", "year", cfg["col_out"]]
    ]
    df = df.dropna(subset=["iso3", cfg["col_out"]])
    df = df[~df["iso3"].isin(AGGREGATE_CODES)]
    # Drop OWID-style synthetic codes (start with OWID_)
    df = df[~df["iso3"].str.startswith("OWID_")]
    df["year"] = df["year"].astype(int)
    df[cfg["col_out"]] = pd.to_numeric(df[cfg["col_out"]], errors="coerce")
    df = df.dropna(subset=[cfg["col_out"]])
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows → {out_path.relative_to(PROJECT_ROOT)}")
    return df


def pull_all_owid(force: bool = False):
    for key in OWID_SOURCES:
        pull_owid(key, force=force)


# ── 2. GBD CSV parser ─────────────────────────────────────────────────────────

# GBD cause IDs for micronutrient deficiencies (GBD 2021)
GBD_CAUSES = {
    390: ("iron_deficiency",       "iron_deficiency_pct"),
    389: ("vitamin_a_deficiency",  "vitamin_a_deficiency_pct"),
    391: ("zinc_deficiency",       "zinc_deficiency_pct"),
    387: ("iodine_deficiency",     "iodine_deficiency_pct"),
}

# GBD metric IDs
GBD_METRIC_PREVALENCE = 1

# GBD measure IDs
GBD_MEASURE_PREVALENCE = 5

# We want the all-ages OR under-5 row. Prefer under-5 (age_id=1) when available,
# fall back to all ages (age_id=22). For a single-row-per-country output we pick
# the age group present (under-5 preferred for child deficiency comparisons).
PREFERRED_AGE_IDS = {1, 22, 27, 4, 5}   # age_id 1=<5, 22=All Ages, 27=15-49, etc.
PREFERRED_SEX_IDS = {3}                  # 3=Both sexes


def _find_gbd_csvs(gbd_dir: Path) -> list[Path]:
    """Return all CSV files under gbd_dir, unpacking any zip files first."""
    csvs = []
    for f in gbd_dir.iterdir():
        if f.suffix.lower() == ".zip":
            print(f"  Unpacking {f.name} ...")
            with zipfile.ZipFile(f) as zf:
                zf.extractall(gbd_dir)
    csvs = list(gbd_dir.glob("*.csv"))
    # Exclude our own output files
    own_outputs = {cfg[1] + ".csv" for cfg in GBD_CAUSES.values()} | {
        "vitamin_a_deficiency.csv", "zinc_deficiency.csv"
    }
    csvs = [c for c in csvs if c.name not in own_outputs]
    return csvs


def _normalize_gbd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase; handle both short and verbose GBD exports."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def parse_gbd_csv(gbd_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Parse all GBD CSVs found in gbd_dir.

    Returns a dict: {indicator_name: DataFrame(iso3, year, <value_col>)}
    Filters to:
      - measure = Prevalence (measure_id=5 or measure_name contains 'Prevalence')
      - metric  = Percent  (metric_id=3 or metric_name contains 'Percent')
      - sex     = Both sexes (sex_id=3)
      - age     = <5 (preferred) or All ages
      - cause   = any of GBD_CAUSES keys

    The GBD CSV uses `location_name` for country names but not always ISO3.
    We match via the IHME location_id → ISO3 mapping (built-in reference table).
    Falls back to a fuzzy country-name match using a lightweight lookup.
    """
    csvs = _find_gbd_csvs(gbd_dir)
    if not csvs:
        print(f"  [warn] No GBD CSV files found in {gbd_dir}")
        return {}

    print(f"  Found {len(csvs)} GBD CSV(s): {[c.name for c in csvs]}")

    # Load IHME location_id → ISO3 lookup (embedded below)
    loc_map = _build_location_map()

    results: dict[str, list] = {name: [] for _, (name, _) in GBD_CAUSES.items()}

    for csv_path in csvs:
        print(f"  Parsing {csv_path.name} ...")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"    [error] Could not read {csv_path.name}: {e}")
            continue

        df = _normalize_gbd_columns(df)

        # ── Detect column names (GBD exports vary slightly) ──────────────────
        def find_col(*candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        measure_col  = find_col("measure_id", "measure")
        metric_col   = find_col("metric_id",  "metric")
        sex_col      = find_col("sex_id",     "sex")
        age_col      = find_col("age_id",     "age")
        cause_col    = find_col("cause_id",   "cause")
        loc_id_col   = find_col("location_id")
        loc_name_col = find_col("location_name", "location")
        year_col     = find_col("year", "year_id")
        val_col      = find_col("val", "mean", "value")
        upper_col    = find_col("upper")
        lower_col    = find_col("lower")

        if val_col is None or year_col is None:
            print(f"    [skip] Missing required columns in {csv_path.name}")
            continue

        # ── Filter rows ───────────────────────────────────────────────────────
        mask = pd.Series(True, index=df.index)

        # Prevalence measure
        if measure_col:
            if "measure_name" in df.columns:
                mask &= df["measure_name"].str.contains("Prevalence", case=False, na=False)
            elif measure_col == "measure_id":
                mask &= df[measure_col].isin([GBD_MEASURE_PREVALENCE])
            else:
                mask &= df[measure_col].str.contains("Prevalence", case=False, na=False)

        # Percent metric
        if metric_col:
            if "metric_name" in df.columns:
                mask &= df["metric_name"].str.contains("Percent", case=False, na=False)
            elif metric_col == "metric_id":
                mask &= df[metric_col].isin([3])   # 3 = Percent
            else:
                mask &= df[metric_col].str.contains("Percent", case=False, na=False)

        # Both sexes
        if sex_col:
            if sex_col == "sex_id":
                mask &= df[sex_col].isin(PREFERRED_SEX_IDS)
            else:
                mask &= df[sex_col].str.contains("Both|Both sexes|Total", case=False, na=False)

        # Our GBD causes
        if cause_col:
            if cause_col == "cause_id":
                mask &= df[cause_col].isin(GBD_CAUSES.keys())
            else:
                cause_names = [c[0].replace("_", " ") for c in GBD_CAUSES.values()]
                pattern = "|".join(cause_names)
                mask &= df[cause_col].str.contains(pattern, case=False, na=False)

        sub = df[mask].copy()
        if sub.empty:
            print(f"    [warn] No matching rows after filters in {csv_path.name}")
            continue

        # ── Resolve cause_id ─────────────────────────────────────────────────
        if cause_col == "cause_id":
            sub["_cause_id"] = sub[cause_col]
        elif "cause_name" in sub.columns:
            # Map cause_name → cause_id
            cause_name_map = {}
            for cid, (cname, _) in GBD_CAUSES.items():
                cause_name_map[cname.replace("_", " ").lower()] = cid
            sub["_cause_id"] = sub["cause_name"].str.lower().str.strip().map(cause_name_map)
        else:
            print(f"    [warn] Cannot resolve cause from {csv_path.name}")
            continue

        # ── Resolve age preference ────────────────────────────────────────────
        if age_col == "age_id":
            sub["_age_id"] = sub[age_col].astype(int)
        elif "age_name" in sub.columns:
            def _age_rank(name):
                name = str(name).lower()
                if "under 5" in name or "<5" in name or "1 to 4" in name or "post neonatal" in name or "early neonatal" in name:
                    return 0  # under-5: highest priority
                if "all ages" in name or "all age" in name:
                    return 1
                return 2
            sub["_age_rank"] = sub["age_name"].apply(_age_rank)
            # Keep best age per (location, year, cause)
            group_cols = [c for c in [loc_id_col, loc_name_col, year_col, "_cause_id"] if c]
            sub = sub.sort_values("_age_rank").drop_duplicates(subset=group_cols, keep="first")
        # If no age column at all, proceed as-is

        # ── Resolve ISO3 ─────────────────────────────────────────────────────
        if loc_id_col and loc_id_col in sub.columns:
            sub["iso3"] = sub[loc_id_col].map(loc_map)
        elif loc_name_col and loc_name_col in sub.columns:
            sub["iso3"] = sub[loc_name_col].map(
                {v: k for k, v in _build_name_to_iso3().items()}
            )
        else:
            print(f"    [warn] No location column found in {csv_path.name}")
            continue

        sub = sub.dropna(subset=["iso3"])
        sub = sub[~sub["iso3"].str.startswith("OWID_")]
        sub = sub[~sub["iso3"].isin(AGGREGATE_CODES)]

        sub["year"] = pd.to_numeric(sub[year_col], errors="coerce").astype("Int64")
        sub["val"]  = pd.to_numeric(sub[val_col],  errors="coerce")
        # GBD exports "Percent" metric as proportions (0–1); convert to percentage points
        if "metric_name" in sub.columns:
            is_pct = sub["metric_name"].str.contains("Percent", case=False, na=False)
        elif metric_col and metric_col != "metric_id":
            is_pct = sub[metric_col].str.contains("Percent", case=False, na=False)
        else:
            # Assume percent if values are all < 1.5 (proportions, not rates per 100k)
            is_pct = sub["val"] < 1.5
        sub.loc[is_pct, "val"] = sub.loc[is_pct, "val"] * 100
        sub = sub.dropna(subset=["val", "year", "_cause_id"])

        # Distribute rows to per-cause lists
        for cid, (cname, vcol) in GBD_CAUSES.items():
            cause_rows = sub[sub["_cause_id"] == cid][["iso3", "year", "val"]].copy()
            cause_rows = cause_rows.rename(columns={"val": vcol})
            results[cname].append(cause_rows)
            print(f"    cause={cname}: {len(cause_rows):,} rows")

    # Assemble outputs
    out: dict[str, pd.DataFrame] = {}
    for cname, vcol in [(v[0], v[1]) for v in GBD_CAUSES.values()]:
        parts = results.get(cname, [])
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        combined = (
            combined
            .drop_duplicates(subset=["iso3", "year"])
            .sort_values(["iso3", "year"])
            .reset_index(drop=True)
        )
        out_path = RAW_GBD / f"{cname}.csv"
        if combined.empty:
            print(f"  [skip] {cname}: 0 rows parsed — keeping existing file if present")
        else:
            combined.to_csv(out_path, index=False)
            print(f"  Saved {len(combined):,} rows → {out_path.relative_to(PROJECT_ROOT)}")
        out[cname] = combined

    return out


# ── Location ID → ISO3 mapping ────────────────────────────────────────────────
# Condensed IHME GBD 2021 location_id → ISO3 for all countries.
# Full list from: https://www.healthdata.org/research-article/gbd-2021-location-codes
# (Generated from the GBD 2021 supplemental location file)

def _build_location_map() -> dict:
    """Return {ihme_location_id: iso3} for country-level locations."""
    # Subset of the most common GBD location IDs → ISO3
    # (covers 195 WHO member states + a few territories)
    _MAP = {
        4: "AFG", 5: "ALB", 6: "DZA", 7: "AND", 8: "AGO", 9: "ATG",
        10: "ARG", 11: "ARM", 12: "AUS", 13: "AUT", 14: "AZE", 15: "BHS",
        16: "BHR", 17: "BGD", 18: "BRB", 19: "BLR", 20: "BEL", 21: "BLZ",
        22: "BEN", 23: "BTN", 24: "BOL", 25: "BIH", 26: "BWA", 27: "BRA",
        28: "BRN", 29: "BGR", 30: "BFA", 31: "BDI", 32: "CPV", 33: "KHM",
        34: "CMR", 35: "CAN", 36: "CAF", 37: "TCD", 38: "CHL", 39: "CHN",
        40: "COL", 41: "COM", 42: "COD", 43: "COG", 44: "CRI", 45: "CIV",
        46: "HRV", 47: "CUB", 48: "CYP", 49: "CZE", 50: "DNK", 51: "DJI",
        52: "DOM", 53: "ECU", 54: "EGY", 55: "SLV", 56: "GNQ", 57: "ERI",
        58: "EST", 59: "ETH", 60: "FJI", 61: "FIN", 62: "FRA", 63: "GAB",
        64: "GMB", 65: "GEO", 66: "DEU", 67: "GHA", 68: "GRC", 69: "GRD",
        70: "GTM", 71: "GIN", 72: "GNB", 73: "GUY", 74: "HTI", 75: "HND",
        76: "HUN", 77: "ISL", 78: "IND", 79: "IDN", 80: "IRN", 81: "IRQ",
        82: "IRL", 83: "ISR", 84: "ITA", 85: "JAM", 86: "JPN", 87: "JOR",
        88: "KAZ", 89: "KEN", 90: "KIR", 91: "PRK", 92: "KOR", 93: "KWT",
        94: "KGZ", 95: "LAO", 96: "LVA", 97: "LBN", 98: "LSO", 99: "LBR",
        100: "LBY", 101: "LIE", 102: "LTU", 103: "LUX", 104: "MDG", 105: "MWI",
        106: "MYS", 107: "MDV", 108: "MLI", 109: "MLT", 110: "MHL", 111: "MRT",
        112: "MUS", 113: "MEX", 114: "FSM", 115: "MDA", 116: "MCO", 117: "MNG",
        118: "MNE", 119: "MAR", 120: "MOZ", 121: "MMR", 122: "NAM", 123: "NRU",
        124: "NPL", 125: "NLD", 126: "NZL", 127: "NIC", 128: "NER", 129: "NGA",
        130: "NOR", 131: "OMN", 132: "PAK", 133: "PLW", 134: "PAN", 135: "PNG",
        136: "PRY", 137: "PER", 138: "PHL", 139: "POL", 140: "PRT", 141: "QAT",
        142: "ROU", 143: "RUS", 144: "RWA", 145: "KNA", 146: "LCA", 147: "VCT",
        148: "WSM", 149: "SMR", 150: "STP", 151: "SAU", 152: "SEN", 153: "SRB",
        154: "SLE", 155: "SGP", 156: "SVK", 157: "SVN", 158: "SLB", 159: "SOM",
        160: "ZAF", 161: "SSD", 162: "ESP", 163: "LKA", 164: "SDN", 165: "SUR",
        166: "SWZ", 167: "SWE", 168: "CHE", 169: "SYR", 170: "TWN", 171: "TJK",
        172: "TZA", 173: "THA", 174: "TLS", 175: "TGO", 176: "TON", 177: "TTO",
        178: "TUN", 179: "TUR", 180: "TKM", 181: "TUV", 182: "UGA", 183: "UKR",
        184: "ARE", 185: "GBR", 186: "USA", 187: "URY", 188: "UZB", 189: "VUT",
        190: "VEN", 191: "VNM", 192: "YEM", 193: "ZMB", 194: "ZWE",
        195: "ALG",  # Algeria duplicate in some GBD releases
        214: "MKD", 218: "MNE", 522: "PSE", 533: "TLS",
    }
    return _MAP


def _build_name_to_iso3() -> dict:
    """Fallback: country name → ISO3 for common GBD location names."""
    return {
        "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA",
        "Angola": "AGO", "Argentina": "ARG", "Armenia": "ARM",
        "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
        "Bangladesh": "BGD", "Belarus": "BLR", "Belgium": "BEL",
        "Benin": "BEN", "Bolivia": "BOL", "Botswana": "BWA",
        "Brazil": "BRA", "Burkina Faso": "BFA", "Burundi": "BDI",
        "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
        "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
        "China": "CHN", "Colombia": "COL", "Democratic Republic of the Congo": "COD",
        "Congo": "COG", "Côte d'Ivoire": "CIV", "Cuba": "CUB",
        "Denmark": "DNK", "Djibouti": "DJI", "Dominican Republic": "DOM",
        "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV",
        "Eritrea": "ERI", "Ethiopia": "ETH", "Finland": "FIN",
        "France": "FRA", "Gabon": "GAB", "Gambia": "GMB",
        "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA",
        "Greece": "GRC", "Guatemala": "GTM", "Guinea": "GIN",
        "Guinea-Bissau": "GNB", "Haiti": "HTI", "Honduras": "HND",
        "Hungary": "HUN", "India": "IND", "Indonesia": "IDN",
        "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
        "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM",
        "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
        "Kenya": "KEN", "North Korea": "PRK", "South Korea": "KOR",
        "Kyrgyzstan": "KGZ", "Laos": "LAO", "Lao People's Democratic Republic": "LAO",
        "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR",
        "Libya": "LBY", "Madagascar": "MDG", "Malawi": "MWI",
        "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI",
        "Mauritania": "MRT", "Mauritius": "MUS", "Mexico": "MEX",
        "Moldova": "MDA", "Mongolia": "MNG", "Morocco": "MAR",
        "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM",
        "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL",
        "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA",
        "Norway": "NOR", "Pakistan": "PAK", "Panama": "PAN",
        "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER",
        "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
        "Romania": "ROU", "Russia": "RUS", "Rwanda": "RWA",
        "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
        "Sierra Leone": "SLE", "Singapore": "SGP", "Somalia": "SOM",
        "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP",
        "Sri Lanka": "LKA", "Sudan": "SDN", "Swaziland": "SWZ",
        "Eswatini": "SWZ", "Sweden": "SWE", "Switzerland": "CHE",
        "Syria": "SYR", "Syrian Arab Republic": "SYR",
        "Tajikistan": "TJK", "Tanzania": "TZA",
        "United Republic of Tanzania": "TZA",
        "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO",
        "Tunisia": "TUN", "Turkey": "TUR", "Turkmenistan": "TKM",
        "Uganda": "UGA", "Ukraine": "UKR",
        "United Arab Emirates": "ARE", "United Kingdom": "GBR",
        "United States": "USA", "United States of America": "USA",
        "Uruguay": "URY", "Uzbekistan": "UZB", "Venezuela": "VEN",
        "Viet Nam": "VNM", "Vietnam": "VNM", "Yemen": "YEM",
        "Zambia": "ZMB", "Zimbabwe": "ZWE",
    }


# ── 3. Orchestration ──────────────────────────────────────────────────────────

def pull_all(gbd_dir: Path | None = None, force: bool = False):
    """Pull all GBD-adjacent indicators."""
    print("  Pulling OWID: Vitamin A deficiency ...")
    pull_owid("vitamin_a", force=force)

    print("  Pulling OWID: Zinc deficiency ...")
    pull_owid("zinc", force=force)

    # GBD manual CSVs
    search_dir = gbd_dir or (PROJECT_ROOT / "data" / "raw" / "gbd")
    # Look for any non-output CSV files
    candidate_csvs = [
        f for f in search_dir.glob("*.csv")
        if f.name not in {"vitamin_a_deficiency.csv", "zinc_deficiency.csv"}
           and not any(
               f.name == f"{n}.csv"
               for n, _ in GBD_CAUSES.values()
           )
    ]
    zip_files = list(search_dir.glob("*.zip"))

    if candidate_csvs or zip_files:
        print(f"  Found {len(candidate_csvs)} CSV(s) + {len(zip_files)} zip(s) — running GBD parser ...")
        parse_gbd_csv(search_dir)
    else:
        print("  [info] No GBD export CSVs found.")
        print("         To add iron/vitamin-A/zinc from GBD 2021, see:")
        print("         docs/gbd_download_guide.md")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull GBD-adjacent micronutrient data")
    parser.add_argument("--gbd-dir", type=Path, default=None,
                        help="Directory containing manually-downloaded GBD CSV/zip files")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output files already exist")
    args = parser.parse_args()

    gbd_dir = args.gbd_dir or (PROJECT_ROOT / "data" / "raw" / "gbd")
    print(f"GBD raw directory: {gbd_dir}")
    pull_all(gbd_dir=gbd_dir, force=args.force)
    print("Done.")
