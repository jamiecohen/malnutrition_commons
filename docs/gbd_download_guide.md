# GBD Data Download Guide

This guide explains how to manually export prevalence data from the IHME GBD
Results Tool and feed it into the Malnutrition Commons pipeline.

> **Why manual?** IHME does not provide a public, unauthenticated API for GBD
> data. Downloads require a free IHME account. Once downloaded, the CSV parser
> in `src/data/pull_gbd.py` handles the rest automatically.

---

## What you need to download

### Export 1 — Micronutrient deficiencies (GBD Estimate: Cause of death and injury)

| Indicator | GBD Cause Name | cause_id |
|---|---|---|
| Iron deficiency | Iron deficiency | 390 |
| Vitamin A deficiency | Vitamin A deficiency | 389 |
| Zinc deficiency | Zinc deficiency | 391 |
| Iodine deficiency | Iodine deficiency | 387 |

### Export 2 — Birth outcomes (GBD Estimate: Cause of death and injury)

| Indicator | GBD Cause Name | cause_id | Notes |
|---|---|---|---|
| Small for gestational age | Short gestation and low birth weight | 341 | Includes SGA + preterm LBW jointly |

> **Note on SGA**: WHO GHO provides preterm birth rate and low birthweight
> programmatically (pulled automatically by `pull_who_gho.py`). GBD is only
> needed for SGA, which has no public API equivalent.

You can combine both exports into a single download by selecting all five causes
at once (see Step 3 below).

---

## Step-by-step instructions

### 1. Create a free IHME account (if you don't have one)

Go to: https://www.healthdata.org/about/register  
Registration is free and instant. No affiliation required.

---

### 2. Open the GBD Results Tool

URL: **https://vizhub.healthdata.org/gbd-results/**

---

### 3. Configure your query

Set the following filters in the left panel:

| Field | Value |
|---|---|
| **GBD Cycle** | GBD 2021 |
| **Measure** | Prevalence |
| **Metric** | Percent |
| **Year** | 2000 to 2021 (select all) |
| **Location** | Select All → then deselect "Global", "Super-regions", "Regions" (keep countries only) |
| **Age** | `<5 years` AND `All Ages` |
| **Sex** | Both |
| **Cause** | See below |

**Causes to select:**
- Under **Nutritional deficiencies**: Iron deficiency, Vitamin A deficiency, Zinc deficiency, Iodine deficiency
- Under **Neonatal disorders**: Short gestation and low birth weight *(this is the SGA proxy)*

**Tip**: Use the cause search box to find each one quickly rather than navigating the full hierarchy.

---

### 4. Download

Click **Download** (top right). The tool exports a zip file named something like:
```
IHME-GBD_2021_DATA-<hash>.zip
```

The zip typically contains one or more CSV files:
- `IHME-GBD_2021_DATA-<hash>-1.csv`
- `IHME-GBD_2021_DATA-<hash>-2.csv` (if large)

Each CSV has these columns:
```
measure_id, measure_name, location_id, location_name,
sex_id, sex_name, age_id, age_name,
cause_id, cause_name, metric_id, metric_name,
year, val, upper, lower
```

---

### 5. Place the file

Move the downloaded **zip file** (do not unzip it manually) to:
```
data/raw/gbd/
```

Example:
```bash
mv ~/Downloads/IHME-GBD_2021_DATA-abc123.zip \
   /path/to/malnutrition_commons/data/raw/gbd/
```

---

### 6. Run the parser

```bash
# From repo root, with venv activated:
python src/data/pull_gbd.py
```

The script will:
1. Unpack the zip
2. Filter rows to Prevalence / Percent / Both sexes / under-5 preferred
3. Map IHME location_ids to ISO3 codes
4. Write one clean CSV per indicator to `data/raw/gbd/`:
   - `iron_deficiency.csv`
   - `vitamin_a_deficiency.csv`  *(supplements OWID version)*
   - `zinc_deficiency.csv`  *(supplements OWID version)*
   - `iodine_deficiency.csv`
   - `sga_prevalence.csv`  *(Short gestation / low birth weight, SGA proxy)*

Then re-run the harmonization pipeline to incorporate the new indicators:
```bash
python src/data/harmonize.py
```

---

## Verification

After running, check row counts:
```bash
python -c "
import pandas as pd
for f in ['iron_deficiency','vitamin_a_deficiency','zinc_deficiency','iodine_deficiency','sga_prevalence']:
    try:
        df = pd.read_csv(f'data/raw/gbd/{f}.csv')
        print(f'{f}: {len(df):,} rows, {df.iso3.nunique()} countries, years {df.year.min()}-{df.year.max()}')
    except FileNotFoundError:
        print(f'{f}: not found')
"
```

Expected output (GBD 2021, all countries, 2000–2021):
```
iron_deficiency: ~3,800 rows, 195 countries, years 2000-2021
vitamin_a_deficiency: ~3,800 rows, 195 countries, years 2000-2021
zinc_deficiency: ~3,800 rows, 195 countries, years 2000-2021
iodine_deficiency: ~3,800 rows, 195 countries, years 2000-2021
```

---

## Notes on data sources

- **OWID versions** (Vitamin A, Zinc) are pulled automatically without any login
  and are appropriate for most cross-country comparisons. They reflect earlier
  estimates (WHO 2017 / Wessells 2012).

- **GBD 2021 versions** provide consistent methodology, annual time series
  2000–2021, and uncertainty intervals — preferred for the final commons.

- **Iron deficiency**: Only available via GBD. There is no equivalent publicly
  accessible time series from OWID or WHO GHO as of 2025.

- The parser prefers **under-5 age group** over "All Ages" when both are present,
  as under-5 is the primary analytic focus for child malnutrition alignment.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Export zip is very large (>100 MB) | You may have selected too many locations or ages. Use "Countries and territories" instead of "All locations". |
| `iso3` column is mostly NaN after parsing | The parser uses IHME location_id → ISO3 map. Check `src/data/pull_gbd.py: _build_location_map()` for any missing IDs. |
| `No matching rows after filters` | Check that the CSV contains Prevalence + Percent rows. Some GBD exports default to "Number" metric — re-export with "Percent" selected. |
| Wrong column names | GBD exports from different years use slightly different column headers. The parser handles both `cause_id` and `cause_name` columns. File an issue if a new column name pattern is encountered. |
