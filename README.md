# Malnutrition Data Commons

A harmonized, queryable data landscape bringing together micronutrient deficiency burden, infectious disease prevalence, intervention coverage, birth outcomes, and economic context data across 226 countries — enabling geographic targeting, burden characterization, and integrated analysis across the Foundation's nutrition investments.

> **June 2026 Learning Session**: This repository demonstrates the Sprint C preview — a working data pipeline, 33 harmonized indicators across 226 countries, and hypothesis-driven cross-indicator analysis illustrating the value of the integrated commons approach. The full production build is a 3–6 month effort led by the LTE hire.

---

## What's Been Built

### Data pipeline
Eight source-specific pull/process scripts ingest and harmonize data from public APIs, bulk downloads, and cohort datasets into analysis-ready formats. Running `python src/data/pull_data.py` reproduces the global dataset from scratch; cohort data is processed separately via `process_mumta.py` and `process_amanhi.py`.

### Harmonized datasets
- **`data/processed/commons_snapshot.csv`** — 226 countries × 76 columns; one row per country using the most recent available value (2010–2023 window, with fallback to best available year)
- **`data/processed/commons_panel.csv`** — 46,011 country–year rows × 44 columns (1960–2024)
- **`data/processed/population.csv`** — World Bank population estimates for rate normalization
- **`data/processed/mumta/`** — 7 CSVs from MUMTA cohort (cohort summary, infant growth, maternal anemia, B. infantis, gut inflammation, microbiome, TAC pathogens)
- **`data/processed/amanhi/`** — 3 CSVs from AMANHI cohort (neonatal B. infantis/B. longum + outcomes, maternal bioanalytes, maternal TAC pathogens)

### Indicators (33 numeric, 7 domains)

| Domain | Indicators |
|--------|-----------|
| **Nutritional status** | Anaemia in children <5, pregnant women, women 15–49; stunting, wasting, underweight |
| **Micronutrient deficiencies** | Iron, vitamin A, zinc, iodine deficiency prevalence (GBD/OWID) |
| **Infectious disease** | TB incidence, HIV prevalence, malaria incidence, measles reported cases |
| **Birth outcomes** | Low birthweight, preterm birth rate, small for gestational age (SGA) |
| **Healthcare coverage** | ANC4+, MCV1, MCV2, DTP3, PCV3, RotaC, ORS treatment |
| **Mortality** | Under-5 mortality, neonatal mortality, maternal mortality ratio |
| **Human capital & food systems** | HCI score, HCI learning years, GDP per capita PPP, severe food insecurity, moderate+severe food insecurity, LSFF wheat flour coverage |

### Cross-indicator analysis
11 hypotheses tested across the Integrated Nutrition Impact Framework causal chain, from upstream food systems drivers through nutritional status, disease burden, and downstream outcomes (mortality, human capital, economic productivity). See `docs/insights_summary.md` for full results.

---

## Repository Structure

```
malnutrition_commons/
├── data/
│   ├── raw/                    # Downloaded source data (gitignored for large files)
│   │   ├── who_gho/            # WHO GHO OData API pulls
│   │   ├── unicef/             # UNICEF/JME malnutrition CSVs
│   │   ├── fao/                # FAO FAOSTAT API pulls
│   │   ├── lsff/               # FFI 2023 wheat flour fortification
│   │   ├── gbd/                # IHME GBD (OWID programmatic + manual download)
│   │   ├── outcomes/           # World Bank outcome indicators
│   │   ├── MUMTA/              # MUMTA Pakistan cohort (RCT, xlsx)
│   │   └── AMANHI/             # AMANHI 3-country cohort (qPCR, bioanalytes, TAC, microbiome)
│   ├── processed/              # Harmonized, analysis-ready datasets
│   │   ├── mumta/              # 7 MUMTA analysis CSVs
│   │   └── amanhi/             # 3 AMANHI analysis CSVs
│   └── sources.md              # Data source inventory and download instructions
├── docs/
│   ├── architecture.md         # Full data commons architecture and indicator framework
│   ├── gbd_download_guide.md   # Step-by-step GBD Results Tool export instructions
│   └── insights_summary.md     # Summary of 11 cross-indicator hypotheses and findings
├── src/
│   ├── data/
│   │   ├── pull_data.py        # Master pull script (runs all sources)
│   │   ├── pull_who_gho.py     # WHO GHO REST API
│   │   ├── pull_unicef.py      # UNICEF/JME malnutrition
│   │   ├── pull_fao.py         # FAO food security
│   │   ├── pull_lsff.py        # LSFF/FFI fortification coverage
│   │   ├── pull_gbd.py         # GBD micronutrient deficiencies (OWID + manual)
│   │   ├── pull_outcomes.py    # World Bank outcomes and context
│   │   ├── pull_dhs_subnational.py  # Nigeria state-level DHS data
│   │   ├── process_mumta.py    # MUMTA cohort raw data → 7 analysis CSVs
│   │   ├── process_amanhi.py   # AMANHI cohort raw data → 3 analysis CSVs
│   │   └── harmonize.py        # Merges all sources into snapshot + panel
│   └── viz/
│       ├── figures.py          # Core chart functions (choropleth, scatter, bar, trend)
│       ├── insights.py         # Hypothesis-driven cross-indicator analysis (H1–H11)
│       ├── mumta.py            # MUMTA cohort visualizations (18 chart functions)
│       ├── amanhi.py           # AMANHI cohort visualizations (18 chart functions)
│       └── product_impact.py   # Product-level DALY/cost modelling
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── outputs/
│   └── slides/insights/        # Saved hypothesis figures (HTML + PNG)
└── requirements.txt
```

---

## Data Sources

| Source | Content | Access |
|--------|---------|--------|
| **WHO GHO** | Anaemia, stunting, wasting, underweight, TB, HIV, malaria, birth outcomes, vaccination coverage, ORS treatment | Public REST API (`ghoapi.azureedge.net`) |
| **UNICEF / JME** | Child stunting, wasting, underweight (JME harmonized estimates) | Public CSV download |
| **FAO FAOSTAT** | Food insecurity prevalence, diet affordability | Public REST API |
| **FFI / LSFF** | Wheat flour fortification coverage (2023) | Public spreadsheet |
| **IHME GBD / OWID** | Iron, vitamin A, zinc, iodine deficiency; SGA prevalence | OWID programmatic + manual GBD download (see `docs/gbd_download_guide.md`) |
| **World Bank** | Under-5/neonatal/maternal mortality, HCI, GDP per capita, food insecurity | Public REST API |
| **MUMTA Cohort** | Pakistan RCT (n=1,884): birth outcomes, maternal anemia, infant growth, B. infantis qPCR, TAC enteropathogens, fecal MPO/LCN2 | Private cohort data (xlsx) |
| **AMANHI Cohort** | 3-country observational (PAK/BGD/TZA, n=729 neonates): B. infantis/B. longum qPCR, serum CRP & ferritin, maternal TAC, taxonomic microbiome profiles | WHO AMANHI study data |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/jamiecohen/malnutrition_commons.git
cd malnutrition_commons
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download supporting data

The processed datasets ship with the repo, so the dashboard works immediately. To refresh data from source or pull additional datasets:

```bash
# Pull Nigeria state-level data for the subnational tab (DHS API)
python src/data/pull_dhs_subnational.py

# (Optional) Re-process MUMTA cohort data (requires raw xlsx in data/raw/MUMTA/)
python src/data/process_mumta.py --force

# (Optional) Re-process AMANHI cohort data (requires raw xlsx in data/raw/AMANHI/)
python src/data/process_amanhi.py --force

# (Optional) Re-pull all global source data from scratch (skips existing files)
python src/data/pull_data.py

# (Optional) Re-harmonize into analysis-ready datasets
python src/data/harmonize.py
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

### Dashboard tabs

| Tab | What's in it |
|-----|-------------|
| **Architecture & Sources** | Data pipeline diagram, two-tier architecture (global + cohort), source documentation |
| **Priority Geographies** | Pakistan, India, Nigeria country profiles with ground-truth survey comparisons and trends |
| **Deep Dive: Pakistan** | Three-cohort view — **MUMTA** (RCT: birth outcomes, anemia, growth, B. infantis, TAC, gut inflammation), **AMANHI** (observational: B. infantis/B. longum across 3 countries, CRP/ferritin bioanalytes, maternal TAC, growth trajectories), and **Cross-Cohort Comparison** (B. infantis rates, pathogen profiles) |
| **Product Impact** | DALY/cost projections for 5 nutrition products with adjustable parameters |
| **Nigeria Subnational** | State-level choropleth maps and zone analysis (NDHS 2018) |
| **Global Context** | Ecological views across 226 countries — choropleth, co-occurrence, causal pathways, country profiles |

### Troubleshooting

- **Nigeria subnational warning**: Run `python src/data/pull_dhs_subnational.py` to download the DHS data
- **`streamlit` not found**: Make sure your virtual environment is activated (`source venv/bin/activate`)
- **`geopandas` install failure on Apple Silicon**: Try `pip install pygeos` first, then re-run `pip install -r requirements.txt`

> **Note**: Iron deficiency, iodine deficiency, and SGA prevalence require a manual download from the IHME GBD Results Tool. See `docs/gbd_download_guide.md` for step-by-step instructions. Vitamin A and zinc deficiency are pulled programmatically from OWID.

---

## Key Findings (Preview)

Eleven cross-indicator hypotheses tested across 226 countries reveal strong empirical support for the Integrated Nutrition Impact Framework causal chain:

- **Vaccination gaps → measles burden**: MCV1 coverage explains 27% of log-scale measles incidence variation; MCV1→MCV2 dropout amplifies risk further (r = 0.39***)
- **ANC4 coverage → better birth outcomes**: Higher ANC4 predicts lower LBW (r = −0.47***) and preterm birth (r = −0.44***)
- **Malaria amplifies anaemia beyond iron deficiency**: Malaria–anaemia correlation (r = 0.79***) persists after statistically controlling for iron deficiency — a strong signal for co-intervention
- **HIV–TB syndemic**: HIV prevalence strongly predicts TB incidence (r = 0.50***), concentrated in Sub-Saharan Africa
- **Nutrition × health system composite**: Countries with stronger health systems carry meaningfully lower nutrition burden (r = −0.29***)
- **Pregnant anaemia → maternal mortality**: r = 0.75*** — one of the strongest single-indicator predictors of MMR
- **Stunting → child mortality**: Stunting predicts U5MR with r = 0.81***; nutrition burden composite reaches r = 0.83***
- **Nutrition burden → human capital**: Strongest relationship in the dataset: r = −0.85*** with HCI, r = −0.80*** with log(GDP per capita)
- **Food insecurity → stunting → U5MR causal chain**: All three links confirmed (r ≥ 0.77***), supporting the upstream food systems framing

See `docs/insights_summary.md` for full statistical detail, interpretation, and implications.

---

## Sprint C Scope (June 2026 Learning Session)

Sprint C is a **preview** of the full commons — demonstrating:

1. **Working data pipeline** — 8 sources (6 global + 2 cohort), 33 global indicators + cohort-level microbiome/biomarker data, reproducible from scratch
2. **Integrated cross-indicator analysis** — 11 hypotheses, empirical effect sizes across 226 countries
3. **Cohort ground-truth** — MUMTA (Pakistan RCT, n=1,884) and AMANHI (3-country observational, n=729 neonates) with B. infantis, enteropathogens, CRP/ferritin, and growth trajectory analyses
4. **Interactive dashboard** — choropleth maps, scatter plots, burden rankings, time series, cohort-level deep dives with cross-cohort comparison
5. **Architecture and resourcing plan** — what the full production build requires

The full production build (LTE-led, Fall 2026) will add DHS household microdata, near-real-time data refresh, and a formal query API layer.

---

## Context

Developed in coordination with the MNCNH and Nutrition PSTs as part of the IDM integrated data strategy. See `docs/architecture.md` for the full data commons design and indicator framework.

**Contacts**: Jamie Cohen (IDM), Emma Bonglack (LTE, incoming)
