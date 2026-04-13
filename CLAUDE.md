# CLAUDE.md — Malnutrition Data Commons

## Project Purpose

This repo implements the **Sprint C: Malnutrition Data Commons Preview** for the June 2026 IDM Learning Session at the Bill & Melinda Gates Foundation.

The goal is to demonstrate the architecture and value of a harmonized malnutrition data commons — pulling from publicly available sources (GBD, DHS, WHO GHO, FAO, UNICEF) and producing illustrative geographic overlays and an interactive dashboard. This is a **preview/proof of concept**, not a production system.

---

## Key Context

- **Audience**: Foundation learning session attendees (MNCNH, Nutrition PST, IDM leadership)
- **Framing**: Credible, resourced plan — "here's what we're building" rather than "here's the finished product"
- **Critical visual**: A geographic overlay showing co-occurrence of micronutrient deficiency (iron), infectious disease burden (TB), and LSFF coverage for 1–2 priority regions
- **Tone**: Vivid enough to show the value of the approach; honest about what's a preview

## What to Show

1. Architecture diagram (data sources → harmonization → indicator framework → query layer)
2. 1–2 illustrative geographic overlays — the "what this will look like" visual
3. Dashboard with filters by geography, indicator, and year
4. Clean data pipeline showing how sources are harmonized

---

## Tech Stack

- **Language**: Python
- **Dashboard**: Streamlit
- **Visualization**: Plotly (interactive), Matplotlib/Seaborn (static for slides)
- **Geospatial**: Geopandas, Folium (choropleth maps)
- **Data processing**: Pandas, requests (API pulls)

---

## Data Sources and APIs

### WHO GHO REST API
- Base URL: `https://ghoapi.azureedge.net/api/`
- Key indicators: anemia (`NUTRITION_ANAEMIA_CHILDREN_PREV`), stunting, wasting, TB incidence (`TB_e_inc_100k`)
- Docs: https://www.who.int/data/gho/info/gho-odata-api

### IHME GBD
- Use pre-downloaded CSV exports from IHME GHD Results Tool for:
  - Iron deficiency (`rei_id=96`), vitamin A deficiency, zinc deficiency
  - Child wasting, child stunting
- Download via: http://ghdx.healthdata.org/gbd-results-tool

### FAO FAOSTAT API
- Base URL: `http://www.fao.org/faostat/api/v1/en/`
- Key indicators: prevalence of undernourishment, food supply adequacy, CoHD

### UNICEF Data Warehouse
- Stunting, wasting, underweight: https://data.unicef.org/resources/dataset/malnutrition-data/
- Use direct CSV downloads

### World Bank
- LSFF proxy: flour fortification coverage, food fortification program data
- API: `https://api.worldbank.org/v2/`

---

## Geographic Focus

Priority geographies for illustrative overlays (per Foundation portfolio):
- **South Asia**: Pakistan, Bangladesh, India (MNCNH alignment, MUMTA cohort)
- **Sub-Saharan Africa**: Nigeria, Ethiopia, DRC (high burden, TB co-occurrence)
- Use ISO3 country codes throughout; standardize to GBD location hierarchy

---

## Code Conventions

- Scripts in `src/data/` handle downloads; scripts in `src/viz/` handle plotting
- All raw data goes in `data/raw/` (gitignored for large files); processed data in `data/processed/`
- Use snake_case for files and variables
- Keep data pull scripts idempotent (skip download if file exists)
- Dashboard (`dashboard/app.py`) must run with a single `streamlit run` command

---

## Important Constraints

- **This is a preview**: Visualizations should be illustrative, not exhaustive. 1–2 well-chosen overlays beat 10 mediocre ones.
- **Public data only**: All data sources must be publicly accessible — no proprietary Foundation data in this repo
- **Reproducible**: Anyone should be able to clone this repo and reproduce all outputs
- **June deadline**: Keep scope tight. Ship something vivid and credible, not comprehensive.
