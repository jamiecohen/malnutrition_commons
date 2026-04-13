# Malnutrition Data Commons

A harmonized, queryable data landscape bringing together micronutrient deficiency burden, infectious disease prevalence, intervention coverage, and diet affordability data across priority geographies — enabling geographic targeting, burden characterization, and integrated analysis across the Foundation's nutrition investments.

> **June 2026 Learning Session**: This repository contains the Sprint C preview — illustrating the architecture, sample data pulls, and visualizations that demonstrate what the full commons will look like when built out. The full build is a 3–6 month effort led by the LTE hire.

---

## Vision

The malnutrition commons is a cross-portfolio data backbone that answers questions no single dataset can answer alone:

- Where do micronutrient deficiencies co-occur with high infectious disease burden?
- What is the geographic overlap between populations targeted by LSFF programs and MNCNH/TB/malaria portfolios?
- How does the integrated burden landscape shift under plausible intervention scenarios?

By harmonizing data from GBD, DHS, WHO GHO, FAO, and UNICEF into a common geographic and temporal framework, the commons enables the kind of cross-cutting analysis that motivates integrated IDM investment.

---

## Repository Structure

```
malnutrition_commons/
├── data/
│   ├── raw/            # Downloaded source data (not committed — see sources.md)
│   ├── processed/      # Harmonized, analysis-ready datasets
│   └── sources.md      # Data source inventory and download instructions
├── docs/
│   └── architecture.md # Full data commons architecture and indicator framework
├── src/
│   ├── data/           # Data download and processing scripts
│   └── viz/            # Visualization utilities
├── dashboard/
│   └── app.py          # Streamlit interactive dashboard
├── notebooks/          # Exploratory analysis
└── requirements.txt
```

---

## Data Sources

| Source | Content | Access |
|--------|---------|--------|
| IHME GBD | Micronutrient deficiency burden (iron, zinc, vitamin A, iodine), wasting, stunting by country/year | Public API + bulk download |
| WHO GHO | Anemia prevalence, undernutrition indicators, TB burden | Public REST API |
| DHS Program | Household-level nutrition, infant feeding, anthropometry | Public data portal |
| FAO | Food security, diet affordability (CoHD), food supply | Public API |
| UNICEF | Child stunting, wasting, underweight | Data warehouse API |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pull illustrative datasets
python src/data/pull_data.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Sprint C Scope (June 2026 Learning Session)

Sprint C is a **preview** of the full commons — not a finished product. For June, we are showing:

1. **Architecture and data specification** — what we're building and why
2. **Illustrative visualizations** — sample geographic overlays showing the value of the integrated approach (e.g., iron deficiency prevalence × TB burden × LSFF coverage for priority regions)
3. **Resourcing plan** — LTE hire, 6–9 month build timeline, Fall 2026 strategic review targets

The full build will be LTE-led after the June session.

---

## Context

This work is part of a broader IDM strategy for the Foundation's nutrition portfolio, developed in coordination with the MNCNH and Nutrition PSTs. See `docs/architecture.md` for the full data commons design and indicator framework.

**Contacts**: Jamie Cohen (IDM), Emma Bonglack (LTE, incoming)
