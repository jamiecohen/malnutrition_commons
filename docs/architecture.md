# Malnutrition Data Commons — Architecture

**Version**: 0.1 (Sprint C preview)  
**Author**: Jamie Cohen  
**Date**: April 2026

---

## Overview

The Malnutrition Data Commons is a harmonized data infrastructure that enables cross-portfolio analysis of nutritional burden, infectious disease co-occurrence, and intervention coverage. It is designed to answer questions that currently require manual assembly of heterogeneous datasets from multiple sources.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│  GBD/IHME    WHO GHO    DHS    FAO    UNICEF    World Bank      │
└──────┬──────────┬────────┬──────┬──────────┬─────────┬─────────┘
       │          │        │      │          │         │
       ▼          ▼        ▼      ▼          ▼         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HARMONIZATION LAYER                          │
│                                                                 │
│  • Standardize to ISO3 country codes                            │
│  • Align to common geographic hierarchy (GBD locations)         │
│  • Normalize to calendar year (handle survey years)             │
│  • Standardize age groups (under 5, reproductive age, all ages) │
│  • Resolve indicator naming conflicts                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INDICATOR FRAMEWORK                          │
│                                                                 │
│  MICRONUTRIENT BURDEN          INFECTIOUS DISEASE BURDEN        │
│  ├─ Iron deficiency prev.      ├─ TB incidence                  │
│  ├─ Anemia prevalence          ├─ Malaria incidence             │
│  ├─ Vitamin A deficiency       └─ Diarrheal disease burden      │
│  ├─ Zinc deficiency                                             │
│  └─ Iodine deficiency          INTERVENTION COVERAGE            │
│                                ├─ LSFF coverage (wheat/maize)   │
│  CHILD NUTRITION               ├─ MMS coverage                  │
│  ├─ Stunting prevalence        ├─ Vitamin A supplementation     │
│  ├─ Wasting prevalence         └─ ORS/zinc treatment            │
│  └─ Underweight prevalence                                      │
│                                DIET / FOOD ENVIRONMENT          │
│  MATERNAL NUTRITION            ├─ Prevalence of undernourishment│
│  ├─ Maternal anemia            ├─ Cost of healthy diet          │
│  └─ Low BMI at conception      └─ Food supply adequacy          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY LAYER                               │
│                                                                 │
│  • Filter by: geography, year, age group, sex, indicator        │
│  • Overlay: co-occurrence queries (burden A × burden B)         │
│  • Rank: countries by composite burden or coverage gap          │
│  • Export: CSV, GeoJSON, or Plotly-compatible formats           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION / OUTPUTS                      │
│                                                                 │
│  • Interactive dashboard (Streamlit)                            │
│  • Choropleth maps (geographic overlays)                        │
│  • Scatter overlays (burden A vs burden B by country)           │
│  • Time series (trend analysis)                                 │
│  • Export-ready static figures (for slides)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Geographic Unit: Country-Level (Phase 1)
The Phase 1 commons operates at the country level. Sub-national data (DHS admin-2, GBD subnational) is a Phase 2 addition. Country-level harmonization is sufficient to demonstrate the value of the integrated approach for the June preview.

### Temporal Alignment
Different sources report for different years (survey vs. modeled vs. administrative). The harmonization layer:
1. Prefers modeled annual estimates where available (GBD, WHO GHO)
2. Propagates survey-year data with explicit year flags for DHS
3. Uses the most recent available estimate within a configurable recency window (default: 5 years)

### Indicator Prioritization for June Preview
For the June 2026 Learning Session, three indicator groups are prioritized:
1. **Iron deficiency + anemia** (micronutrient burden, links to maternal model in Sprint A/B)
2. **TB incidence** (infectious disease burden, links to IDM TB portfolio)
3. **LSFF coverage proxy** (intervention coverage, Nutrition PST investment)

These three dimensions are the minimum needed to demonstrate the geographic co-occurrence analysis.

---

## Build Timeline (Full Commons)

| Phase | Timeline | Owner | Deliverable |
|-------|----------|-------|-------------|
| Phase 0 (Preview) | April–June 2026 | Jamie + Emma? | Architecture doc, 1–2 illustrative visuals, dashboard shell |
| Phase 1 (Core Build) | June–September 2026 | Emma (LTE) | All priority indicators harmonized, dashboard v1, country profiles |
| Phase 2 (Extension) | September–December 2026 | Emma + IDM | Sub-national data, remnant sample integration, dynamic query interface |

---

## Open Questions (for LTE onboarding)

1. Should the harmonized data be stored as flat CSVs, a SQLite database, or a cloud-hosted solution? For Phase 1, flat CSVs are fine; Phase 2 may benefit from a lightweight database.
2. How should we handle the DHS microdata access workflow (registration required) vs. the aggregate indicators that are freely downloadable?
3. What is the right geographic hierarchy — GBD locations, UN M.49, or World Bank regions — for the cross-portfolio analysis?
