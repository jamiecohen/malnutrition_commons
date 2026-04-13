# Data Sources

This document inventories all data sources used in the Malnutrition Data Commons preview.

---

## WHO Global Health Observatory (GHO)

**Access**: Public REST API — no authentication required  
**Base URL**: `https://ghoapi.azureedge.net/api/`  
**Script**: `src/data/pull_who_gho.py`

| Indicator Code | Description | Unit |
|---|---|---|
| `NUTRITION_ANAEMIA_CHILDREN_PREV` | Anaemia in children under 5, prevalence | % |
| `NUTRITION_ANAEMIA_WOMEN_PREV` | Anaemia in women of reproductive age | % |
| `NCD_BMI_18A` | Adult overweight (proxy for diet transition) | % |
| `TB_e_inc_100k` | TB incidence per 100,000 population | rate |
| `WHS9_93` | Prevalence of stunting (children <5) | % |
| `WHS9_96` | Prevalence of wasting (children <5) | % |
| `WHS9_97` | Prevalence of underweight (children <5) | % |

---

## IHME Global Burden of Disease (GBD)

**Access**: Public bulk downloads via GHDx Results Tool  
**URL**: http://ghdx.healthdata.org/gbd-results-tool  
**Script**: `src/data/pull_gbd.py` (parses pre-downloaded exports)

Key measures:
- Iron deficiency: prevalence and DALYs by country, age, sex, year
- Vitamin A deficiency
- Zinc deficiency
- Protein-energy malnutrition

Download parameters: GBD 2021, measure = Prevalence, metric = Rate, location = All countries, age = All ages / Under 5, year = 2000–2021

---

## UNICEF Malnutrition Data

**Access**: Direct CSV download  
**URL**: https://data.unicef.org/resources/dataset/malnutrition-data/  
**File**: `data/raw/unicef_malnutrition.csv`

Indicators: stunting, wasting, severe wasting, overweight, underweight (children under 5)  
Coverage: ~140 countries, most recent survey year

---

## FAO FAOSTAT

**Access**: Public REST API  
**Base URL**: `http://www.fao.org/faostat/api/v1/en/`  
**Script**: `src/data/pull_fao.py`

Key datasets:
- Suite of Food Security Indicators (includes PoU — prevalence of undernourishment)
- Food Balances (food supply per capita)

---

## World Bank — Food Fortification

**Access**: Public REST API  
**URL**: `https://api.worldbank.org/v2/`

Used as proxy for LSFF (large-scale food fortification) coverage where dedicated data unavailable. Supplemented with FFI (Food Fortification Initiative) country data where available.

---

## Natural Earth (Geospatial Boundaries)

**Access**: Public, bundled with geopandas  
**Resolution**: 1:110m (for overview maps), 1:50m (for regional detail)

Used for country-level choropleth maps.
