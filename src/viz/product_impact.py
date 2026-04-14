"""
Product-level impact modelling for the Malnutrition Data Commons Scenario Planner.

Five portfolio products:
  1. LSFF         — Large-Scale Food Fortification (wheat flour, iron + folic acid)
  2. MMS          — Multiple Micronutrient Supplementation (switch from IFA)
  3. IV-Iron      — Intravenous iron for severe anaemia in pregnancy
  4. Maternal Gut — Maternal probiotic/synbiotic supplementation during pregnancy
  5. Infant Gut   — Infant gut microbiome support (e.g. B. infantis EVC001)

Two-tier parameters:
  🔒 data_params  — drawn from the commons snapshot (country-specific, locked)
  ⚙️ adj_params   — user-adjustable sliders with evidence-based defaults

Headline output metrics:
  • LBW averted              (annual, absolute count)
  • Stunted children averted (N-year cumulative, absolute count)
  • Maternal deaths averted  (annual, absolute count)

Combination logic: sequential multiplicative for shared outcomes.
  remaining_risk *= (1 − RRR_i)  applied in upstream→downstream order.

Evidence anchors:
  LSFF:         Das et al. Cochrane 2019; Bhutta et al. Lancet 2013
  MMS vs IFA:   SUMMIT meta-analysis 2022 (Bourassa et al.)
  IV-Iron:      Pavord et al. 2015; Govindappagari & Burwick 2019
  Maternal Gut: Wickens et al. 2017; Odamaki et al. 2020; emerging RCT evidence
  Infant Gut:   Frese et al. 2017; Nguyen et al. 2021; MALED consortium
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Styling ───────────────────────────────────────────────────────────────────
FOUNDATION_BLUE = "#003366"
ACCENT_ORANGE   = "#E87722"
BG_LIGHT        = "#F8F9FA"
FONT            = dict(family="Arial, sans-serif", color="#1A1A2E")

PRODUCT_COLORS = {
    "lsff":         "#2A9D8F",
    "mms":          "#E87722",
    "iv_iron":      "#CC3333",
    "maternal_gut": "#7C3AED",
    "infant_gut":   "#0369A1",
}

# ── Regional birth rates (per 1,000 population) ───────────────────────────────
_REGIONAL_BIRTH_RATES = {
    "AFR":     37.0,
    "SEAR":    20.0,
    "EMR":     26.0,
    "WPR":     14.0,
    "EUR":     10.0,
    "AMR":     16.0,
    "default": 22.0,
}


def estimate_annual_births(country_row: pd.Series, population: float) -> float:
    """Return estimated annual births from regional crude birth rate lookup."""
    region = str(country_row.get("who_region", "") or "")
    if "Africa" in region or "AFR" in region:
        rate = _REGIONAL_BIRTH_RATES["AFR"]
    elif "South-East Asia" in region or "SEAR" in region:
        rate = _REGIONAL_BIRTH_RATES["SEAR"]
    elif "Eastern Mediterranean" in region or "EMR" in region:
        rate = _REGIONAL_BIRTH_RATES["EMR"]
    elif "Western Pacific" in region or "WPR" in region:
        rate = _REGIONAL_BIRTH_RATES["WPR"]
    elif "Europe" in region or "EUR" in region:
        rate = _REGIONAL_BIRTH_RATES["EUR"]
    elif "Americas" in region or "AMR" in region:
        rate = _REGIONAL_BIRTH_RATES["AMR"]
    else:
        rate = _REGIONAL_BIRTH_RATES["default"]
    return population * rate / 1_000


# ── Product registry ──────────────────────────────────────────────────────────
PRODUCT_REGISTRY: dict[str, dict] = {

    "lsff": {
        "name":        "Large-Scale Food Fortification",
        "short":       "LSFF",
        "emoji":       "🌾",
        "description": (
            "Mandatory wheat flour fortification with iron and folic acid. "
            "Reduces iron deficiency prevalence, with downstream effects on LBW "
            "and stunting via maternal and child anaemia pathways."
        ),
        "outcomes": ["lbw", "stunting"],
        "evidence_tag": "Das et al. Cochrane 2019 · Bhutta et al. Lancet 2013",
        "tier": "data-informed",

        # 🔒 data-informed parameters (pulled from snapshot)
        "data_params": [
            dict(key="iron_deficiency_pct",      label="Iron deficiency prevalence (%)",
                 source="GBD estimate"),
            dict(key="lsff_coverage_proxy_pct",  label="Current LSFF coverage (%)",
                 source="Legislation proxy: no programme=0, voluntary=20, mandatory=75"),
        ],

        # ⚙️ adjustable parameters (sliders)
        "adj_params": [
            dict(key="lsff_target_coverage", label="Target LSFF coverage (%)",
                 default=75.0, min=0.0, max=100.0, step=5.0,
                 help="Target population coverage after scale-up."),
            dict(key="lsff_efficacy_iron", label="Efficacy: iron deficiency reduction (fraction)",
                 default=0.36, min=0.10, max=0.60, step=0.02,
                 help="Fractional reduction in iron deficiency per unit of coverage added. "
                      "Default 0.36 from Cochrane meta-analysis (Das et al. 2019)."),
            dict(key="lsff_iron_to_lbw", label="Iron def → LBW coefficient (pp/pp)",
                 default=0.10, min=0.03, max=0.22, step=0.01,
                 help="Estimated pp reduction in LBW per 1 pp reduction in iron deficiency. "
                      "Derived from cross-country ANC/iron deficiency/LBW associations (H2 analysis)."),
        ],
    },

    "mms": {
        "name":        "Multiple Micronutrient Supplementation",
        "short":       "MMS",
        "emoji":       "💊",
        "description": (
            "Switch ANC4 attenders from iron-folic acid (IFA) to MMS (15 micronutrients). "
            "Reduces LBW and preterm birth through broader micronutrient coverage. "
            "Framed as product switch, not coverage expansion."
        ),
        "outcomes": ["lbw", "preterm"],
        "evidence_tag": "SUMMIT meta-analysis 2022 · Bourassa et al. 2019",
        "tier": "data-informed",

        "data_params": [
            dict(key="anc4_coverage_pct",        label="ANC4+ coverage (%)",
                 source="DHS/MICS"),
            dict(key="low_birthweight_pct",      label="Low birthweight prevalence (%)",
                 source="WHO/UNICEF"),
            dict(key="preterm_birth_rate_pct",   label="Preterm birth rate (%)",
                 source="WHO"),
        ],

        "adj_params": [
            dict(key="mms_switch_pct", label="IFA → MMS switch rate among ANC4 attenders (%)",
                 default=60.0, min=10.0, max=100.0, step=5.0,
                 help="Fraction of women currently receiving IFA who switch to MMS."),
            dict(key="mms_rrr_lbw", label="RRR: LBW (MMS vs. IFA)",
                 default=0.15, min=0.05, max=0.30, step=0.01,
                 help="Relative risk reduction in low birthweight vs. IFA. "
                      "Default 0.15 from SUMMIT 2022 meta-analysis."),
            dict(key="mms_rrr_preterm", label="RRR: Preterm birth (MMS vs. IFA)",
                 default=0.10, min=0.03, max=0.22, step=0.01,
                 help="Relative risk reduction in preterm birth vs. IFA. "
                      "Default 0.10 from SUMMIT 2022."),
        ],
    },

    "iv_iron": {
        "name":        "IV-Iron for Severe Pregnancy Anaemia",
        "short":       "IV-Iron",
        "emoji":       "💉",
        "description": (
            "Intravenous iron treatment for severely anaemic pregnant women at ANC contact. "
            "Addresses haemorrhage-related maternal mortality in high-anaemia settings."
        ),
        "outcomes": ["maternal_deaths"],
        "evidence_tag": "Pavord et al. 2015 · Govindappagari & Burwick 2019",
        "tier": "mostly-adjustable",

        "data_params": [
            dict(key="anaemia_pregnant_women_pct",  label="Anaemia in pregnant women (%)",
                 source="WHO GHO"),
            dict(key="anc4_coverage_pct",           label="ANC4+ coverage (%)",
                 source="DHS/MICS"),
            dict(key="maternal_mortality_per100k",  label="Maternal mortality ratio (/100k births)",
                 source="WHO/World Bank"),
        ],

        "adj_params": [
            dict(key="iv_severe_fraction", label="Fraction with severe anaemia (Hb < 7 g/dL)",
                 default=0.30, min=0.10, max=0.55, step=0.05,
                 help="Fraction of anaemic pregnant women with severe anaemia. "
                      "WHO estimates ~30% of pregnancy anaemia is severe in high-burden settings."),
            dict(key="iv_treatment_coverage",
                 label="IV-Iron treatment coverage among severe cases with ANC contact (%)",
                 default=25.0, min=5.0, max=40.0, step=5.0,
                 help="Coverage capped at 40% given clinical infrastructure constraints. "
                      "Reflects need for skilled administration, cold-chain, monitoring."),
            dict(key="iv_rrr_maternal_death",
                 label="RRR: Maternal death (treated vs. untreated severe anaemia)",
                 default=0.30, min=0.10, max=0.55, step=0.05,
                 help="Estimated relative risk reduction in maternal death for IV-iron treated "
                      "vs. untreated severe pregnancy anaemia. Pavord et al. 2015."),
            dict(key="iv_haemorrhage_fraction",
                 label="Fraction of maternal deaths attributable to haemorrhage/anaemia",
                 default=0.27, min=0.15, max=0.50, step=0.05,
                 help="WHO estimate: ~27% of maternal deaths from obstetric haemorrhage, "
                      "for which severe anaemia is a key risk amplifier."),
        ],
    },

    "maternal_gut": {
        "name":        "Maternal Gut Microbiome Support",
        "short":       "Maternal Gut",
        "emoji":       "🤰",
        "description": (
            "Probiotic/synbiotic supplementation during pregnancy targeting maternal gut "
            "barrier function, systemic inflammation, and nutrient absorption. "
            "May reduce LBW and preterm birth. Evidence is early-stage."
        ),
        "outcomes": ["lbw", "preterm"],
        "evidence_tag": "Wickens et al. 2017 · Odamaki et al. 2020 · emerging RCT evidence",
        "tier": "adjustable",

        "data_params": [
            dict(key="anc4_coverage_pct",       label="ANC4+ coverage (%)",
                 source="DHS/MICS"),
            dict(key="low_birthweight_pct",     label="Low birthweight prevalence (%)",
                 source="WHO/UNICEF"),
            dict(key="preterm_birth_rate_pct",  label="Preterm birth rate (%)",
                 source="WHO"),
        ],

        "adj_params": [
            dict(key="maternal_gut_coverage",
                 label="Coverage among pregnant women (%)",
                 default=25.0, min=5.0, max=60.0, step=5.0,
                 help="Reach among pregnant women via ANC, CHW, or community channel. "
                      "Capped internally at ANC4 rate (most plausible delivery platform)."),
            dict(key="maternal_gut_rrr_lbw",
                 label="RRR: Low birthweight",
                 default=0.08, min=0.02, max=0.20, step=0.01,
                 help="Estimated RRR for LBW. Conservative default (0.08) reflects "
                      "limited RCT evidence; upper range informed by inflammation/nutrient "
                      "absorption mechanisms. ⚠ Treat as directional."),
            dict(key="maternal_gut_rrr_preterm",
                 label="RRR: Preterm birth",
                 default=0.06, min=0.02, max=0.15, step=0.01,
                 help="Estimated RRR for preterm birth from maternal probiotic trials. "
                      "⚠ Evidence base is limited — treat as directional."),
        ],
    },

    "infant_gut": {
        "name":        "Infant Gut Microbiome Support (e.g. B. infantis)",
        "short":       "Infant Gut / B. infantis",
        "emoji":       "👶",
        "description": (
            "Probiotic supplementation (e.g. B. infantis EVC001) targeting environmental "
            "enteropathy (EED) and gut dysbiosis in infancy. Improves HMO utilisation, "
            "reduces gut inflammation, and may meaningfully reduce stunting in high-burden settings."
        ),
        "outcomes": ["stunting"],
        "evidence_tag": "Frese et al. 2017 · Nguyen et al. 2021 · MALED consortium",
        "tier": "adjustable",

        "data_params": [
            dict(key="stunting_pct_who", label="Stunting prevalence <5 (%)",
                 source="UNICEF/JME"),
        ],

        "adj_params": [
            dict(key="infant_gut_coverage",
                 label="Program reach (% of under-2 population)",
                 default=25.0, min=5.0, max=60.0, step=5.0,
                 help="Coverage among infants under 2 years via CHW, facility, or "
                      "breastfeeding support programme. Under-2 is the key window for B. infantis."),
            dict(key="infant_gut_rrr_stunting",
                 label="RRR: Stunting prevalence",
                 default=0.10, min=0.03, max=0.25, step=0.01,
                 help="Estimated fractional reduction in stunting prevalence among treated infants. "
                      "Default 0.10 informed by B. infantis gut-health and EED literature. "
                      "⚠ Stunting-specific RCT data is limited — treat as directional."),
            dict(key="infant_gut_years",
                 label="Program duration (years, for cumulative count)",
                 default=5, min=1, max=10, step=1,
                 help="Cumulative stunting impact scaled by program duration."),
        ],
    },
}


def product_params_defaults(product_key: str) -> dict:
    """Return a dict of {param_key: default_value} for all adjustable params of a product."""
    reg = PRODUCT_REGISTRY.get(product_key, {})
    return {p["key"]: p["default"] for p in reg.get("adj_params", []) if "default" in p}


# ── Impact computations ───────────────────────────────────────────────────────

def compute_product_impact(
    product_key: str,
    country_row: pd.Series,
    population: float,
    params: dict,
) -> dict:
    """
    Compute standalone health impact of one product for a given country.

    Args:
        product_key:  One of "lsff", "mms", "iv_iron", "maternal_gut", "infant_gut"
        country_row:  Row from commons_snapshot.csv (pd.Series)
        population:   Country population (from population.csv)
        params:       Dict of adjustable parameter values (keyed by param key)

    Returns:
        dict with:
          product              — product key
          lbw_averted          — annual LBW cases averted (float)
          stunted_averted      — annual stunted children averted (float; scale by years for cumulative)
          maternal_deaths_averted — annual maternal deaths averted (float)
          annual_births        — estimated annual births (float)
          details              — intermediate calculation steps
    """
    annual_births = estimate_annual_births(country_row, population)
    result: dict = {
        "product":                   product_key,
        "lbw_averted":               0.0,
        "stunted_averted":           0.0,
        "maternal_deaths_averted":   0.0,
        "annual_births":             annual_births,
        "details":                   {},
    }

    def _get(col: str, default: float = 0.0) -> float:
        v = country_row.get(col)
        return float(v) if pd.notna(v) and v != "" else default

    # ── LSFF ─────────────────────────────────────────────────────────────────
    if product_key == "lsff":
        iron_def      = _get("iron_deficiency_pct")
        current_cov   = _get("lsff_coverage_proxy_pct")
        target_cov    = float(params.get("lsff_target_coverage", 75.0))
        efficacy      = float(params.get("lsff_efficacy_iron", 0.36))
        iron_to_lbw   = float(params.get("lsff_iron_to_lbw", 0.10))

        cov_delta_pp     = max(0.0, target_cov - current_cov)
        delta_iron_pp    = iron_def * (cov_delta_pp / 100.0) * efficacy
        delta_lbw_pp     = delta_iron_pp * iron_to_lbw

        # Secondary: iron def → stunting via growth impairment (coefficient 0.15)
        delta_stunting_pp = delta_iron_pp * 0.15

        result["lbw_averted"]      = max(0.0, annual_births * delta_lbw_pp / 100.0)
        result["stunted_averted"]  = max(0.0, annual_births * 5 * delta_stunting_pp / 100.0 / 5)
        result["details"] = dict(
            cov_delta_pp=cov_delta_pp,
            delta_iron_pp=delta_iron_pp,
            delta_lbw_pp=delta_lbw_pp,
            delta_stunting_pp=delta_stunting_pp,
        )

    # ── MMS ──────────────────────────────────────────────────────────────────
    elif product_key == "mms":
        anc4          = _get("anc4_coverage_pct")
        lbw_base      = _get("low_birthweight_pct")
        preterm_base  = _get("preterm_birth_rate_pct")
        switch_pct    = float(params.get("mms_switch_pct", 60.0))
        rrr_lbw       = float(params.get("mms_rrr_lbw", 0.15))
        rrr_preterm   = float(params.get("mms_rrr_preterm", 0.10))

        anc_attenders     = annual_births * (anc4 / 100.0)
        switched          = anc_attenders * (switch_pct / 100.0)
        lbw_averted       = switched * (lbw_base / 100.0) * rrr_lbw
        preterm_averted   = switched * (preterm_base / 100.0) * rrr_preterm

        result["lbw_averted"] = max(0.0, lbw_averted)
        result["details"] = dict(
            anc_attenders=anc_attenders,
            switched=switched,
            lbw_averted=lbw_averted,
            preterm_averted=preterm_averted,
        )

    # ── IV-Iron ───────────────────────────────────────────────────────────────
    elif product_key == "iv_iron":
        anaemia_pw    = _get("anaemia_pregnant_women_pct")
        anc4          = _get("anc4_coverage_pct")
        mmr           = _get("maternal_mortality_per100k")
        sev_frac      = float(params.get("iv_severe_fraction", 0.30))
        tx_cov        = float(params.get("iv_treatment_coverage", 25.0)) / 100.0
        rrr           = float(params.get("iv_rrr_maternal_death", 0.30))
        haem_frac     = float(params.get("iv_haemorrhage_fraction", 0.27))

        severe_pw            = annual_births * (anaemia_pw / 100.0) * sev_frac
        anc_attenders        = annual_births * (anc4 / 100.0)
        treatable            = min(severe_pw, anc_attenders) * tx_cov

        total_mat_deaths     = annual_births * (mmr / 100_000.0)
        addressable_deaths   = total_mat_deaths * haem_frac

        # Fraction of addressable deaths "owned" by severe anaemia population
        sev_share            = (anaemia_pw / 100.0 * sev_frac)
        risk_multiplier      = 3.5  # severe anaemic women ~3.5x elevated mortality risk
        # Expected deaths among severe anaemic women
        if annual_births > 0 and sev_share > 0:
            # weighted correction: sev_share * risk_mult / (sev_share*risk_mult + (1-sev_share))
            sev_pop_weight   = sev_share * risk_multiplier
            baseline_weight  = (1.0 - sev_share)
            denominator      = sev_pop_weight + baseline_weight
            sev_death_share  = sev_pop_weight / denominator if denominator > 0 else 0.5
        else:
            sev_death_share  = 0.0

        deaths_among_severe  = addressable_deaths * sev_death_share
        if severe_pw > 0:
            per_capita_risk  = deaths_among_severe / severe_pw
        else:
            per_capita_risk  = 0.0

        deaths_averted = treatable * per_capita_risk * rrr

        result["maternal_deaths_averted"] = max(0.0, deaths_averted)
        result["details"] = dict(
            severe_pw=severe_pw,
            treatable=treatable,
            total_mat_deaths=total_mat_deaths,
            addressable_deaths=addressable_deaths,
            deaths_among_severe=deaths_among_severe,
            per_capita_risk=per_capita_risk,
            deaths_averted=deaths_averted,
        )

    # ── Maternal gut microbiome support ──────────────────────────────────────
    elif product_key == "maternal_gut":
        anc4        = _get("anc4_coverage_pct")
        lbw_base    = _get("low_birthweight_pct")
        preterm_base = _get("preterm_birth_rate_pct")
        coverage    = float(params.get("maternal_gut_coverage", 25.0)) / 100.0
        rrr_lbw     = float(params.get("maternal_gut_rrr_lbw", 0.08))
        rrr_preterm = float(params.get("maternal_gut_rrr_preterm", 0.06))

        # Cap effective coverage at ANC4 rate — most plausible delivery platform
        effective_cov   = min(coverage, anc4 / 100.0)
        pregnant_reached = annual_births * effective_cov
        lbw_averted     = pregnant_reached * (lbw_base / 100.0) * rrr_lbw
        preterm_averted = pregnant_reached * (preterm_base / 100.0) * rrr_preterm

        result["lbw_averted"] = max(0.0, lbw_averted)
        result["details"] = dict(
            effective_coverage=effective_cov,
            pregnant_reached=pregnant_reached,
            lbw_averted=lbw_averted,
            preterm_averted=preterm_averted,
        )

    # ── Infant gut microbiome support (e.g. B. infantis) ─────────────────────
    elif product_key == "infant_gut":
        stunting_base = _get("stunting_pct_who")
        coverage      = float(params.get("infant_gut_coverage", 25.0)) / 100.0
        rrr_stunting  = float(params.get("infant_gut_rrr_stunting", 0.10))

        # Target under-2 population — key window for B. infantis colonisation
        u2_pop                 = annual_births * 2.0
        infants_reached        = u2_pop * coverage
        stunted_in_reached     = infants_reached * (stunting_base / 100.0)
        stunted_averted_annual = stunted_in_reached * rrr_stunting

        result["stunted_averted"] = max(0.0, stunted_averted_annual)
        result["details"] = dict(
            u2_pop=u2_pop,
            infants_reached=infants_reached,
            stunted_in_reached=stunted_in_reached,
            stunted_averted_annual=stunted_averted_annual,
        )

    return result


def compute_combined_impact(
    active_products: list[str],
    country_row: pd.Series,
    population: float,
    params_dict: dict[str, dict],
) -> dict:
    """
    Combine impact across active products using sequential multiplicative logic.

    For shared outcomes (LBW: LSFF + MMS; stunting: LSFF + Gut), this prevents
    double-counting by applying each product's RRR to the *remaining* risk pool
    after upstream products have been applied.

    Args:
        active_products: Ordered list of product keys to activate (upstream first).
        country_row:     Snapshot row for the selected country.
        population:      Country population.
        params_dict:     {product_key: {param_key: value}} for all active products.

    Returns:
        dict with:
          annual_births               float
          lbw_total                   float (annual)
          lbw_by_product              {product_key: float}
          stunting_total_5yr          float (cumulative, using infant_gut_years)
          stunting_by_product         {product_key: float}
          maternal_deaths_total       float (annual)
          maternal_deaths_by_product  {product_key: float}
          individual                  {product_key: impact_dict}
    """

    def _get(col: str, default: float = 0.0) -> float:
        v = country_row.get(col)
        return float(v) if pd.notna(v) and v != "" else default

    annual_births   = estimate_annual_births(country_row, population)
    lbw_base_pct    = _get("low_birthweight_pct")
    stunting_base_pct = _get("stunting_pct_who")
    mmr             = _get("maternal_mortality_per100k")

    # Baseline case counts
    lbw_baseline_cases    = annual_births * (lbw_base_pct / 100.0)
    u5_pop                = annual_births * 5.0
    stunting_baseline_cases = u5_pop * (stunting_base_pct / 100.0)

    # Compute individual product impacts
    individual: dict[str, dict] = {}
    for pk in active_products:
        params = params_dict.get(pk, {})
        individual[pk] = compute_product_impact(pk, country_row, population, params)

    # ── LBW: LSFF → MMS → maternal_gut (upstream → downstream) ─────────────
    remaining_lbw = 1.0
    lbw_by_product: dict[str, float] = {}
    for pk in ["lsff", "mms", "maternal_gut"]:
        if pk not in active_products:
            continue
        if lbw_baseline_cases > 0:
            rrr = min(individual[pk]["lbw_averted"] / lbw_baseline_cases, 0.70)
        else:
            rrr = 0.0
        prev = remaining_lbw
        remaining_lbw *= (1.0 - rrr)
        lbw_by_product[pk] = lbw_baseline_cases * (prev - remaining_lbw)

    lbw_total = sum(lbw_by_product.values())

    # ── Stunting: LSFF → infant_gut (upstream → downstream) ─────────────────
    infant_gut_years = int(params_dict.get("infant_gut", {}).get("infant_gut_years", 5))
    remaining_stunting = 1.0
    stunting_by_product: dict[str, float] = {}
    for pk in ["lsff", "infant_gut"]:
        if pk not in active_products:
            continue
        years = infant_gut_years if pk == "infant_gut" else infant_gut_years
        annual_averted = individual[pk]["stunted_averted"]
        cumulative = annual_averted * years
        if stunting_baseline_cases * years > 0:
            rrr = min(cumulative / (stunting_baseline_cases * years), 0.70)
        else:
            rrr = 0.0
        prev = remaining_stunting
        remaining_stunting *= (1.0 - rrr)
        stunting_by_product[pk] = stunting_baseline_cases * years * (prev - remaining_stunting)

    stunting_total_5yr = sum(stunting_by_product.values())

    # ── Maternal deaths: IV-Iron only (primary driver) ───────────────────────
    maternal_by_product: dict[str, float] = {}
    if "iv_iron" in active_products:
        maternal_by_product["iv_iron"] = individual["iv_iron"]["maternal_deaths_averted"]

    maternal_total = sum(maternal_by_product.values())

    # ── Baseline case counts (for % averted calculation in charts) ────────────
    stunting_baseline_5yr = stunting_baseline_cases * infant_gut_years
    maternal_deaths_baseline = annual_births * (mmr / 100_000.0)

    return dict(
        annual_births=annual_births,
        # Totals
        lbw_total=lbw_total,
        stunting_total_5yr=stunting_total_5yr,
        maternal_deaths_total=maternal_total,
        # Per-product breakdowns
        lbw_by_product=lbw_by_product,
        stunting_by_product=stunting_by_product,
        maternal_deaths_by_product=maternal_by_product,
        # Baselines (for % averted)
        lbw_baseline_cases=lbw_baseline_cases,
        stunting_baseline_5yr=stunting_baseline_5yr,
        maternal_deaths_baseline=maternal_deaths_baseline,
        # Other
        program_years=infant_gut_years,
        individual=individual,
    )


# ── DALY & cost defaults ──────────────────────────────────────────────────────

# DALYs per averted case (GBD-informed; all adjustable in UI)
#   LBW:           neonatal mortality risk + lifelong cognitive/metabolic disability
#                  GBD 2019 child growth failure burden → ~8 DALYs per LBW case
#   Stunting:      lifetime cognitive impairment + increased mortality risk
#                  GBD 2019 stunting disability weight + YLL → ~2.8 DALYs per case
#   Maternal death: YLL only, ~30 years remaining life expectancy at death
DALY_DEFAULTS = {
    "daly_per_lbw":            8.0,
    "daly_per_stunted_child":  2.8,
    "daly_per_maternal_death": 30.0,
}

# Incremental cost per unit delivered (USD; adjustable in UI)
#   LSFF:         $/person/year covered by fortification (food systems delivery)
#   MMS:          marginal $/woman switched from IFA (drug cost differential)
#   IV-Iron:      $/treatment course (drug + administration)
#   Maternal Gut: $/pregnancy course (probiotic sachets)
#   Infant Gut:   $/infant course (B. infantis product)
COST_DEFAULTS = {
    "cost_lsff_per_person_yr":     0.15,
    "cost_mms_per_woman":          3.00,
    "cost_iv_iron_per_treatment":  50.00,
    "cost_maternal_gut_per_course": 2.50,
    "cost_infant_gut_per_course":   3.00,
}


def compute_daly_cost(
    combined: dict,
    daly_weights: dict,
    cost_params: dict,
    population: float,
    country_row: pd.Series,
) -> dict:
    """
    Back-of-envelope DALY and program cost estimation.

    Args:
        combined:     Output of compute_combined_impact()
        daly_weights: {"daly_per_lbw", "daly_per_stunted_child", "daly_per_maternal_death"}
        cost_params:  {"cost_lsff_per_person_yr", "cost_mms_per_woman", ...}
        population:   Country population
        country_row:  Snapshot row (for coverage data)

    Returns dict with:
        dalys_total          — total annual DALYs averted
        dalys_by_outcome     — {outcome: dalys}
        cost_total_usd       — total annual program cost (USD)
        cost_by_product      — {product_key: annual_cost_usd}
        cost_per_daly        — USD per DALY averted (None if dalys_total == 0)
        inputs               — key intermediate values for display
    """
    program_years = combined.get("program_years", 5)
    individual    = combined.get("individual", {})

    def _dw(key: str) -> float:
        return float(daly_weights.get(key, DALY_DEFAULTS.get(key, 0)))

    def _cp(key: str) -> float:
        return float(cost_params.get(key, COST_DEFAULTS.get(key, 0)))

    # ── DALYs ─────────────────────────────────────────────────────────────────
    # Annual LBW DALYs
    lbw_dalys = combined.get("lbw_total", 0) * _dw("daly_per_lbw")

    # Annual stunting DALYs (convert cumulative back to annual rate × lifetime DALY weight)
    stunting_annual = (
        combined.get("stunting_total_5yr", 0) / max(program_years, 1)
    )
    stunting_dalys = stunting_annual * _dw("daly_per_stunted_child")

    # Annual maternal death DALYs
    maternal_dalys = combined.get("maternal_deaths_total", 0) * _dw("daly_per_maternal_death")

    dalys_by_outcome = {
        "LBW":             lbw_dalys,
        "Stunting":        stunting_dalys,
        "Maternal deaths": maternal_dalys,
    }
    dalys_total = sum(dalys_by_outcome.values())

    # ── Costs ─────────────────────────────────────────────────────────────────
    def _get(col: str, default: float = 0.0) -> float:
        v = country_row.get(col)
        return float(v) if pd.notna(v) and v != "" else default

    cost_by_product: dict[str, float] = {}

    # LSFF: cost per person covered by the incremental coverage increase
    if "lsff" in individual:
        details = individual["lsff"].get("details", {})
        cov_delta = details.get("cov_delta_pp", 0)
        people_newly_covered = population * cov_delta / 100.0
        cost_by_product["lsff"] = people_newly_covered * _cp("cost_lsff_per_person_yr")

    # MMS: marginal cost per woman switched
    if "mms" in individual:
        switched = individual["mms"].get("details", {}).get("switched", 0)
        cost_by_product["mms"] = switched * _cp("cost_mms_per_woman")

    # IV-Iron: cost per treatment course
    if "iv_iron" in individual:
        treatable = individual["iv_iron"].get("details", {}).get("treatable", 0)
        cost_by_product["iv_iron"] = treatable * _cp("cost_iv_iron_per_treatment")

    # Maternal Gut: cost per pregnancy course
    if "maternal_gut" in individual:
        reached = individual["maternal_gut"].get("details", {}).get("pregnant_reached", 0)
        cost_by_product["maternal_gut"] = reached * _cp("cost_maternal_gut_per_course")

    # Infant Gut: cost per infant course
    if "infant_gut" in individual:
        reached = individual["infant_gut"].get("details", {}).get("infants_reached", 0)
        cost_by_product["infant_gut"] = reached * _cp("cost_infant_gut_per_course")

    cost_total = sum(cost_by_product.values())
    cost_per_daly = cost_total / dalys_total if dalys_total > 0 else None

    return dict(
        dalys_total=dalys_total,
        dalys_by_outcome=dalys_by_outcome,
        cost_total_usd=cost_total,
        cost_by_product=cost_by_product,
        cost_per_daly=cost_per_daly,
    )


# ── Charts ────────────────────────────────────────────────────────────────────

def impact_bars_chart(
    combined: dict,
    country_name: str,
    height: int = 480,
) -> go.Figure:
    """
    Three-row horizontal bar chart — one row per headline outcome.

    Each row shows one horizontal bar per active product (independent colors)
    plus a "Combined" bar using the sequential multiplicative total.
    Each row has its own independent x-axis so different-magnitude outcomes
    are all legible.

    Rows (top → bottom):
      1. LBW averted (annual)
      2. Stunted children averted (N-year)
      3. Maternal deaths averted (annual)
    """
    gut_years = combined.get("program_years", 5)
    lbw_bp    = combined.get("lbw_by_product", {})
    stunt_bp  = combined.get("stunting_by_product", {})
    mat_bp    = combined.get("maternal_deaths_by_product", {})

    # Baselines for % averted
    lbw_baseline      = combined.get("lbw_baseline_cases", 0)
    stunting_baseline  = combined.get("stunting_baseline_5yr", 0)
    mat_baseline       = combined.get("maternal_deaths_baseline", 0)

    all_products = sorted(
        set(list(lbw_bp) + list(stunt_bp) + list(mat_bp)),
        key=lambda p: list(PRODUCT_REGISTRY).index(p) if p in PRODUCT_REGISTRY else 99,
    )

    outcome_specs = [
        (lbw_bp,   combined.get("lbw_total", 0),   lbw_baseline,
         "LBW cases averted (annual)"),
        (stunt_bp, combined.get("stunting_total_5yr", 0), stunting_baseline,
         f"Stunted children averted ({gut_years}-year)"),
        (mat_bp,   combined.get("maternal_deaths_total", 0), mat_baseline,
         "Maternal deaths averted (annual)"),
    ]

    def _pct_str(val: float, baseline: float) -> str:
        """Return '(X.X%)' string, or '' if baseline is zero."""
        if baseline > 0:
            return f"({val / baseline * 100:.1f}%)"
        return ""

    # Only include rows that have at least one nonzero value
    active_rows = [
        (bp, tot, baseline, title) for bp, tot, baseline, title in outcome_specs
        if tot > 0 or any(bp.get(pk, 0) > 0 for pk in all_products)
    ]
    if not active_rows:
        fig = go.Figure()
        fig.add_annotation(
            text="Enable products on the left to see impact projections.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=13, color="#888"), showarrow=False,
        )
        fig.update_layout(height=height, paper_bgcolor="white",
                          margin=dict(l=20, r=20, t=40, b=20))
        return fig

    n_rows = len(active_rows)
    row_heights = [1.0 / n_rows] * n_rows

    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=row_heights,
        vertical_spacing=0.20,
    )

    for row_idx, (bp, total, baseline, row_title) in enumerate(active_rows, start=1):
        # Collect y-labels, x-values, colors, and hover text for all bars
        y_labels    = []
        x_vals      = []
        colors      = []
        hover_texts = []

        for pk in all_products:
            val = bp.get(pk, 0.0)
            if val <= 0:
                continue
            reg    = PRODUCT_REGISTRY.get(pk, {})
            plabel = f"{reg.get('emoji', '')} {reg.get('short', pk)}"
            pct    = _pct_str(val, baseline)
            y_labels.append(plabel)
            x_vals.append(val)
            colors.append(PRODUCT_COLORS.get(pk, "#888888"))
            hover_texts.append(
                f"<b>{plabel}</b><br>"
                f"Averted: <b>{val:,.0f}</b>"
                + (f" {pct}" if pct else "")
                + (f"<br>of {baseline:,.0f} baseline cases" if baseline > 0 else "")
            )

        # Combined total bar (only when multiple products contribute)
        if total > 0 and len(y_labels) >= 1:
            pct = _pct_str(total, baseline)
            y_labels.append("⬛ Combined")
            x_vals.append(total)
            colors.append(FOUNDATION_BLUE)
            hover_texts.append(
                f"<b>Combined (sequential)</b><br>"
                f"Averted: <b>{total:,.0f}</b>"
                + (f" {pct}" if pct else "")
                + (f"<br>of {baseline:,.0f} baseline cases" if baseline > 0 else "")
            )

        if not x_vals:
            fig.add_annotation(
                text="No data",
                x=0.5, xref=f"x{row_idx} domain" if row_idx > 1 else "x domain",
                y=0.5, yref=f"y{row_idx} domain" if row_idx > 1 else "y domain",
                font=dict(size=11, color="#999"), showarrow=False,
            )
            continue

        # Inside bar text: "N,NNN (X.X%)" when bar is wide enough, else just count
        max_x = max(x_vals)
        texts = []
        for v in x_vals:
            if v / max_x > 0.25:
                pct = _pct_str(v, baseline)
                texts.append(f"{v:,.0f}  {pct}" if pct else f"{v:,.0f}")
            elif v / max_x > 0.10:
                texts.append(f"{v:,.0f}")
            else:
                texts.append("")

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_labels,
                orientation="h",
                marker_color=colors,
                marker_line_width=0,
                opacity=0.88,
                text=texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=10, color="white"),
                customdata=hover_texts,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=row_idx, col=1,
        )

        # Row title + baseline annotation above each subplot
        yref_domain = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        baseline_note = f" — baseline: {baseline:,.0f}" if baseline > 0 else ""
        fig.add_annotation(
            text=f"<b>{row_title}</b><span style='color:#888;font-size:10px'>{baseline_note}</span>",
            x=0, xref="paper",
            y=1.0, yref=yref_domain,
            xanchor="left", yanchor="bottom",
            font=dict(size=12, color=FOUNDATION_BLUE),
            showarrow=False,
        )

        fig.update_xaxes(
            showgrid=True, gridcolor="#EEEEEE",
            tickformat=",d", zeroline=False,
            tickfont=dict(size=10),
            row=row_idx, col=1,
        )
        fig.update_yaxes(
            showgrid=False,
            tickfont=dict(size=11),
            autorange="reversed",
            row=row_idx, col=1,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{country_name}</b> — portfolio product impact",
            font=dict(size=14, color=FOUNDATION_BLUE),
            x=0.0, xanchor="left",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=FONT,
        height=height,
        showlegend=False,
        margin=dict(l=160, r=60, t=55, b=40),
    )
    return fig


def waterfall_chart(
    combined: dict,
    country_name: str,
    outcome: str = "lbw",
    height: int = 300,
) -> go.Figure:
    """
    Contribution bar chart showing each product's share of impact for one outcome,
    with a "Total (combined)" bar using sequential multiplicative logic.

    outcome: "lbw" | "stunting" | "maternal_deaths"
    """
    gut_years = combined.get("program_years", 5)
    if outcome == "lbw":
        by_product = combined.get("lbw_by_product", {})
        total      = combined.get("lbw_total", 0)
        baseline   = combined.get("lbw_baseline_cases", 0)
        title_str  = "LBW cases averted (annual)"
        note       = "(sequential multiplicative — not additive)"
    elif outcome == "stunting":
        by_product = combined.get("stunting_by_product", {})
        total      = combined.get("stunting_total_5yr", 0)
        baseline   = combined.get("stunting_baseline_5yr", 0)
        title_str  = f"Stunted children averted ({gut_years}-year)"
        note       = "(sequential multiplicative — not additive)"
    else:  # maternal_deaths
        by_product = combined.get("maternal_deaths_by_product", {})
        total      = combined.get("maternal_deaths_total", 0)
        baseline   = combined.get("maternal_deaths_baseline", 0)
        title_str  = "Maternal deaths averted (annual)"
        note       = ""

    def _fmt(val: float) -> str:
        """Format value + % of baseline."""
        if baseline > 0:
            return f"{val:,.0f}  ({val / baseline * 100:.1f}%)"
        return f"{val:,.0f}"

    def _hover(label: str, val: float) -> str:
        pct = f" ({val / baseline * 100:.1f}% of {baseline:,.0f} baseline)" if baseline > 0 else ""
        return f"<b>{label}</b><br>Averted: <b>{val:,.0f}</b>{pct}"

    # Products with nonzero contribution
    products = [(pk, v) for pk, v in by_product.items() if v > 0]
    if not products:
        fig = go.Figure()
        fig.add_annotation(
            text="No active products contribute to this outcome",
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=12, color="#888"), showarrow=False,
        )
        fig.update_layout(
            height=height, paper_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    # Build bars: one per product + combined total (only if >1 product)
    labels      = []
    vals        = []
    colors      = []
    texts       = []
    hover_data  = []

    for pk, val in products:
        reg   = PRODUCT_REGISTRY.get(pk, {})
        label = f"{reg.get('emoji', '')} {reg.get('short', pk)}"
        labels.append(label)
        vals.append(val)
        colors.append(PRODUCT_COLORS.get(pk, "#888888"))
        texts.append(_fmt(val))
        hover_data.append(_hover(label, val))

    # Add "Combined total" only when multiple products contribute
    if len(products) > 1:
        labels.append("⬛ Combined total")
        vals.append(total)
        colors.append(FOUNDATION_BLUE)
        texts.append(_fmt(total))
        hover_data.append(_hover("Combined (sequential)", total))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=texts,
        textposition="outside",
        textfont=dict(size=11, color="#333"),
        customdata=hover_data,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

    # Dashed line at additive sum to show combination efficiency
    if len(products) > 1:
        additive_sum = sum(v for _, v in products)
        if additive_sum > total * 1.02:   # only show if meaningful difference
            fig.add_vline(
                x=additive_sum,
                line=dict(color="#BBBBBB", dash="dot", width=1.5),
                annotation=dict(
                    text=f"Additive: {additive_sum:,.0f}",
                    font=dict(size=9, color="#999"),
                    xanchor="right", yanchor="bottom",
                ),
            )

    # % averted subtitle under title
    pct_note = f" — {total / baseline * 100:.1f}% of baseline" if baseline > 0 else ""

    fig.update_layout(
        title=dict(
            text=(f"{title_str}{pct_note}"
                  f"  <sup style='font-size:10px;color:#888'>{note}</sup>"),
            font=dict(size=12, color=FOUNDATION_BLUE), x=0.02,
        ),
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE", title=""),
        yaxis=dict(showgrid=False, autorange="reversed"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=FONT,
        height=height,
        margin=dict(l=140, r=80, t=50, b=30),
        showlegend=False,
    )
    return fig


def country_context_card(
    country_row: pd.Series,
    combined: dict,
) -> dict:
    """
    Return a dict of context values for the country mini-profile panel.
    Used to drive st.metric cards in the dashboard — not a chart.
    """
    def _get(col: str) -> str:
        v = country_row.get(col)
        return f"{float(v):.1f}" if pd.notna(v) and v != "" else "—"

    return {
        "stunting":        _get("stunting_pct_who"),
        "lbw":             _get("low_birthweight_pct"),
        "anaemia_pw":      _get("anaemia_pregnant_women_pct"),
        "anc4":            _get("anc4_coverage_pct"),
        "iron_def":        _get("iron_deficiency_pct"),
        "mmr":             _get("maternal_mortality_per100k"),
        "lsff_cov":        _get("lsff_coverage_proxy_pct"),
        "annual_births":   combined.get("annual_births", 0),
    }
