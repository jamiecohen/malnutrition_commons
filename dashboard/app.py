"""
Malnutrition Data Commons — Interactive Dashboard
June 2026 Learning Session Preview

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.viz.figures import (
    INDICATOR_CONFIG,
    choropleth_map,
    cooccurrence_scatter,
    burden_bar,
    trend_lines,
    FOUNDATION_BLUE,
)
from src.viz.triple_burden import (
    composite_burden_map,
    cooccurrence_scatter as tb_scatter,
    burden_profile_bars,
    lsff_coverage_map,
    lsff_gap_scatter,
)
from src.viz.insights import (
    _norm_composite,
    h1_vaccination_measles,
    h2_anc_birth_outcomes,
    h3_malaria_anaemia,
    h4_hiv_tb,
    h5_system_vs_burden,
    h6_lsff_gap,
    h7_vitamin_a_measles,
    h8_maternal_anaemia_mortality,
    h9_undernutrition_child_mortality,
    h10_nutrition_human_capital,
    h11_food_insecurity_pathway,
    burden_heatmap,
)
from src.viz.scenarios import (
    INTERVENTION_CHAINS,
    fit_model,
    project_outcome,
    project_two_step,
    population_impact,
    scenario_scatter,
    two_step_scatter,
)
from src.viz.product_impact import (
    PRODUCT_REGISTRY,
    PRODUCT_COLORS,
    DALY_DEFAULTS,
    COST_DEFAULTS,
    compute_product_impact,
    compute_combined_impact,
    compute_daly_cost,
    product_params_defaults,
    impact_bars_chart,
    country_context_card,
)
from src.viz.subnational import (
    load_nigeria_data,
    nigeria_choropleth,
    nigeria_multi_map,
    nigeria_zone_bars,
    nigeria_scatter,
    INDICATOR_CONFIG as NGA_INDICATOR_CONFIG,
)

# ── Optional MUMTA import ───────────────────────────────────────────────────
try:
    from src.viz.mumta import (
        cohort_overview_metrics,
        birth_outcomes_by_arm,
        birth_outcomes_by_risk_factor,
        birth_weight_distribution,
        adverse_outcome_summary,
        maternal_anemia_trajectory,
        infant_growth_curves,
        binfantis_colonization,
        model_vs_cohort_comparison,
        pathogen_detection_heatmap,
        pathogen_burden_trajectory,
        gut_inflammation_vs_growth,
        top_pathogens_by_timepoint,
        binfantis_colonization_corrected,
        binfantis_by_arm,
        binfantis_vs_pathogens,
        binfantis_vs_inflammation,
        binfantis_vs_growth,
        ARM_LABELS as MUMTA_ARM_LABELS,
    )
    _MUMTA_VIZ_AVAILABLE = True
except Exception as _mumta_err:
    _MUMTA_VIZ_AVAILABLE = False
    _MUMTA_IMPORT_ERROR = str(_mumta_err)

# ── Optional AMANHI import ──────────────────────────────────────────────────
try:
    from src.viz.amanhi import (
        binfantis_by_site as amanhi_binfantis_by_site,
        binfantis_vs_outcomes as amanhi_binfantis_vs_outcomes,
        binfantis_vs_growth as amanhi_binfantis_vs_growth,
        binfantis_cross_cohort as amanhi_binfantis_cross_cohort,
        crp_distribution as amanhi_crp_distribution,
        ferritin_distribution as amanhi_ferritin_distribution,
        crp_vs_birth_outcomes as amanhi_crp_vs_birth_outcomes,
        ferritin_vs_birth_outcomes as amanhi_ferritin_vs_birth_outcomes,
        crp_ferritin_scatter as amanhi_crp_ferritin_scatter,
        tac_top_pathogens as amanhi_tac_top_pathogens,
        tac_pathogen_burden_vs_outcomes as amanhi_tac_burden_vs_outcomes,
        tac_cross_cohort_comparison as amanhi_tac_cross_cohort,
        # Tier 2
        growth_trajectory_by_binfantis as amanhi_growth_traj_binfantis,
        growth_trajectory_by_blongum as amanhi_growth_traj_blongum,
        growth_by_colonization_group as amanhi_growth_by_colon_group,
        growth_faltering_cross_country as amanhi_growth_faltering_xc,
        binfantis_dose_response as amanhi_binfantis_dose_response,
        growth_trajectory_waz as amanhi_growth_traj_waz,
    )
    _AMANHI_VIZ_AVAILABLE = True
except Exception as _amanhi_err:
    _AMANHI_VIZ_AVAILABLE = False
    _AMANHI_IMPORT_ERROR = str(_amanhi_err)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malnutrition Data Commons",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Page & layout ── */
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* ── Headings ── */
    h1 { color: #003366 !important; font-family: Arial, sans-serif; font-size: 2rem; }
    h2, h3, h4 { color: #003366 !important; font-family: Arial, sans-serif; }

    /* ── Body text: force dark on light background ── */
    p, li, span, div { color: #1A1A2E; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DDE3EA;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label { color: #1A1A2E !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #003366 !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #333333 !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #DDE3EA;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: #555E6E !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #003366 !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* ── Tab labels ── */
    [data-testid="stTabs"] button {
        color: #444D5C !important;
        font-weight: 600;
        font-size: 0.88rem;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #003366 !important;
        border-bottom-color: #003366 !important;
    }

    /* ── Selectbox / slider labels ── */
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stRadio"] label { color: #1A1A2E !important; font-weight: 600; font-size: 0.85rem; }

    /* ── Radio options ── */
    [data-testid="stRadio"] div[role="radiogroup"] label { color: #333333 !important; font-weight: 400; }

    /* ── Expander ── */
    [data-testid="stExpander"] summary { color: #003366 !important; font-weight: 600; }

    /* ── Info / warning banners ── */
    [data-testid="stAlert"] { color: #1A1A2E !important; }

    /* ── Context box (custom HTML) ── */
    .context-box {
        background: #EBF4FF;
        border-left: 4px solid #003366;
        padding: 12px 18px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #1A2A3A !important;
        margin-bottom: 1rem;
    }
    .context-box b { color: #003366; }

    /* ── Sprint badge (sidebar) ── */
    .sprint-badge {
        background: #003366;
        color: #FFFFFF !important;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .sprint-meta { font-size: 0.78rem; color: #555E6E !important; line-height: 1.5; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border: 1px solid #DDE3EA; border-radius: 6px; }

    /* ── Horizontal rule ── */
    hr { border-color: #DDE3EA; }

    /* ── Source badge ── */
    .source-badge {
        display: inline-block;
        background: #F0F4F8;
        border: 1px solid #C8D6E5;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.72rem;
        color: #555E6E;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
    }
    .source-badge.ground-truth {
        background: #E8F5E9;
        border-color: #81C784;
        color: #2E7D32;
    }
    .source-badge.modelled {
        background: #FFF3E0;
        border-color: #FFB74D;
        color: #E65100;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    snap = pd.read_csv(ROOT / "data" / "processed" / "commons_snapshot.csv")
    panel = pd.read_csv(ROOT / "data" / "processed" / "commons_panel.csv")
    pop = pd.read_csv(ROOT / "data" / "processed" / "population.csv")[["iso3", "population"]]
    return snap, panel, pop


@st.cache_data
def prepare_insights_df(snap_hash):
    """Enrich snapshot with derived columns needed by insights functions."""
    snap, _, pop = load_data()
    df = snap.merge(pop, on="iso3", how="left")
    df["measles_per100k"] = df["measles_reported_cases"] / df["population"] * 100_000
    df["mcv_dropout_pct"] = df["mcv1_coverage_pct"] - df["mcv2_coverage_pct"]
    cov_cols = ["anc4_coverage_pct", "mcv1_coverage_pct", "dtp3_coverage_pct",
                "pcv3_coverage_pct", "rotac_coverage_pct"]
    df["health_system_score"] = _norm_composite(df, cov_cols, higher_is_better=True)
    burden_cols = ["anaemia_children_pct", "stunting_pct_who",
                   "iron_deficiency_pct", "low_birthweight_pct"]
    df["nutrition_burden_score"] = _norm_composite(df, burden_cols, higher_is_better=False)
    return df


snap, panel, _pop = load_data()
# Use a stable hash (row count) so cache_data works with the dataframe
_insights_df = prepare_insights_df(len(snap))


@st.cache_data
def load_subnational():
    """Load Nigeria state data and GeoJSON boundary (cached)."""
    try:
        return load_nigeria_data()
    except Exception:
        return None, None


_nga_df, _nga_geojson = load_subnational()

# Indicator display options
INDICATOR_OPTIONS = {v[0]: k for k, v in INDICATOR_CONFIG.items()}
INDICATOR_LABELS  = {k: v[0] for k, v in INDICATOR_CONFIG.items()}

# WHO regions for filtering
WHO_REGIONS = sorted(snap["who_region"].dropna().unique().tolist())
ALL_REGIONS_LABEL = "All regions"

# Priority countries for Learning Session demo
PRIORITY_COUNTRIES = {
    "South Asia":         ["PAK", "BGD", "IND", "NPL", "LKA"],
    "Sub-Saharan Africa": ["NGA", "ETH", "COD", "MOZ", "TZA", "GHA", "KEN", "UGA"],
    "East Africa":        ["ETH", "KEN", "TZA", "UGA", "RWA"],
}

# The six Foundation priority countries used across all analyses
PRIORITY_ISO3 = ["IND", "PAK", "BGD", "NGA", "ETH", "COD"]

# The three deep-dive priority countries for Tab 2
DEEP_DIVE_COUNTRIES = {"Pakistan": "PAK", "India": "IND", "Nigeria": "NGA"}


# ── Country profile helpers ───────────────────────────────────────────────────

# Indicators grouped by domain for the country profile radar and breakdown bars
PROFILE_DOMAINS = {
    "Child Nutrition":    ["stunting_pct_who", "wasting_pct", "underweight_pct", "anaemia_children_pct"],
    "Micronutrients":     ["iron_deficiency_pct", "vitamin_a_deficiency_pct", "zinc_deficiency_pct", "iodine_deficiency_pct"],
    "Birth Outcomes":     ["low_birthweight_pct", "preterm_birth_rate_pct", "anaemia_pregnant_women_pct"],
    "Healthcare":         ["anc4_coverage_pct", "mcv1_coverage_pct", "dtp3_coverage_pct", "pcv3_coverage_pct", "rotac_coverage_pct"],
    "Infectious Disease": ["tb_incidence_per100k", "hiv_prevalence_pct", "malaria_incidence_per1000"],
    "Child Survival":     ["u5_mortality_per1000", "neonatal_mortality_per1000", "maternal_mortality_per100k"],
    "Human Capital":      ["hci_score", "gdp_per_capita_ppp", "severe_food_insecurity_pct"],
}

# True = higher raw value = worse outcome (used to orient domain scores so 100 = worst globally)
INDICATOR_HIGHER_IS_BAD = {
    "stunting_pct_who": True,  "wasting_pct": True,          "underweight_pct": True,
    "anaemia_children_pct": True, "anaemia_pregnant_women_pct": True,
    "iron_deficiency_pct": True,  "vitamin_a_deficiency_pct": True,
    "zinc_deficiency_pct": True,  "iodine_deficiency_pct": True,
    "low_birthweight_pct": True,  "preterm_birth_rate_pct": True,
    "anc4_coverage_pct": False,   "mcv1_coverage_pct": False,   "mcv2_coverage_pct": False,
    "dtp3_coverage_pct": False,   "pcv3_coverage_pct": False,   "rotac_coverage_pct": False,
    "tb_incidence_per100k": True, "hiv_prevalence_pct": True,   "malaria_incidence_per1000": True,
    "u5_mortality_per1000": True, "neonatal_mortality_per1000": True, "maternal_mortality_per100k": True,
    "hci_score": False,           "gdp_per_capita_ppp": False,  "severe_food_insecurity_pct": True,
    "measles_reported_cases": True, "ors_coverage_pct": False,  "lsff_coverage_proxy_pct": False,
}

# Six key metrics shown in the header card row
PROFILE_KEY_METRICS = [
    ("stunting_pct_who",        "Stunting <5",       "%",    False),
    ("anaemia_children_pct",    "Anaemia <5",        "%",    False),
    ("u5_mortality_per1000",    "U5 Mortality",      "/1k",  False),
    ("anc4_coverage_pct",       "ANC4 Coverage",     "%",    True),
    ("hci_score",               "HCI Score",         "",     True),
    ("severe_food_insecurity_pct", "Food Insecurity", "%",   False),
]


@st.cache_data
def compute_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    """For every country, compute a 0-100 burden score per domain (100 = worst globally).

    Each indicator is converted to a percentile rank (0-100).
    For 'higher is bad' indicators the rank is used directly.
    For 'higher is good' indicators (coverage, HCI) the rank is inverted.
    Domain score = mean of available indicator ranks within the domain.
    """
    domain_series = {}
    for domain, cols in PROFILE_DOMAINS.items():
        parts = []
        for col in cols:
            if col not in df.columns:
                continue
            ranks = df[col].rank(pct=True, na_option="keep") * 100
            if INDICATOR_HIGHER_IS_BAD.get(col, True):
                parts.append(ranks)
            else:
                parts.append(100 - ranks)
        if parts:
            domain_series[domain] = pd.concat(parts, axis=1).mean(axis=1)
    return pd.DataFrame(domain_series, index=df.index)


def _build_radar_fig(iso3: str, domain_scores: pd.DataFrame, df: pd.DataFrame) -> go.Figure:
    """Polar radar: country domain scores vs. global and regional medians."""
    idx = df.index[df["iso3"] == iso3]
    if len(idx) == 0:
        return None
    row = df.loc[idx[0]]
    region = row.get("who_region", "")
    country_name = row.get("country_name", iso3)
    region_mask = df["who_region"] == region

    domains = list(PROFILE_DOMAINS.keys())
    c_scores, g_meds, r_meds = [], [], []
    for d in domains:
        if d not in domain_scores.columns:
            c_scores.append(50); g_meds.append(50); r_meds.append(50)
            continue
        col = domain_scores[d]
        c_val = col.loc[idx[0]] if idx[0] in col.index else np.nan
        c_scores.append(c_val if pd.notna(c_val) else col.median())
        g_meds.append(col.median())
        r_meds.append(col[region_mask].median() if region_mask.any() else col.median())

    # Close the polygons
    d_closed  = domains + [domains[0]]
    c_closed  = c_scores + [c_scores[0]]
    g_closed  = g_meds   + [g_meds[0]]
    r_closed  = r_meds   + [r_meds[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=g_closed, theta=d_closed, fill="toself",
        fillcolor="rgba(200,200,200,0.25)",
        line=dict(color="#AAAAAA", width=1.5, dash="dot"),
        name="Global median",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_closed, theta=d_closed, fill="toself",
        fillcolor="rgba(0,100,200,0.10)",
        line=dict(color="#4488CC", width=1.5, dash="dash"),
        name=f"{region or 'Regional'} median",
    ))
    fig.add_trace(go.Scatterpolar(
        r=c_closed, theta=d_closed, fill="toself",
        fillcolor="rgba(232,119,34,0.25)",
        line=dict(color="#E87722", width=2.5),
        name=country_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9), gridcolor="#EEEEEE"),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="#FAFAFA",
        ),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", y=-0.12),
        paper_bgcolor="white",
        height=420,
        margin=dict(l=60, r=60, t=40, b=60),
    )
    return fig


def _build_domain_bars(iso3: str, df: pd.DataFrame, domain: str) -> go.Figure:
    """Grouped horizontal bars: country value vs. global and regional median."""
    idx = df.index[df["iso3"] == iso3]
    if len(idx) == 0:
        return None
    row = df.loc[idx[0]]
    region = row.get("who_region", "")
    country_name = row.get("country_name", iso3)
    region_df = df[df["who_region"] == region]

    cols = [c for c in PROFILE_DOMAINS.get(domain, []) if c in df.columns]
    labels = [INDICATOR_CONFIG.get(c, (c, "", ""))[0] for c in cols]
    units  = [INDICATOR_CONFIG.get(c, (c, "", ""))[2] for c in cols]
    c_vals = [row.get(c) for c in cols]
    g_meds = [df[c].median() for c in cols]
    r_meds = [region_df[c].median() for c in cols]

    # Short labels for y-axis
    short = [l.split("(")[0].strip() for l in labels]
    y_labels = [f"{s} ({u})" if u else s for s, u in zip(short, units)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g_meds, y=y_labels, orientation="h", name="Global median",
        marker_color="#CCCCCC", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=r_meds, y=y_labels, orientation="h", name=f"{region or 'Regional'} median",
        marker_color="#7BA7D0", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=[v if pd.notna(v) else 0 for v in c_vals], y=y_labels, orientation="h",
        name=country_name, marker_color="#E87722",
        text=[f"{v:.1f}" if pd.notna(v) else "n/a" for v in c_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE", title=""),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(280, len(cols) * 75),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        margin=dict(l=190, r=70, t=50, b=30),
        font=dict(family="Arial, sans-serif", color="#1A1A2E", size=11),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Bill_%26_Melinda_Gates_Foundation_logo.svg/320px-Bill_%26_Melinda_Gates_Foundation_logo.svg.png", width=200)
    st.markdown("---")

    st.markdown("### Filters")

    region_filter = st.selectbox(
        "WHO Region",
        [ALL_REGIONS_LABEL] + WHO_REGIONS,
        index=0,
    )

    income_levels = sorted(snap["income_level"].dropna().unique().tolist()) if "income_level" in snap.columns else []
    income_filter = st.selectbox(
        "Income Level",
        ["All income levels"] + income_levels,
        index=0,
    )

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
- **WHO GHO** — anaemia, TB, vaccination, birth outcomes
- **UNICEF / JME** — stunting, wasting, underweight
- **GBD / OWID** — iron, vitamin A, zinc, iodine deficiency
- **World Bank** — mortality, HCI, GDP, food insecurity
- **FAO** — food security
- **FFI / LSFF** — fortification coverage
- **MUMTA** — cohort data (Pakistan)
    """)
    st.markdown("---")
    st.markdown(
        '<div class="sprint-badge">Sprint C Preview</div>'
        '<div class="sprint-meta">June 2026 Learning Session<br>Full build: LTE-led, Fall 2026</div>',
        unsafe_allow_html=True,
    )


# ── Filter snapshot ───────────────────────────────────────────────────────────
def apply_filters(df):
    if region_filter != ALL_REGIONS_LABEL:
        df = df[df["who_region"] == region_filter]
    if income_filter != "All income levels" and "income_level" in df.columns:
        df = df[df["income_level"] == income_filter]
    return df

filtered_snap = apply_filters(snap)
filtered_panel = apply_filters(panel)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Malnutrition Data Commons")
st.markdown(
    '<div class="context-box">'
    '<b>What you\'re looking at:</b> A preview of the integrated data landscape that IDM is building '
    'to enable cross-portfolio analysis — harmonizing micronutrient burden, infectious disease, '
    'and intervention coverage data across priority geographies. '
    'This is an illustrative preview for the June 2026 Learning Session; the full commons build is LTE-led.'
    '</div>',
    unsafe_allow_html=True,
)


# ── Summary metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Countries", f"{filtered_snap['iso3'].nunique()}")
with col2:
    st.metric("Indicators", "33")
with col3:
    val = filtered_snap["anaemia_children_pct"].median()
    st.metric("Median Anaemia <5", f"{val:.1f}%" if pd.notna(val) else "—")
with col4:
    val = filtered_snap["stunting_pct_who"].median()
    st.metric("Median Stunting", f"{val:.1f}%" if pd.notna(val) else "—")
with col5:
    val = filtered_snap["u5_mortality_per1000"].median()
    st.metric("Median U5MR", f"{val:.0f}/1k" if pd.notna(val) else "—")
with col6:
    val = filtered_snap["hci_score"].median()
    st.metric("Median HCI", f"{val:.2f}" if pd.notna(val) else "—")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab layout (restructured for priority-geography narrative) ────────────────
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏗️ Architecture & Sources",
    "🎯 Priority Geographies",
    "🇵🇰 Deep Dive: Pakistan",
    "🧪 Product Impact",
    "🇳🇬 Nigeria Subnational",
    "🌍 Global Context",
])


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 1: Architecture & Sources ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Malnutrition Data Commons — Architecture Preview")
    st.markdown(
        "The Malnutrition Data Commons harmonizes publicly available nutrition, "
        "infectious disease, and intervention coverage data from multiple global "
        "sources into a single queryable layer. This enables cross-portfolio "
        "analysis that no single data source can support alone."
    )

    st.markdown("---")

    # ── Data sources ─────────────────────────────────────────────────────────
    st.markdown("### Data Sources")
    _src_col1, _src_col2 = st.columns(2)
    with _src_col1:
        st.markdown("""
**Global modelled estimates**
- **WHO GHO** — Anaemia prevalence, TB incidence, vaccination coverage, birth outcomes
- **IHME GBD** — Iron, vitamin A, zinc, iodine deficiency; disease burden estimates
- **UNICEF / JME** — Stunting, wasting, underweight (joint WHO/UNICEF/World Bank estimates)
        """)
    with _src_col2:
        st.markdown("""
**Contextual & intervention data**
- **World Bank** — Mortality rates, Human Capital Index, GDP, food insecurity
- **FAO FAOSTAT** — Food supply adequacy, undernourishment prevalence
- **FFI / LSFF** — Wheat flour fortification legislation and coverage proxy
        """)

    st.markdown("""
**Cohort & survey data**
- **DHS Program** — Subnational survey microdata (Nigeria NDHS 2018)
- **MUMTA Cohort** — Multi-arm RCT in Matiari, Sindh, Pakistan (1,884 enrolled;
  maternal nutrition, birth outcomes, infant growth, B. infantis qPCR, TAC enteropathogens,
  fecal inflammation markers)
- **AMANHI Cohort** — WHO multi-country observational study (Pakistan, Bangladesh, Tanzania):
  neonatal B. infantis/B. longum qPCR (n=729), maternal serum bioanalytes incl. CRP &
  ferritin (n=1,937), maternal TAC enteropathogen panel (n=107), taxonomic microbiome profiles
    """)

    st.markdown("---")

    # ── Architecture diagram (text-based) ────────────────────────────────────
    st.markdown("### Architecture")
    st.markdown(
        '<div class="context-box">'
        '<b>Data flow:</b> Raw sources (APIs + CSV exports + cohort data) '
        '→ <code>src/data/</code> pull & process scripts (idempotent) '
        '→ <code>data/processed/</code> harmonized CSVs (ISO3 + year indexed; '
        'cohort-level CSVs for MUMTA & AMANHI) '
        '→ Indicator framework (33 global indicators, 7 domains + cohort-level microbiome/biomarker data) '
        '→ Query layer (this dashboard + static exports for slides)'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        '<div class="context-box">'
        '<b>Two data tiers:</b><br>'
        '① <b>Global ecological</b> — 226 countries × 33 indicators from modelled estimates '
        '(WHO, GBD, UNICEF, World Bank, FAO). Enables geographic targeting and cross-indicator analysis.<br>'
        '② <b>Cohort ground-truth</b> — Individual-level data from MUMTA (Pakistan RCT, n=1,884) '
        'and AMANHI (3-country observational, n=729 neonates + 1,937 bioanalytes). '
        'Provides causal evidence and biomarker resolution not available in global estimates.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Priority geographies ─────────────────────────────────────────────────
    st.markdown("### Priority Geographies")
    st.markdown(
        "The commons is built around three deep-dive countries aligned with "
        "Foundation portfolio priorities. These are where we have the richest "
        "data overlap and the most direct programme relevance."
    )

    _geo_col1, _geo_col2, _geo_col3 = st.columns(3)
    with _geo_col1:
        st.markdown(
            "**🇵🇰 Pakistan**  \n"
            "Two cohort datasets: **MUMTA** (Matiari RCT) and **AMANHI** "
            "(Karachi observational). Richest data overlap: maternal nutrition, "
            "birth outcomes, infant microbiome, serum biomarkers, enteropathogens."
        )
    with _geo_col2:
        st.markdown(
            "**🇮🇳 India**  \n"
            "Largest absolute burden. NFHS-5 provides "
            "state-level ground truth for key nutrition indicators. "
            "Critical for LSFF and MMS scale-up modelling."
        )
    with _geo_col3:
        st.markdown(
            "**🇳🇬 Nigeria**  \n"
            "Sharpest within-country gradient (North-South). "
            "NDHS 2018 state-level data available. "
            "TB and malaria co-occurrence with undernutrition."
        )

    st.markdown("---")
    st.markdown(
        "_This preview uses the most recent available year per indicator per country "
        "(2010-2023 window). The full commons build (LTE-led, Fall 2026) will add "
        "longitudinal harmonization, subnational layers for all priority countries, "
        "and direct API query access._"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 2: Priority Geographies ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Priority Geography Deep-Dive")
    st.markdown(
        "Country profiles for Pakistan, India, and Nigeria — comparing modelled "
        "global estimates to ground-truth survey data where available."
    )

    # ── Country selector + view selector (side by side) ─────────────────────
    _pg_sel_col1, _pg_sel_col2 = st.columns([1, 2])
    with _pg_sel_col1:
        _pg_country = st.selectbox(
            "Select country",
            list(DEEP_DIVE_COUNTRIES.keys()),
            index=0,
            key="pg_country_sel",
        )
    with _pg_sel_col2:
        _pg_view = st.radio(
            "View",
            ["Overview", "Trends"],
            horizontal=True,
            key="pg_view_sel",
        )

    _pg_iso3 = DEEP_DIVE_COUNTRIES[_pg_country]
    _pg_row = snap[snap["iso3"] == _pg_iso3]

    if _pg_row.empty:
        st.warning(f"No data found for {_pg_country} in commons snapshot.")
    else:
        _pg_row = _pg_row.iloc[0]
        _pg_region = _pg_row.get("who_region", "—")

        # ══════════════════════════════════════════════════════════════════════
        # ── Overview: metric cards + radar + domain bars ─────────────────────
        # ══════════════════════════════════════════════════════════════════════
        if _pg_view == "Overview":
            # ── Key indicator metric cards ───────────────────────────────────
            _PG_INDICATORS = [
                ("stunting_pct_who", "Stunting <5", "%", "UNICEF/JME", False),
                ("low_birthweight_pct", "Low Birthweight", "%", "WHO GHO", False),
                ("anaemia_children_pct", "Anaemia <5", "%", "WHO GHO", False),
                ("iron_deficiency_pct", "Iron Deficiency", "%", "IHME GBD", False),
                ("anc4_coverage_pct", "ANC4+ Coverage", "%", "WHO GHO", False),
                ("maternal_mortality_per100k", "Maternal Mortality", "/100k", "WHO/MMEIG", False),
                ("lsff_coverage_proxy_pct", "LSFF Coverage (proxy)", "%", "FFI", False),
            ]

            _GROUND_TRUTH_SOURCES = {
                "PAK": {
                    "stunting_pct_who": "PDHS 2017-18",
                    "low_birthweight_pct": "PDHS 2017-18",
                    "anc4_coverage_pct": "PDHS 2017-18",
                },
                "IND": {
                    "stunting_pct_who": "NFHS-5 2019-21",
                    "anaemia_children_pct": "NFHS-5 2019-21",
                    "anc4_coverage_pct": "NFHS-5 2019-21",
                },
                "NGA": {
                    "stunting_pct_who": "NDHS 2018",
                    "anaemia_children_pct": "NDHS 2018",
                    "anc4_coverage_pct": "NDHS 2018",
                },
            }

            _gt_map = _GROUND_TRUTH_SOURCES.get(_pg_iso3, {})

            st.markdown("### Key Indicators")
            _pg_cols = st.columns(len(_PG_INDICATORS))
            for _i, (_col, _lbl, _unit, _src, _) in enumerate(_PG_INDICATORS):
                _val = _pg_row.get(_col)
                _gmed = snap[_col].median() if _col in snap.columns else np.nan
                _is_gt = _col in _gt_map
                _src_display = _gt_map.get(_col, _src)
                _badge_class = "ground-truth" if _is_gt else "modelled"
                _badge_tip = "National survey" if _is_gt else "Modelled estimate"

                with _pg_cols[_i]:
                    if pd.notna(_val) and pd.notna(_gmed):
                        _delta = _val - _gmed
                        _higher_good = _col in ("anc4_coverage_pct", "lsff_coverage_proxy_pct")
                        st.metric(
                            _lbl,
                            f"{_val:.1f}{_unit}",
                            delta=f"{_delta:+.1f}{_unit} vs. global",
                            delta_color="normal" if _higher_good else "inverse",
                            help=f"Global median: {_gmed:.1f}{_unit}",
                        )
                    else:
                        st.metric(_lbl, f"{_val:.1f}{_unit}" if pd.notna(_val) else "—")
                    st.markdown(
                        f'<span class="source-badge {_badge_class}" title="{_badge_tip}">'
                        f'{_src_display}</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── Radar + domain bars ──────────────────────────────────────────
            _pg_domain_scores = compute_domain_scores(filtered_snap)
            _pg_radar_col, _pg_bars_col = st.columns([1, 1])

            with _pg_radar_col:
                st.markdown(
                    "**Burden profile by domain**  \n"
                    "<small style='color:#666'>0 = best globally &middot; "
                    "100 = worst globally &middot; outward = higher burden</small>",
                    unsafe_allow_html=True,
                )
                _pg_radar = _build_radar_fig(_pg_iso3, _pg_domain_scores, filtered_snap)
                if _pg_radar:
                    st.plotly_chart(_pg_radar, use_container_width=True)
                else:
                    st.info("Insufficient data for radar chart.")

            with _pg_bars_col:
                st.markdown("**Indicator breakdown vs. benchmarks**")
                _pg_domain = st.selectbox(
                    "Domain",
                    list(PROFILE_DOMAINS.keys()),
                    key="pg_domain_sel",
                )
                _pg_bars = _build_domain_bars(_pg_iso3, filtered_snap, _pg_domain)
                if _pg_bars:
                    st.plotly_chart(_pg_bars, use_container_width=True)
                else:
                    st.info("No data for this domain.")

            # ── Pakistan MUMTA callout ───────────────────────────────────────
            if _pg_iso3 == "PAK":
                st.markdown("---")
                st.markdown(
                    '<div class="context-box">'
                    '<b>MUMTA cohort data available</b> — Pakistan is the site of the MUMTA '
                    'prospective birth cohort in Matiari district. See the '
                    '<b>Deep Dive: Pakistan</b> tab for cohort-level '
                    'data on maternal nutrition, birth outcomes, infant growth, and B. infantis '
                    'colonization.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── Nigeria subnational callout ──────────────────────────────────
            if _pg_iso3 == "NGA" and _nga_df is not None:
                st.markdown("---")
                st.markdown(
                    '<div class="context-box">'
                    '<b>Subnational data available</b> — See the '
                    '<b>Nigeria Subnational</b> tab for state-level maps, '
                    'zone comparisons, and coverage vs. burden analysis (NDHS 2018).'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── Full indicator table (expandable) ────────────────────────────
            _pg_region_snap = snap[snap["who_region"] == _pg_region]
            with st.expander("View all indicators"):
                _pg_tbl_rows = []
                for _c, (_lbl, _, _unit) in INDICATOR_CONFIG.items():
                    if _c not in snap.columns:
                        continue
                    _val = _pg_row.get(_c)
                    _gmed = snap[_c].median()
                    _rmed = _pg_region_snap[_c].median() if _c in _pg_region_snap.columns else np.nan
                    _pctile = (
                        snap[_c].rank(pct=True).get(_pg_row.name) * 100
                        if _c in snap.columns and pd.notna(_val) else np.nan
                    )
                    _yr = _pg_row.get(f"{_c}_year", "")
                    _pg_tbl_rows.append({
                        "Indicator": _lbl,
                        "Value": f"{_val:.1f} {_unit}".strip() if pd.notna(_val) else "—",
                        "Global median": f"{_gmed:.1f} {_unit}".strip() if pd.notna(_gmed) else "—",
                        f"{_pg_region} median": f"{_rmed:.1f} {_unit}".strip() if pd.notna(_rmed) else "—",
                        "Global percentile": f"{_pctile:.0f}th" if pd.notna(_pctile) else "—",
                        "Data year": int(_yr) if pd.notna(_yr) and str(_yr) != "" else "—",
                    })
                st.dataframe(
                    pd.DataFrame(_pg_tbl_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Data source legend ───────────────────────────────────────────
            st.markdown("---")
            st.markdown(
                '<small style="color:#555">'
                '<span class="source-badge ground-truth">Survey name</span> = national survey ground-truth &nbsp;&nbsp;'
                '<span class="source-badge modelled">Source</span> = modelled estimate (IHME GBD, WHO GHO, etc.)'
                '</small>',
                unsafe_allow_html=True,
            )

        # ══════════════════════════════════════════════════════════════════════
        # ── Trends: time series for the selected priority country ────────────
        # ══════════════════════════════════════════════════════════════════════
        elif _pg_view == "Trends":
            st.markdown(f"### {_pg_country} — Indicator Trends Over Time")

            # Find indicators with ≥3 data points for this country
            _pg_trend_opts = {
                INDICATOR_CONFIG[c][0]: c
                for c in INDICATOR_CONFIG
                if c in panel.columns
                and panel[(panel["iso3"] == _pg_iso3) & panel[c].notna()].shape[0] >= 3
            }

            if not _pg_trend_opts:
                st.info("No time series data available for this country.")
            else:
                # Default to a sensible set of indicators
                _pg_def_trends = [l for l in _pg_trend_opts if any(
                    kw in l.lower() for kw in ["stunt", "anaemia", "mortality", "hci"]
                )][:3]

                _pg_sel_trends = st.multiselect(
                    "Select indicators to plot",
                    list(_pg_trend_opts.keys()),
                    default=_pg_def_trends if _pg_def_trends else list(_pg_trend_opts.keys())[:3],
                    key="pg_trend_indicators",
                )

                # Option to overlay other priority countries for comparison
                _pg_other_countries = {
                    k: v for k, v in DEEP_DIVE_COUNTRIES.items() if v != _pg_iso3
                }
                _pg_compare = st.multiselect(
                    "Compare with",
                    list(_pg_other_countries.keys()),
                    default=[],
                    key="pg_trend_compare",
                    help="Overlay other priority countries for comparison",
                )
                _pg_compare_iso3 = [_pg_other_countries[n] for n in _pg_compare]
                _pg_all_iso3 = [_pg_iso3] + _pg_compare_iso3

                if _pg_sel_trends:
                    # One chart per selected indicator, with all selected countries overlaid
                    _pg_trend_colors = {
                        "PAK": "#2A9D8F",
                        "IND": "#E87722",
                        "NGA": "#003366",
                    }
                    _pg_country_names = {v: k for k, v in DEEP_DIVE_COUNTRIES.items()}

                    for _ti, _tlbl in enumerate(_pg_sel_trends):
                        _tc = _pg_trend_opts[_tlbl]
                        _tunit = INDICATOR_CONFIG[_tc][2]
                        _tfig = go.Figure()

                        for _iso in _pg_all_iso3:
                            _tpd = (
                                panel[(panel["iso3"] == _iso) & panel[_tc].notna()]
                                .sort_values("year")
                            )
                            if not _tpd.empty:
                                _is_primary = _iso == _pg_iso3
                                _tfig.add_trace(go.Scatter(
                                    x=_tpd["year"],
                                    y=_tpd[_tc],
                                    mode="lines+markers",
                                    name=_pg_country_names.get(_iso, _iso),
                                    line=dict(
                                        width=3 if _is_primary else 1.5,
                                        color=_pg_trend_colors.get(_iso, "#999999"),
                                        dash="solid" if _is_primary else "dot",
                                    ),
                                    marker=dict(size=7 if _is_primary else 4),
                                ))

                        _tfig.update_layout(
                            title=dict(
                                text=f"{_tlbl}" + (f" ({_tunit})" if _tunit else ""),
                                font=dict(size=14),
                            ),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            xaxis=dict(
                                showgrid=True, gridcolor="#EEEEEE",
                                title="Year", tickformat="d",
                            ),
                            yaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
                            legend=dict(
                                orientation="h", yanchor="bottom", y=1.02,
                                font=dict(size=11),
                            ),
                            height=340,
                            margin=dict(l=60, r=40, t=60, b=40),
                            font=dict(family="Arial, sans-serif", color="#1A1A2E"),
                        )
                        st.plotly_chart(_tfig, use_container_width=True)
                else:
                    st.info("Select at least one indicator to view trends.")


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 3: Deep Dive — Pakistan ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Deep Dive: Pakistan — Cohort Data")
    st.markdown(
        "Two complementary Pakistani cohorts provide ground-truth individual-level data "
        "on maternal nutrition, birth outcomes, infant growth, and the gut microbiome."
    )

    _pak_cohort = st.radio(
        "Select cohort",
        ["MUMTA Cohort", "AMANHI Cohort", "Cross-Cohort Comparison"],
        horizontal=True,
        key="pak_cohort_selector",
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # ── AMANHI Cohort ────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    if _pak_cohort == "AMANHI Cohort":
        st.markdown("## AMANHI — WHO Multi-Country Observational Cohort")
        st.markdown(
            "The Alliance for Maternal and Newborn Health Improvement (AMANHI) cohort "
            "studies the maternal gut microbiome across **Pakistan, Bangladesh, and Tanzania**. "
            "Unlike the interventional MUMTA cohort, AMANHI is observational — providing "
            "a broader population baseline with unique biomarkers (serum CRP, ferritin) "
            "not available in MUMTA."
        )

        _amanhi_neo_path = ROOT / "data" / "processed" / "amanhi" / "amanhi_neonatal.csv"
        _amanhi_bio_path = ROOT / "data" / "processed" / "amanhi" / "amanhi_bioanalytes.csv"
        _amanhi_tac_path = ROOT / "data" / "processed" / "amanhi" / "amanhi_maternal_tac.csv"

        if not _amanhi_neo_path.exists():
            st.info(
                "AMANHI data processing required. Run:\n\n"
                "```\npython3 src/data/process_amanhi.py\n```"
            )
        elif not _AMANHI_VIZ_AVAILABLE:
            _err_msg = globals().get("_AMANHI_IMPORT_ERROR", "unknown error")
            st.warning("AMANHI visualization module failed to load.")
            st.code(_err_msg, language="text")
        else:
            _amanhi_neo = pd.read_csv(_amanhi_neo_path)
            _amanhi_bio = pd.read_csv(_amanhi_bio_path) if _amanhi_bio_path.exists() else None
            _amanhi_tac = pd.read_csv(_amanhi_tac_path) if _amanhi_tac_path.exists() else None

            # ── Cohort Overview Metrics ──────────────────────────────────────
            st.markdown("### Cohort Overview")
            _pak_am = _amanhi_neo[_amanhi_neo["site"] == "PAK"]
            _bgd_am = _amanhi_neo[_amanhi_neo["site"] == "BGD"]
            _tza_am = _amanhi_neo[_amanhi_neo["site"] == "TZA"]

            _am1, _am2, _am3 = st.columns(3)
            _am1.metric("Pakistan", f"{len(_pak_am)} neonates")
            _am2.metric("Bangladesh", f"{len(_bgd_am)} neonates")
            _am3.metric("Tanzania", f"{len(_tza_am)} neonates")

            _am4, _am5, _am6, _am7 = st.columns(4)
            _am4.metric("LBW (PAK)", f"{_pak_am['lbw'].mean():.0%}")
            _am5.metric("Preterm (PAK)", f"{_pak_am['preterm'].mean():.0%}")
            _am6.metric("SGA (PAK)", f"{_pak_am['sga'].mean():.0%}")
            _am7.metric("Mean BW (PAK)", f"{_pak_am['birth_weight_g'].mean():,.0f}g")

            if _amanhi_bio is not None:
                _am8, _am9, _am10, _am11 = st.columns(4)
                _am8.metric("Iron Deficient", f"{_amanhi_bio['iron_deficient'].mean():.0%}")
                _am9.metric("CRP Elevated", f"{_amanhi_bio['crp_elevated'].mean():.0%}")
                _am10.metric("Median Ferritin", f"{_amanhi_bio['ferritin_ng_ml'].median():.0f} ng/mL")
                _am11.metric("Median CRP", f"{_amanhi_bio['crp_mg_dl'].median():.2f} mg/dL")

            st.markdown("---")

            # ── Section 1: B. infantis ───────────────────────────────────────
            st.markdown("### B. infantis Colonization")
            st.markdown(
                "Neonatal stool samples tested for *B. infantis* and *B. longum* by qPCR. "
                "AMANHI provides a cross-country snapshot (single neonatal timepoint) across "
                "three sites with varying colonization rates."
            )

            _am_bi_view = st.radio(
                "B. infantis view",
                ["By country", "vs Birth outcomes", "vs Growth"],
                horizontal=True, key="amanhi_bi_view",
            )

            if _am_bi_view == "By country":
                st.plotly_chart(amanhi_binfantis_by_site(_amanhi_neo), use_container_width=True)
                st.info(
                    "**Key finding**: B. infantis colonization varies 2-fold across sites — "
                    "highest in Bangladesh (58%), intermediate in Pakistan (51%), "
                    "lowest in Tanzania (34%). Delivery mode, breastfeeding practices, "
                    "and environmental factors may explain the gradient."
                )
            elif _am_bi_view == "vs Birth outcomes":
                st.plotly_chart(amanhi_binfantis_vs_outcomes(_amanhi_neo), use_container_width=True)
                st.info(
                    "**Interpretation**: Comparing birth outcomes between B. infantis colonized "
                    "and non-colonized neonates. Note: B. infantis status is measured *after* "
                    "birth, so any associations are correlational, not causal."
                )
            else:
                st.plotly_chart(amanhi_binfantis_vs_growth(_amanhi_neo), use_container_width=True)
                st.info(
                    "**Note**: Growth z-scores at birth and 6 months by B. infantis status. "
                    "AMANHI has the advantage of 6-month follow-up anthropometry linked to "
                    "neonatal microbiome status (n=266 Pakistan neonates)."
                )

            st.markdown("---")

            # ── Section 2: Bioanalytes — CRP & Ferritin ──────────────────────
            if _amanhi_bio is not None:
                st.markdown("### Maternal Bioanalytes: CRP & Iron Status")
                st.markdown(
                    "AMANHI uniquely provides **serum CRP** (systemic inflammation marker) "
                    "and **ferritin** (iron stores) — biomarkers not available in the MUMTA "
                    "dataset. This enables direct testing of the inflammation–iron–birth outcome "
                    "pathway."
                )

                _am_bio_view = st.radio(
                    "Bioanalyte view",
                    ["CRP distribution", "Ferritin distribution", "CRP vs Ferritin",
                     "CRP vs Outcomes", "Iron status vs Outcomes"],
                    horizontal=True, key="amanhi_bio_view",
                )

                if _am_bio_view == "CRP distribution":
                    st.plotly_chart(amanhi_crp_distribution(_amanhi_bio), use_container_width=True)
                    st.info(
                        "**Key finding**: 47% of pregnant women have elevated CRP (>0.5 mg/dL), "
                        "indicating widespread subclinical inflammation in this Pakistani population. "
                        "Elevated CRP during pregnancy is associated with adverse birth outcomes "
                        "and may confound ferritin-based iron deficiency assessment."
                    )
                elif _am_bio_view == "Ferritin distribution":
                    st.plotly_chart(amanhi_ferritin_distribution(_amanhi_bio), use_container_width=True)
                    st.info(
                        "**Key finding**: 70% of women are iron deficient (ferritin <15 ng/mL). "
                        "However, ferritin is an acute-phase reactant — in the presence of "
                        "inflammation (47% have elevated CRP), some women classified as iron "
                        "replete may actually be iron deficient with falsely elevated ferritin."
                    )
                elif _am_bio_view == "CRP vs Ferritin":
                    st.plotly_chart(amanhi_crp_ferritin_scatter(_amanhi_bio), use_container_width=True)
                    st.info(
                        "**Interpretation**: The lower-left quadrant (low CRP, low ferritin) represents "
                        "true iron deficiency without inflammation. The upper-right quadrant "
                        "(elevated CRP, higher ferritin) may mask iron deficiency — WHO recommends "
                        "adjusting ferritin thresholds when CRP is elevated."
                    )
                elif _am_bio_view == "CRP vs Outcomes":
                    st.plotly_chart(amanhi_crp_vs_birth_outcomes(_amanhi_bio, _amanhi_neo), use_container_width=True)
                    st.info(
                        "**Note**: This links maternal serum CRP status to neonatal birth outcomes "
                        "via participant ID. Only participants with both bioanalyte and neonatal "
                        "outcome data are included."
                    )
                else:
                    st.plotly_chart(amanhi_ferritin_vs_birth_outcomes(_amanhi_bio, _amanhi_neo), use_container_width=True)
                    st.info(
                        "**Note**: Iron deficiency (ferritin <15 ng/mL) vs birth outcomes. "
                        "This direct biomarker–outcome link is one of the unique contributions "
                        "of the AMANHI dataset to the commons."
                    )

                st.markdown("---")

            # ── Section 3: Maternal TAC Enteropathogens ──────────────────────
            if _amanhi_tac is not None:
                st.markdown("### Maternal Enteropathogens (TAC Panel)")
                st.markdown(
                    "TaqMan Array Card (TAC) qPCR panel detecting 55 enteric pathogen targets "
                    "in maternal stool samples from 107 Pakistani mothers. "
                    "High pathogen burden during pregnancy may contribute to maternal inflammation "
                    "and adverse birth outcomes through environmental enteric dysfunction."
                )

                _am_tac_view = st.radio(
                    "TAC view",
                    ["Top pathogens", "Pathogen burden vs Outcomes"],
                    horizontal=True, key="amanhi_tac_view",
                )

                if _am_tac_view == "Top pathogens":
                    st.plotly_chart(amanhi_tac_top_pathogens(_amanhi_tac), use_container_width=True)
                    st.info(
                        "**Key finding**: EAEC (62%) and EPEC (56%) are the most prevalent "
                        "maternal enteropathogens, consistent with high environmental exposure "
                        "in this setting. Giardia (24%) and ETEC (32%) are also common."
                    )
                else:
                    st.plotly_chart(amanhi_tac_burden_vs_outcomes(_amanhi_tac), use_container_width=True)
                    st.info(
                        "**Interpretation**: Does higher maternal pathogen burden associate with "
                        "worse birth outcomes? Total number of detected pathogens per mother "
                        "compared across outcome groups."
                    )

                st.markdown("---")

            # ── Section 4: B. infantis + B. longum vs Growth Trajectories ────
            st.markdown("### B. infantis & B. longum vs Growth Trajectories")
            st.markdown(
                "AMANHI uniquely links neonatal microbiome status to anthropometry at both "
                "**birth and 6 months** across three countries — enabling us to test whether "
                "B. infantis or B. longum colonization protects against growth faltering. "
                "All 729 neonates have both timepoints."
            )

            _am_growth_view = st.radio(
                "Growth trajectory view",
                ["HAZ by B. infantis", "WAZ by B. infantis", "HAZ by B. longum",
                 "Colonization groups", "Cross-country faltering", "Dose-response (Ct)"],
                horizontal=True, key="amanhi_growth_view",
            )

            if _am_growth_view == "HAZ by B. infantis":
                st.plotly_chart(amanhi_growth_traj_binfantis(_amanhi_neo), use_container_width=True)
                st.warning(
                    "**Counter-intuitive finding**: In Pakistan and Tanzania, B. infantis+ neonates "
                    "show *steeper* growth faltering (larger ΔHAZ decline) than B. infantis− neonates. "
                    "Bangladesh shows a different pattern — B. infantis+ infants maintain their HAZ "
                    "while B. infantis− infants decline. This cross-country inconsistency suggests "
                    "confounding rather than a direct causal effect. Possible confounders include "
                    "breastfeeding patterns, SES, delivery mode, and environmental exposures."
                )
            elif _am_growth_view == "WAZ by B. infantis":
                st.plotly_chart(amanhi_growth_traj_waz(_amanhi_neo), use_container_width=True)
                st.info(
                    "**Weight trajectory**: WAZ (weight-for-age) patterns parallel HAZ — "
                    "B. infantis+ neonates show more weight faltering in Pakistan and Tanzania. "
                    "The consistent HAZ/WAZ pattern suggests an underlying confounder affecting "
                    "overall growth, not just linear growth."
                )
            elif _am_growth_view == "HAZ by B. longum":
                st.plotly_chart(amanhi_growth_traj_blongum(_amanhi_neo), use_container_width=True)
                st.info(
                    "**B. longum trajectory**: Similar counter-intuitive pattern — B. longum+ "
                    "neonates show steeper growth faltering in Pakistan and Tanzania. "
                    "This reinforces the confounding interpretation: colonization with *any* "
                    "Bifidobacterium species tracks with worse growth in these settings."
                )
            elif _am_growth_view == "Colonization groups":
                _am_site_map = {"PAK": "Pakistan", "BGD": "Bangladesh", "TZA": "Tanzania"}
                _am_colon_site = st.selectbox(
                    "Site", ["PAK", "BGD", "TZA"],
                    format_func=lambda s: _am_site_map.get(s, s),
                    key="amanhi_colon_site",
                )
                st.plotly_chart(amanhi_growth_by_colon_group(_amanhi_neo, _am_colon_site), use_container_width=True)
                st.info(
                    "**4-group analysis**: Separating by co-colonization status (Both, B.inf only, "
                    "B.long only, Neither). The 'Neither' group in Pakistan actually shows the "
                    "*least* growth faltering — again suggesting the relationship is driven by "
                    "confounding factors associated with colonization in this environment."
                )
            elif _am_growth_view == "Cross-country faltering":
                st.plotly_chart(amanhi_growth_faltering_xc(_amanhi_neo), use_container_width=True)
                st.info(
                    "**Cross-country comparison**: Mean ΔHAZ (birth→6mo) with standard error bars. "
                    "The direction and magnitude of the B. infantis–growth association varies "
                    "substantially across countries. In Bangladesh, B. infantis+ infants show *less* "
                    "faltering; in Pakistan and Tanzania, *more*. This heterogeneity argues against "
                    "a simple protective mechanism and for site-specific confounding."
                )
            else:  # Dose-response
                _am_dose_site = st.selectbox(
                    "Site", ["PAK", "BGD", "TZA"],
                    format_func=lambda s: {"PAK": "Pakistan", "BGD": "Bangladesh", "TZA": "Tanzania"}.get(s, s),
                    key="amanhi_dose_site",
                )
                st.plotly_chart(amanhi_binfantis_dose_response(_amanhi_neo, _am_dose_site), use_container_width=True)
                st.info(
                    "**Dose-response**: Among B. infantis+ neonates, does higher bacterial load "
                    "(lower Ct value) predict better or worse growth at 6 months? A negative "
                    "Spearman r would mean more bacteria → worse HAZ; positive r → better HAZ. "
                    "This tests whether the counter-intuitive finding holds even within colonized infants."
                )

        st.markdown("---")
        st.markdown(
            "_Source: WHO AMANHI (Alliance for Maternal and Newborn Health Improvement) — "
            "PDO Maternal Gut Microbiome study. Pakistan site (Karachi). "
            "Data: neonatal qPCR (n=266 PAK), bioanalytes (n=1,937), maternal TAC (n=107)._"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ── Cross-Cohort Comparison ──────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    elif _pak_cohort == "Cross-Cohort Comparison":
        st.markdown("## Cross-Cohort: MUMTA vs AMANHI (Pakistan)")
        st.markdown(
            "Comparing findings from two complementary Pakistani cohorts:\n\n"
            "| | MUMTA | AMANHI |\n"
            "|--|-------|--------|\n"
            "| **Design** | RCT (4 arms) | Observational |\n"
            "| **Site** | Matiari, Sindh | Karachi |\n"
            "| **N** | 1,884 enrolled | 266 neonates (PAK) |\n"
            "| **Timepoints** | 10 (birth–12mo) | Birth + 6mo |\n"
            "| **Unique data** | Intervention effects, longitudinal microbiome | Serum CRP/ferritin, cross-country |\n"
        )

        _mumta_bi_path = ROOT / "data" / "processed" / "mumta" / "mumta_binfantis.csv"
        _mumta_tac_path_file = ROOT / "data" / "processed" / "mumta" / "mumta_tac_pathogens.csv"
        _amanhi_neo_path = ROOT / "data" / "processed" / "amanhi" / "amanhi_neonatal.csv"
        _amanhi_tac_path = ROOT / "data" / "processed" / "amanhi" / "amanhi_maternal_tac.csv"

        _have_mumta_cross = _mumta_bi_path.exists() and _MUMTA_VIZ_AVAILABLE
        _have_amanhi_cross = _amanhi_neo_path.exists() and _AMANHI_VIZ_AVAILABLE

        if _have_mumta_cross and _have_amanhi_cross:
            _amanhi_neo = pd.read_csv(_amanhi_neo_path)
            _mumta_bi = pd.read_csv(_mumta_bi_path)

            # ── B. infantis cross-cohort ─────────────────────────────────────
            st.markdown("### B. infantis Colonization: MUMTA vs AMANHI")
            st.markdown(
                "AMANHI captures a single neonatal timepoint; MUMTA provides longitudinal "
                "tracking across multiple infant timepoints. Comparing colonization rates "
                "across cohorts helps validate prevalence estimates."
            )
            st.plotly_chart(
                amanhi_binfantis_cross_cohort(_amanhi_neo, _mumta_bi),
                use_container_width=True,
            )
            st.info(
                "**Interpretation**: AMANHI's neonatal rate (~51%) serves as a baseline; "
                "MUMTA's longitudinal data shows how colonization evolves over the first "
                "year of life. Differences may reflect cohort design (observational vs RCT), "
                "geographic site (Karachi vs Matiari), and timing of sample collection."
            )

            st.markdown("---")

            # ── TAC cross-cohort ─────────────────────────────────────────────
            if _amanhi_tac_path.exists() and _mumta_tac_path_file.exists():
                st.markdown("### Maternal Pathogen Profile: MUMTA vs AMANHI")
                st.markdown(
                    "Both cohorts include TAC panels for Pakistani mothers. Comparing "
                    "pathogen detection rates across sites reveals shared and site-specific "
                    "enteric pathogen exposures."
                )
                _amanhi_tac = pd.read_csv(_amanhi_tac_path)
                _mumta_tac_data = pd.read_csv(_mumta_tac_path_file)
                st.plotly_chart(
                    amanhi_tac_cross_cohort(_amanhi_tac, _mumta_tac_data),
                    use_container_width=True,
                )
                st.info(
                    "**Note**: AMANHI maternal TAC covers 107 mothers (Karachi); "
                    "MUMTA TAC is a substudy of ~200 mothers (Matiari). "
                    "Pathogen naming may differ slightly between labs — "
                    "only overlapping targets are shown."
                )
        else:
            st.warning("Both MUMTA and AMANHI processed data are required for cross-cohort comparison.")

        st.markdown("---")
        st.markdown(
            "_Cross-cohort comparison of MUMTA (Matiari, Sindh) and AMANHI (Karachi). "
            "Different study designs, sites, and populations — interpret differences accordingly._"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ── MUMTA Cohort (original content) ──────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("## MUMTA — Prospective Birth Cohort (RCT)")
        st.markdown(
            "The MUMTA cohort (Matiari, Sindh) provides ground-truth individual-level data "
            "on maternal nutrition, birth outcomes, infant growth, and gut microbiome composition. "
            "This tab compares cohort observations with the modelled country-level estimates "
            "used elsewhere in the commons."
        )

        # Check for processed MUMTA data
        _mumta_summary_path = ROOT / "data" / "processed" / "mumta" / "mumta_cohort_summary.csv"
        _mumta_data_exists = _mumta_summary_path.exists()

        if not _mumta_data_exists:
            st.info(
                "MUMTA data processing required. Run:\n\n"
                "```\npython3 src/data/process_mumta.py\n```\n\n"
                "This will process the raw cohort data and generate summary CSVs "
                "in `data/processed/mumta/`."
            )
        elif not _MUMTA_VIZ_AVAILABLE:
            _err_msg = globals().get("_MUMTA_IMPORT_ERROR", "unknown error")
            st.warning(
                "MUMTA visualization module failed to load. Ensure `src/viz/mumta.py` exists "
                "and exports the required plotting functions."
            )
            st.code(_err_msg, language="text")
        else:
            # Load MUMTA processed data
            _mumta_cohort = pd.read_csv(_mumta_summary_path)
            _mumta_dir = ROOT / "data" / "processed" / "mumta"

            # ── Section 1: Cohort Overview ───────────────────────────────────────
            st.markdown("### Cohort Overview")
            _m = cohort_overview_metrics(_mumta_cohort)
            _ov1, _ov2, _ov3, _ov4 = st.columns(4)
            _ov1.metric("Enrolled", f"{_m.get('n_enrolled', 0):,}")
            _ov2.metric("Live births", f"{_m.get('n_live_births', 0):,}")
            _ov3.metric("Stillbirths", f"{_m.get('n_stillbirths', 0):,}")
            _ov4.metric("Miscarriages", f"{_m.get('n_miscarriages', 0):,}")

            _ov5, _ov6, _ov7, _ov8 = st.columns(4)
            _ov5.metric("LBW (<2500g)", f"{_m.get('lbw_pct', 0)}%")
            _ov6.metric("Preterm (<37wk)", f"{_m.get('preterm_pct', 0)}%")
            _ov7.metric("Mean birthweight", f"{_m.get('mean_birth_weight', 0):,.0f}g")
            _ov8.metric("Stunted at birth", f"{_m.get('stunted_birth_pct', 0)}%")

            _ov9, _ov10, _ov11, _ov12 = st.columns(4)
            _ov9.metric("Anaemia 19wk", f"{_m.get('anaemia_19wk_pct', 0)}%")
            _ov10.metric("Anaemia 32wk", f"{_m.get('anaemia_32wk_pct', 0)}%",
                         delta=f"+{(_m.get('anaemia_32wk_pct',0) - _m.get('anaemia_19wk_pct',0)):.0f}pp",
                         delta_color="inverse")
            _ov11.metric("Iron def 19wk", f"{_m.get('iron_def_19wk_pct', 0)}%")
            _ov12.metric("Iron def 32wk", f"{_m.get('iron_def_32wk_pct', 0)}%")

            st.markdown("---")

            # ── Section 2: Birth Outcomes by Treatment Arm ────────────────────────
            st.markdown("### Birth Outcomes by Treatment Arm")
            _fig_bo = birth_outcomes_by_arm(_mumta_cohort)
            st.plotly_chart(_fig_bo, use_container_width=True)

            st.markdown("---")

            # ── Section 3: Birth Outcomes by Maternal Risk Factor ────────────────
            st.markdown("### Birth Outcomes by Maternal Risk Factor")
            st.markdown(
                "How do birth outcomes differ by maternal nutritional status at enrollment? "
                "Select a risk factor to stratify outcomes."
            )

            _rf_options = {
                "Anaemia (Hb < 11 g/dL at 19 weeks)": "anaemia",
                "BMI category (enrollment)": "bmi",
                "MUAC status (< 23 cm = malnourished)": "muac",
                "Iron deficiency (ferritin < 15 ng/mL at 19 weeks)": "iron_deficiency",
            }
            _rf_col1, _rf_col2 = st.columns([2, 1])
            with _rf_col1:
                _rf_label = st.selectbox(
                    "Risk factor",
                    list(_rf_options.keys()),
                    index=0,
                    key="mumta_rf_sel",
                )
            _rf_key = _rf_options[_rf_label]

            # Grouped bar chart: outcome rates by stratum
            _fig_rf = birth_outcomes_by_risk_factor(_mumta_cohort, risk_factor=_rf_key)
            st.plotly_chart(_fig_rf, use_container_width=True)

            # Birth weight distribution histogram
            _fig_bw = birth_weight_distribution(_mumta_cohort, risk_factor=_rf_key)
            st.plotly_chart(_fig_bw, use_container_width=True)

            # Summary table across all risk factors
            with st.expander("View summary table — all risk factors"):
                _rf_table = adverse_outcome_summary(_mumta_cohort)
                if not _rf_table.empty:
                    st.dataframe(_rf_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Insufficient data for summary table.")

            st.markdown("---")

            # ── Section 4: Maternal Iron & Anemia ────────────────────────────────
            st.markdown("### Maternal Iron & Anemia Trajectory")
            _anemia_path = _mumta_dir / "mumta_maternal_anemia.csv"
            if _anemia_path.exists():
                _anemia_df = pd.read_csv(_anemia_path)
                _fig_anemia = maternal_anemia_trajectory(_anemia_df)
                st.plotly_chart(_fig_anemia, use_container_width=True)
            else:
                st.info("Maternal anemia trajectory data not yet processed.")

            st.markdown("---")

            # ── Section 4: Infant Growth ─────────────────────────────────────────
            st.markdown("### Infant Growth Curves")
            _growth_path = _mumta_dir / "mumta_infant_growth.csv"
            if _growth_path.exists():
                _growth_df = pd.read_csv(_growth_path)
                _growth_metric = st.selectbox(
                    "Z-score metric",
                    ["laz", "waz", "wlz"],
                    format_func={"laz": "Length-for-age (LAZ)",
                                 "waz": "Weight-for-age (WAZ)",
                                 "wlz": "Weight-for-length (WLZ)"}.get,
                    key="mumta_growth_metric",
                )
                _fig_growth = infant_growth_curves(_growth_df, metric=_growth_metric)
                st.plotly_chart(_fig_growth, use_container_width=True)
            else:
                st.info("Infant growth data not yet processed.")

            st.markdown("---")

            # ── Section 5: B. infantis Deep Dive ────────────────────────────────
            st.markdown("### B. infantis Colonization & Protective Effects")
            st.markdown(
                "*B. infantis* is a key commensal bacterium hypothesized to protect against "
                "enteric pathogens and gut inflammation, potentially improving nutrient absorption "
                "and infant growth. This section explores colonization patterns and associations "
                "with pathogens, gut inflammation, and growth in the MUMTA substudy (~100–106 "
                "infants with qPCR data per timepoint)."
            )
            _binfantis_path = _mumta_dir / "mumta_binfantis.csv"
            if _binfantis_path.exists():
                _binfantis_df = pd.read_csv(_binfantis_path)

                _binf_view = st.radio(
                    "B. infantis view",
                    [
                        "Colonization trajectory",
                        "By treatment arm",
                        "vs. Enteric pathogens",
                        "vs. Gut inflammation",
                        "vs. Infant growth",
                    ],
                    horizontal=True,
                    key="mumta_binf_view",
                )

                if _binf_view == "Colonization trajectory":
                    _fig_binf_traj = binfantis_colonization_corrected(_binfantis_df)
                    st.plotly_chart(_fig_binf_traj, use_container_width=True)
                    st.markdown(
                        '<div class="context-box">'
                        '<b>Key pattern:</b> Infant B. infantis colonization rises from 77% at 1–2 months '
                        'to 92% at 5–6 months (among tested). Maternal colonization is much lower '
                        '(2–26%), suggesting infant acquisition is primarily environmental/dietary '
                        'rather than vertical transmission. Note: only ~100–106 participants per '
                        'timepoint were tested in the qPCR substudy.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                elif _binf_view == "By treatment arm":
                    _fig_binf_arm = binfantis_by_arm(_binfantis_df)
                    st.plotly_chart(_fig_binf_arm, use_container_width=True)
                    st.markdown(
                        '<div class="context-box">'
                        '<b>Arm comparison:</b> Maamta (Arm B) shows the highest colonization (92–96%), '
                        'while Maamta+Choline+Nicotinamide (Arm D) shows the lowest (61–86%). '
                        'Small sample sizes per arm (~20–36) limit statistical power, but the '
                        'pattern suggests the choline/nicotinamide combination may alter the gut '
                        'environment in ways that affect B. infantis colonization.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                elif _binf_view == "vs. Enteric pathogens":
                    _tac_path = _mumta_dir / "mumta_tac_pathogens.csv"
                    if _tac_path.exists():
                        _tac_df_binf = pd.read_csv(_tac_path)
                        _fig_binf_path = binfantis_vs_pathogens(_binfantis_df, _tac_df_binf)
                        st.plotly_chart(_fig_binf_path, use_container_width=True)
                        st.markdown(
                            '<div class="context-box">'
                            '<b>Mixed signal:</b> B. infantis+ infants show <i>lower</i> detection of '
                            'EAEC and Shigella/EIEC (consistent with competitive exclusion), but '
                            '<i>higher</i> detection of Campylobacter and Giardia. This may reflect '
                            'confounding by shared environmental exposures rather than a causal effect '
                            '— infants in environments with more diverse microbiome exposure may acquire '
                            'both B. infantis and enteric pathogens. Sample sizes are small (n≈14 for '
                            'B. infantis− group), limiting statistical interpretation.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("TAC pathogen data not available.")

                elif _binf_view == "vs. Gut inflammation":
                    _inflam_path_binf = _mumta_dir / "mumta_gut_inflammation.csv"
                    if _inflam_path_binf.exists():
                        _inflam_df_binf = pd.read_csv(_inflam_path_binf)
                        _fig_binf_inflam = binfantis_vs_inflammation(_binfantis_df, _inflam_df_binf)
                        st.plotly_chart(_fig_binf_inflam, use_container_width=True)
                        st.markdown(
                            '<div class="context-box">'
                            '<b>Counter-intuitive finding:</b> B. infantis+ infants show <i>higher</i> '
                            'median MPO than B. infantis− infants. This likely reflects the small and '
                            'imbalanced comparison group (n≈8–24 for B. infantis−) rather than a true '
                            'harmful effect. MPO levels are extremely high across all infants in this '
                            'cohort (median >1000 ng/mL), consistent with widespread environmental '
                            'enteric dysfunction (EED) in this setting.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("Gut inflammation data not available.")

                else:  # vs. Infant growth
                    _growth_path_binf = _mumta_dir / "mumta_infant_growth.csv"
                    if _growth_path_binf.exists():
                        _growth_df_binf = pd.read_csv(_growth_path_binf)
                        _fig_binf_growth = binfantis_vs_growth(_binfantis_df, _growth_df_binf)
                        st.plotly_chart(_fig_binf_growth, use_container_width=True)
                        st.markdown(
                            '<div class="context-box">'
                            '<b>No clear protective signal:</b> B. infantis+ infants actually show '
                            'slightly <i>lower</i> LAZ than B. infantis− infants at most timepoints, '
                            'though the difference is within standard error bounds. The B. infantis− '
                            'group is very small (n≈8–24), and these are observational comparisons '
                            'subject to confounding. A protective effect of B. infantis may require '
                            'higher colonization density (lower Ct) rather than simple presence/absence.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("Infant growth data not available.")
            else:
                st.info("B. infantis data not yet processed. Run `python src/data/process_mumta.py --force`.")

            st.markdown("---")

            # ── Section 6: Enteric Pathogen Panel (TAC) ──────────────────────────
            st.markdown("### Enteric Pathogen Panel (TaqMan Array Card)")
            st.markdown(
                "TAC data from a substudy (~200 participants) provides qPCR-based detection "
                "of 37 enteric pathogens across maternal and infant stool specimens. "
                "Ct < 35 = detected. These data reveal the scale of environmental enteric "
                "pathogen exposure in this population."
            )
            _tac_path = _mumta_dir / "mumta_tac_pathogens.csv"
            if _tac_path.exists():
                _tac_df = pd.read_csv(_tac_path)

                _tac_view = st.radio(
                    "TAC view",
                    ["Detection heatmap", "Top pathogens", "Pathogen burden trajectory"],
                    horizontal=True,
                    key="mumta_tac_view",
                )

                if _tac_view == "Detection heatmap":
                    _tac_specimen = st.radio(
                        "Specimen",
                        ["maternal", "infant"],
                        horizontal=True,
                        key="mumta_tac_specimen",
                    )
                    _fig_tac_hm = pathogen_detection_heatmap(_tac_df, specimen_type=_tac_specimen)
                    st.plotly_chart(_fig_tac_hm, use_container_width=True)

                elif _tac_view == "Top pathogens":
                    _fig_tac_top = top_pathogens_by_timepoint(_tac_df, n_top=10)
                    st.plotly_chart(_fig_tac_top, use_container_width=True)

                else:  # Pathogen burden trajectory
                    _fig_tac_burden = pathogen_burden_trajectory(_tac_df)
                    st.plotly_chart(_fig_tac_burden, use_container_width=True)

                st.markdown(
                    '<div class="context-box">'
                    '<b>Key observation:</b> Maternal stool shows high baseline pathogen carriage '
                    '(Giardia ~57%, EAEC ~58%, EPEC ~62% at 19 weeks). Infant pathogen acquisition '
                    'accelerates after 3 months, with Giardia rising from 6% at 1-2 months to 43% '
                    'by 12 months — consistent with the environmental enteric dysfunction (EED) '
                    'pathway hypothesized to impair nutrient absorption and linear growth.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("TAC pathogen data not yet processed. Run `python src/data/process_mumta.py --force`.")

            st.markdown("---")

            # ── Section 7: Gut Inflammation & Growth ─────────────────────────────
            st.markdown("### Gut Inflammation & Infant Growth")
            st.markdown(
                "Linking fecal MPO (myeloperoxidase, a marker of gut inflammation / EED) "
                "to concurrent infant length-for-age Z-scores (LAZ). Higher gut inflammation "
                "is hypothesized to impair nutrient absorption and drive growth faltering."
            )
            _inflam_path = _mumta_dir / "mumta_gut_inflammation.csv"
            _growth_path = _mumta_dir / "mumta_infant_growth.csv"
            if _inflam_path.exists() and _growth_path.exists():
                _inflam_df = pd.read_csv(_inflam_path)
                _growth_df_gut = pd.read_csv(_growth_path)
                _fig_mpo_growth = gut_inflammation_vs_growth(_inflam_df, _growth_df_gut)
                st.plotly_chart(_fig_mpo_growth, use_container_width=True)

                st.markdown(
                    '<div class="context-box">'
                    '<b>Relevance to product strategy:</b> If gut inflammation mediates growth '
                    'faltering, then interventions targeting the maternal/infant gut (Azithromycin '
                    'in Arm C, Choline+Nicotinamide in Arm D) may modify this pathway. The MPO–LAZ '
                    'relationship across arms will indicate whether the interventions reduce gut '
                    'inflammation and whether that translates to improved growth.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Gut inflammation or infant growth data not yet processed.")

            st.markdown("---")

            # ── Section 8: Model vs. Cohort Validation ───────────────────────────
            st.markdown("### Model vs. Cohort Comparison")
            st.markdown(
                "How do the modelled country-level estimates (used in the Product Impact tab) "
                "compare against observed cohort-level values from the MUMTA cohort?"
            )
            _pak_row = snap[snap["iso3"] == "PAK"]
            if not _pak_row.empty:
                _fig_validation = model_vs_cohort_comparison(_mumta_cohort, _pak_row.iloc[0])
                st.plotly_chart(_fig_validation, use_container_width=True)
            else:
                st.info("Pakistan snapshot data not available for comparison.")

        st.markdown("---")
        st.markdown(
            "_Source: MUMTA Prospective Birth Cohort, Matiari District, Sindh, Pakistan. "
            "Data collection ongoing. Contact: Nutrition PST._"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 4: Product Impact Scenario Planner ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="context-box">'
        '<b>Portfolio product impact modeller.</b> Toggle products on/off and adjust '
        'parameters to estimate the combined impact of LSFF, MMS, IV-Iron, Maternal Gut, and '
        'Infant Gut (B. infantis) for a selected country. Parameters are two-tier: '
        '<b>🔒 data-informed</b> (drawn from the commons snapshot) and '
        '<b>⚙️ adjustable</b> (evidence-anchored sliders). '
        '<em>Outputs are order-of-magnitude signals, not forecasts.</em>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Country selector (full width) ─────────────────────────────────────────
    _pi_all = (snap[["iso3", "country_name"]].dropna(subset=["country_name"])
               .drop_duplicates().sort_values("country_name"))
    _pi_iso_to_name = dict(zip(_pi_all["iso3"], _pi_all["country_name"]))
    _pi_name_to_iso = dict(zip(_pi_all["country_name"], _pi_all["iso3"]))
    _pi_priority_display = [f"★ {_pi_iso_to_name[i]}" for i in PRIORITY_ISO3 if i in _pi_iso_to_name]
    _pi_other_display    = [n for n in _pi_all["country_name"]
                            if n not in [_pi_iso_to_name.get(i, "") for i in PRIORITY_ISO3]]

    _pi_sel_col, _ = st.columns([2, 3])
    with _pi_sel_col:
        _pi_country_sel = st.selectbox(
            "Country",
            _pi_priority_display + _pi_other_display,
            index=0,
            key="pi_country",
            help="★ = Foundation priority country",
        )
    _pi_country_name = _pi_country_sel.lstrip("★").strip()
    _pi_iso3 = _pi_name_to_iso.get(_pi_country_name, "")
    _pi_row  = snap[snap["iso3"] == _pi_iso3].iloc[0] if _pi_iso3 else None
    _pi_pop  = float(_pop[_pop["iso3"] == _pi_iso3]["population"].values[0]) \
               if _pi_iso3 in _pop["iso3"].values else 10_000_000.0

    # ── Three-column layout ───────────────────────────────────────────────────
    prod_col, chart_col, ctx_col = st.columns([1, 2, 1], gap="medium")

    # ── Left: product toggles + parameter panels ──────────────────────────────
    with prod_col:
        st.markdown("#### 🎛️ Products & Parameters")
        _active_products: list[str] = []
        _all_params: dict[str, dict] = {}

        for pk, reg in PRODUCT_REGISTRY.items():
            color = PRODUCT_COLORS.get(pk, "#888")
            _toggled = st.checkbox(
                f"{reg['emoji']} **{reg['short']}**",
                value=(pk in ["lsff", "mms"]),
                key=f"pi_toggle_{pk}",
                help=reg["description"],
            )
            if _toggled:
                _active_products.append(pk)

            with st.expander(
                f"Parameters — {reg['short']}",
                expanded=(_toggled and pk in ["lsff", "mms"]),
            ):
                if not _toggled:
                    st.caption("(Enable product above to activate)")

                # 🔒 Data-informed params — display as read-only
                if reg.get("data_params") and _pi_row is not None:
                    st.caption("🔒 **Data-informed** (from snapshot)")
                    for dp in reg["data_params"]:
                        _v = _pi_row.get(dp["key"])
                        _display = f"{float(_v):.1f}" if pd.notna(_v) and _v != "" else "—"
                        st.markdown(
                            f"<small style='color:#333'><b>{dp['label']}</b>: "
                            f"<span style='color:#003366;font-weight:700'>{_display}</span>"
                            f"<br><span style='color:#888'>{dp.get('source','')}</span></small>",
                            unsafe_allow_html=True,
                        )

                # MMS: note on IFA coverage proxy
                if pk == "mms":
                    st.markdown(
                        "<small style='color:#888'>ℹ️ IFA coverage is proxied by ANC4+ attendance — "
                        "women attending ≥4 ANC visits are assumed to be the population receiving IFA. "
                        "Actual IFA distribution rates within ANC4 attenders vary by country.</small>",
                        unsafe_allow_html=True,
                    )

                # ⚙️ Adjustable params — sliders
                _params = product_params_defaults(pk)
                if reg.get("adj_params"):
                    st.caption("⚙️ **Adjustable**")
                    for ap in reg["adj_params"]:
                        apk = ap["key"]
                        if ap.get("type") == "select":
                            _params[apk] = st.selectbox(
                                ap["label"],
                                options=ap["options"],
                                index=ap["options"].index(ap["default"]),
                                key=f"pi_{pk}_{apk}",
                                help=ap.get("help", ""),
                                disabled=not _toggled,
                            )
                        else:
                            _step = ap.get("step", 0.5)
                            _default = float(ap.get("default", ap.get("min", 0)))
                            # Use integer slider for whole-number steps
                            if _step >= 1 and _default == int(_default):
                                _params[apk] = st.slider(
                                    ap["label"],
                                    min_value=int(ap.get("min", 0)),
                                    max_value=int(ap.get("max", 100)),
                                    value=int(_default),
                                    step=int(_step),
                                    key=f"pi_{pk}_{apk}",
                                    help=ap.get("help", ""),
                                    disabled=not _toggled,
                                )
                            else:
                                _params[apk] = st.slider(
                                    ap["label"],
                                    min_value=float(ap.get("min", 0)),
                                    max_value=float(ap.get("max", 1)),
                                    value=float(_default),
                                    step=float(_step),
                                    format="%.2f" if _step < 0.1 else "%.1f",
                                    key=f"pi_{pk}_{apk}",
                                    help=ap.get("help", ""),
                                    disabled=not _toggled,
                                )

                _all_params[pk] = _params

            st.markdown("")  # spacer

        # Pull infant_gut program duration through to combined for chart labels
        _gut_years = int(_all_params.get("infant_gut", {}).get("infant_gut_years", 5))

    # ── Compute combined impact ───────────────────────────────────────────────
    if _pi_row is not None and _active_products:
        _combined = compute_combined_impact(
            _active_products, _pi_row, _pi_pop, _all_params
        )
        _combined["program_years"] = _gut_years
    else:
        _combined = dict(
            annual_births=0, lbw_total=0, lbw_by_product={},
            stunting_total_5yr=0, stunting_by_product={},
            maternal_deaths_total=0, maternal_deaths_by_product={},
            lbw_baseline_cases=0, stunting_baseline_5yr=0, maternal_deaths_baseline=0,
            individual={}, program_years=_gut_years,
        )

    # ── Center: headline cards + charts ──────────────────────────────────────
    with chart_col:
        # Headline metric cards
        _hc1, _hc2, _hc3 = st.columns(3)
        _lbw_total   = _combined.get("lbw_total", 0)
        _stunt_total = _combined.get("stunting_total_5yr", 0)
        _mat_total   = _combined.get("maternal_deaths_total", 0)

        # Baselines for % averted
        _lbw_base   = _combined.get("lbw_baseline_cases", 0)
        _stunt_base = _combined.get("stunting_baseline_5yr", 0)
        _mat_base   = _combined.get("maternal_deaths_baseline", 0)

        def _pct_delta(averted: float, baseline: float, suffix: str = "") -> str | None:
            if averted and baseline > 0:
                return f"{averted / baseline * 100:.1f}% of baseline{suffix}"
            return None

        with _hc1:
            st.metric(
                "LBW averted (annual)",
                f"{_lbw_total:,.0f}" if _lbw_total else "—",
                delta=_pct_delta(_lbw_total, _lbw_base),
                delta_color="off",
                help="Estimated low birthweight cases averted per year across active products.",
            )
        with _hc2:
            st.metric(
                f"Stunted children averted ({_gut_years}-yr)",
                f"{_stunt_total:,.0f}" if _stunt_total else "—",
                delta=_pct_delta(_stunt_total, _stunt_base),
                delta_color="off",
                help=f"Estimated cumulative stunting cases averted over {_gut_years} years.",
            )
        with _hc3:
            st.metric(
                "Maternal deaths averted (annual)",
                f"{_mat_total:,.0f}" if _mat_total else "—",
                delta=_pct_delta(_mat_total, _mat_base),
                delta_color="off",
                help="Estimated maternal deaths averted per year (IV-Iron pathway).",
            )

        st.markdown("---")

        if not _active_products:
            st.info("Enable at least one product on the left to see impact projections.")
        elif _pi_row is None:
            st.warning("Country data not available.")
        else:
            # Impact bars chart (all outcomes, all products)
            _fig_bars = impact_bars_chart(_combined, _pi_country_name, height=360)
            st.plotly_chart(_fig_bars, use_container_width=True)


        # ── DALY & cost section ───────────────────────────────────────────────
        if _active_products and _pi_row is not None and _combined.get("lbw_total", 0) + _combined.get("stunting_total_5yr", 0) + _combined.get("maternal_deaths_total", 0) > 0:
            with st.expander("📊 Health economics (back of envelope)", expanded=True):
                _he_l, _he_r = st.columns([1, 1], gap="large")

                with _he_l:
                    st.caption("**DALY weights** (GBD-informed, adjustable)")
                    _daly_lbw = st.slider(
                        "DALYs per LBW case averted",
                        min_value=2.0, max_value=20.0,
                        value=float(DALY_DEFAULTS["daly_per_lbw"]),
                        step=0.5, key="he_daly_lbw",
                        help="GBD 2019 estimate ~8 DALYs per LBW birth (neonatal mortality risk + "
                             "lifelong cognitive/metabolic disability).",
                    )
                    _daly_stunt = st.slider(
                        "DALYs per stunted child averted",
                        min_value=0.5, max_value=10.0,
                        value=float(DALY_DEFAULTS["daly_per_stunted_child"]),
                        step=0.1, key="he_daly_stunt",
                        help="GBD 2019 stunting disability weight + YLL ~2.8 DALYs per case averted.",
                    )
                    _daly_mat = st.slider(
                        "DALYs per maternal death averted",
                        min_value=10.0, max_value=50.0,
                        value=float(DALY_DEFAULTS["daly_per_maternal_death"]),
                        step=1.0, key="he_daly_mat",
                        help="YLL only: ~30 years remaining life expectancy at time of maternal death.",
                    )

                with _he_r:
                    st.caption("**Program cost assumptions** (USD, adjustable)")
                    _cost_inputs = {}
                    _cost_labels = {
                        "lsff":         ("LSFF — $/person/year covered",          "cost_lsff_per_person_yr"),
                        "mms":          ("MMS — marginal $/woman switched",        "cost_mms_per_woman"),
                        "iv_iron":      ("IV-Iron — $/treatment course",           "cost_iv_iron_per_treatment"),
                        "maternal_gut": ("Maternal Gut — $/pregnancy course",      "cost_maternal_gut_per_course"),
                        "infant_gut":   ("Infant Gut — $/infant course",           "cost_infant_gut_per_course"),
                    }
                    for pk in _active_products:
                        if pk not in _cost_labels:
                            continue
                        label, cost_key = _cost_labels[pk]
                        reg = PRODUCT_REGISTRY[pk]
                        default_cost = COST_DEFAULTS[cost_key]
                        _cost_inputs[cost_key] = st.number_input(
                            f"{reg['emoji']} {label}",
                            min_value=0.01, max_value=500.0,
                            value=float(default_cost),
                            step=0.25 if default_cost < 10 else 5.0,
                            format="%.2f",
                            key=f"he_cost_{pk}",
                        )

                # Compute DALYs & costs
                _daly_weights = {
                    "daly_per_lbw":            _daly_lbw,
                    "daly_per_stunted_child":   _daly_stunt,
                    "daly_per_maternal_death":  _daly_mat,
                }
                _he = compute_daly_cost(
                    _combined, _daly_weights, _cost_inputs,
                    _pi_pop, _pi_row,
                )

                st.markdown("---")

                # ── Per-product table (headline) ──────────────────────────────
                _cpd = _he["cost_per_daly"]
                _rows = []
                for pk in _active_products:
                    reg = PRODUCT_REGISTRY.get(pk, {})
                    ind = _combined.get("individual", {}).get(pk, {})
                    _pk_lbw_d   = ind.get("lbw_averted", 0) * _daly_lbw
                    _pk_stunt_d = ind.get("stunted_averted", 0) * _daly_stunt
                    _pk_mat_d   = ind.get("maternal_deaths_averted", 0) * _daly_mat
                    _pk_dalys   = _pk_lbw_d + _pk_stunt_d + _pk_mat_d
                    _pk_cost    = _he["cost_by_product"].get(pk, 0)
                    _pk_cpd     = (_pk_cost / _pk_dalys) if _pk_dalys > 0 else None
                    _rows.append({
                        "Product":           f"{reg.get('emoji','')} {reg.get('short', pk)}",
                        "DALYs averted/yr":  f"{_pk_dalys:,.0f}",
                        "Annual cost (USD)": f"${_pk_cost:,.0f}",
                        "$/DALY":            f"${_pk_cpd:,.0f}" if _pk_cpd is not None else "—",
                    })

                # Combined totals row
                _cost_m = _he["cost_total_usd"] / 1e6
                _cpd_str = f"${_cpd:,.0f}" if _cpd is not None else "—"
                _rows.append({
                    "Product":           "⬛ Combined (sequential)",
                    "DALYs averted/yr":  f"{_he['dalys_total']:,.0f}",
                    "Annual cost (USD)": f"${_cost_m:.1f}M" if _cost_m >= 0.1 else f"${_he['cost_total_usd']:,.0f}",
                    "$/DALY":            _cpd_str,
                })

                st.dataframe(
                    pd.DataFrame(_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                # Cost-effectiveness badge under the table
                if _cpd is not None:
                    if _cpd < 100:
                        _ce_badge = "✓ **Extremely cost-effective** (<$100/DALY)"
                        _ce_color = "#2A9D8F"
                    elif _cpd < 500:
                        _ce_badge = "✓ **Highly cost-effective** ($100–500/DALY)"
                        _ce_color = "#2A9D8F"
                    elif _cpd < 2000:
                        _ce_badge = "~ **Cost-effective** ($500–2,000/DALY)"
                        _ce_color = "#E87722"
                    else:
                        _ce_badge = "↑ **Above standard threshold** (>$2,000/DALY)"
                        _ce_color = "#CC3333"
                    st.markdown(
                        f"<div style='color:{_ce_color};font-size:0.85rem;margin-top:4px'>"
                        f"Combined portfolio: {_ce_badge}</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "<small style='color:#888'>"
                    "DALYs = YLL + YLD. Costs are incremental (additional cost vs. current programme). "
                    "Benchmarks: WHO cost-effectiveness threshold ≈ 1–3× GDP per capita; "
                    "DCP3 rates LSFF at $20–30/DALY, MMS at $30–100/DALY. "
                    "These are order-of-magnitude estimates only."
                    "</small>",
                    unsafe_allow_html=True,
                )

        # Methodology note
        with st.expander("Methodology & limitations"):
            st.markdown("""
**How these estimates are calculated**

Each product uses a product-specific impact function grounded in RCT/meta-analysis evidence:

| Product | Key pathway | Primary outcomes | Evidence anchor |
|---|---|---|---|
| 🌾 LSFF | Coverage delta → iron deficiency reduction → LBW, stunting | LBW, Stunting | Das et al. Cochrane 2019; Bhutta et al. Lancet 2013 |
| 💊 MMS | ANC4 attenders × IFA→MMS switch rate × RRR vs. IFA | LBW, Preterm | SUMMIT meta-analysis 2022; Bourassa et al. 2019 |
| 💉 IV-Iron | Severely anaemic pw with ANC contact × RRR maternal death from haemorrhage | Maternal deaths | Pavord et al. 2015; Govindappagari & Burwick 2019 |
| 🤰 Maternal Gut | Pregnant women reached via ANC × RRR for LBW/preterm (gut barrier → inflammation → nutrient absorption pathway) | LBW, Preterm | Wickens et al. 2017; Odamaki et al. 2020; emerging RCT evidence — ⚠ treat as directional |
| 👶 Infant Gut (B. infantis) | Under-2 infants reached × RRR for stunting (EED reduction via HMO utilisation and gut inflammation pathway) | Stunting | Frese et al. 2017; Nguyen et al. 2021; MALED consortium — ⚠ treat as directional |

**Gut pathway distinction**: The two gut products operate through different mechanisms and windows.
*Maternal Gut* acts prenatally — improving maternal gut barrier integrity and systemic inflammation during pregnancy, with hypothesised downstream effects on birth weight and preterm risk.
*Infant Gut* (e.g. B. infantis EVC001) acts in the first 1–2 years of life — restoring microbiome composition disrupted by environmental enteropathy (EED), improving HMO catabolism and nutrient absorption, with cumulative effects on linear growth and stunting over the programme period.

**Combination logic**: Sequential multiplicative applied in upstream → downstream order.
For shared outcomes (e.g., LBW from LSFF, MMS, and Maternal Gut; stunting from LSFF and Infant Gut):
`remaining_risk *= (1 − RRR_i)` — prevents double-counting.

**Population inputs**: Annual births estimated from WHO regional crude birth rates
(Sub-Saharan Africa ≈ 37/1,000; South Asia ≈ 20/1,000). Infant Gut targets the under-2 population (annual births × 2).

**These are order-of-magnitude signals**, not forecasts. They do not account for:
- Implementation quality, supply chain, or behaviour-change barriers
- Time lags between programme scale-up and observable outcomes
- Counterfactual (what else would change without the intervention)
- Programme interaction effects beyond the sequential multiplicative model
- For gut products especially: limited stunting-specific RCT data; evidence base is early-stage

Comparable estimates: Bhutta et al. *Lancet* 2013 package costing;
GNR 2022 progress tracking; COSTING model (UNICEF/WHO).
            """)

    # ── Right: country context mini-profile ───────────────────────────────────
    with ctx_col:
        st.markdown("#### 📋 Country Context")
        if _pi_row is not None:
            _ctx = country_context_card(_pi_row, _combined)

            st.markdown(
                f"<div style='font-size:0.85rem;color:#003366;font-weight:700;"
                f"margin-bottom:6px'>{_pi_country_name}</div>",
                unsafe_allow_html=True,
            )

            _ctx_items = [
                ("Stunting <5", _ctx["stunting"], "%"),
                ("Low birthweight", _ctx["lbw"], "%"),
                ("Anaemia in pregnancy", _ctx["anaemia_pw"], "%"),
                ("ANC4+ coverage", _ctx["anc4"], "%"),
                ("Iron deficiency", _ctx["iron_def"], "%"),
                ("Maternal mortality", _ctx["mmr"], "/100k"),
                ("LSFF coverage (proxy)", _ctx["lsff_cov"], "%"),
            ]
            for _lbl, _val, _unit in _ctx_items:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:4px 0;border-bottom:1px solid #EEE;font-size:0.82rem'>"
                    f"<span style='color:#555'>{_lbl}</span>"
                    f"<span style='color:#003366;font-weight:700'>{_val}{_unit}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            _ab = _combined.get("annual_births", 0)
            if _ab:
                st.markdown(
                    f"<div style='margin-top:8px;font-size:0.8rem;color:#888'>"
                    f"Est. annual births: <b>{_ab:,.0f}</b></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            # Per-product data coverage note
            st.caption("**Data availability for modelling**")
            for pk, reg in PRODUCT_REGISTRY.items():
                if pk not in _active_products:
                    continue
                missing = [
                    dp["label"].split("(")[0].strip()
                    for dp in reg.get("data_params", [])
                    if pd.isna(_pi_row.get(dp["key"]))
                       or _pi_row.get(dp["key"]) == ""
                ]
                if missing:
                    st.markdown(
                        f"<small style='color:#CC3333'>⚠ {reg['short']}: "
                        f"missing {', '.join(missing)}</small>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<small style='color:#2A9D8F'>✓ {reg['short']}: "
                        f"all data available</small>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Select a country to view context.")


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 5: Nigeria Subnational ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    if _nga_df is None:
        st.warning(
            "Nigeria subnational data not found. "
            "Run `python src/data/pull_dhs_subnational.py` to download it."
        )
    else:
        st.markdown(
            "**Nigeria state-level nutrition profile — NDHS 2018.**  \n"
            "Data from 37 states (36 + FCT Abuja) via the DHS Program public API. "
            "The North-South gradient is one of the starkest within-country patterns "
            "in the dataset: North West/North East states carry 2-3x the stunting "
            "and anaemia burden of South West states, with correspondingly lower "
            "ANC4 and vaccination coverage."
        )

        nga_view_t7 = st.radio(
            "View",
            ["State choropleth", "All indicators grid", "Zone comparison", "Coverage vs. burden scatter"],
            horizontal=True,
            key="nga_view_t7",
        )

        if nga_view_t7 == "State choropleth":
            nga_ind_label_t7 = st.selectbox(
                "Indicator",
                [v[0] for v in NGA_INDICATOR_CONFIG.values()],
                index=0,
                key="nga_ind_sel_t7",
            )
            nga_ind_col_t7 = next(
                k for k, v in NGA_INDICATOR_CONFIG.items() if v[0] == nga_ind_label_t7
            )
            fig_nga_t7 = nigeria_choropleth(_nga_df, _nga_geojson, nga_ind_col_t7, height=560)
            st.plotly_chart(fig_nga_t7, use_container_width=True)

        elif nga_view_t7 == "All indicators grid":
            st.markdown(
                "_Each panel shows one indicator across 37 states. "
                "The spatial pattern is consistent: burden concentrates in the North, "
                "coverage gaps mirror it._"
            )
            fig_nga_grid_t7 = nigeria_multi_map(_nga_df, _nga_geojson, height=860)
            st.plotly_chart(fig_nga_grid_t7, use_container_width=True)

        elif nga_view_t7 == "Zone comparison":
            fig_nga_zones_t7 = nigeria_zone_bars(_nga_df, height=440)
            st.plotly_chart(fig_nga_zones_t7, use_container_width=True)

            # Summary stats
            with st.expander("Zone-level summary table"):
                zone_summary_t7 = (
                    _nga_df.groupby("zone")[list(NGA_INDICATOR_CONFIG.keys())]
                    .agg(["median", "min", "max"])
                    .round(1)
                )
                st.dataframe(zone_summary_t7, use_container_width=True)

        else:  # scatter
            nga_labels_t7 = {v[0].split("(")[0].strip(): k for k, v in NGA_INDICATOR_CONFIG.items()}
            col_x_t7, col_y_t7, col_sz_t7 = st.columns(3)
            with col_x_t7:
                x_lbl_t7 = st.selectbox("X-axis", list(nga_labels_t7.keys()),
                                         index=list(nga_labels_t7.keys()).index("ANC 4+ visits coverage"),
                                         key="nga_x_t7")
            with col_y_t7:
                y_lbl_t7 = st.selectbox("Y-axis", list(nga_labels_t7.keys()),
                                         index=list(nga_labels_t7.keys()).index("Stunting prevalence <5"),
                                         key="nga_y_t7")
            with col_sz_t7:
                sz_opts_t7 = ["(none)"] + list(nga_labels_t7.keys())
                sz_lbl_t7 = st.selectbox("Bubble size", sz_opts_t7,
                                          index=sz_opts_t7.index("Anaemia in children 6–59 mo"),
                                          key="nga_sz_t7")
            fig_nga_sc_t7 = nigeria_scatter(
                _nga_df,
                x_col=nga_labels_t7[x_lbl_t7],
                y_col=nga_labels_t7[y_lbl_t7],
                size_col=nga_labels_t7.get(sz_lbl_t7) if sz_lbl_t7 != "(none)" else None,
                height=540,
            )
            st.plotly_chart(fig_nga_sc_t7, use_container_width=True)

        # State data table
        with st.expander("View state-level data table"):
            display_cols_t7 = ["state_name", "zone"] + [
                c for c in NGA_INDICATOR_CONFIG if c in _nga_df.columns
            ]
            st.dataframe(
                _nga_df[display_cols_t7].sort_values("stunting_pct", ascending=False)
                .reset_index(drop=True).round(1),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            "_Source: Nigeria Demographic and Health Survey 2018 (NDHS 2018) via "
            "[DHS Program API](https://api.dhsprogram.com/). "
            "Boundaries: [geoBoundaries](https://www.geoboundaries.org/) Nigeria ADM1._"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 6: Global Context (merged ecological views + Causal Pathways) ────────
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Global ecological estimates — 226 countries")
    st.markdown(
        '<div class="context-box">'
        '<b>Note:</b> These views use modelled estimates (IHME GBD, WHO GHO) and should be '
        'interpreted as ecological signals, not ground-truth. For priority-country deep-dives '
        'with ground-truth comparisons, see Tab 2 (Priority Geographies).'
        '</div>',
        unsafe_allow_html=True,
    )

    _gc_view = st.selectbox(
        "Select view",
        ["Geographic Overview", "Indicator Explorer", "Co-occurrence", "Trends", "Causal Pathways", "Country Profile"],
        index=0,
        key="gc_view_sel",
    )

    # ── Geographic Overview ──────────────────────────────────────────────────
    if _gc_view == "Geographic Overview":
        col_left, col_right = st.columns([3, 1])
        with col_right:
            map_indicator = st.selectbox(
                "Indicator",
                list(INDICATOR_OPTIONS.keys()),
                index=0,
                key="map_ind",
            )
            map_ind_col = INDICATOR_OPTIONS[map_indicator]

        with col_left:
            fig_gc_map = choropleth_map(filtered_snap, map_ind_col, height=520)
            st.plotly_chart(fig_gc_map, use_container_width=True)

        # Data table
        with st.expander("View data table"):
            display_cols = ["country_name", "iso3", "who_region", map_ind_col]
            display_cols = [c for c in display_cols if c in filtered_snap.columns]
            tbl = filtered_snap[display_cols].dropna(subset=[map_ind_col]).sort_values(map_ind_col, ascending=False)
            st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

    # ── Indicator Explorer (Country Rankings) ────────────────────────────────
    elif _gc_view == "Indicator Explorer":
        col_l, col_r = st.columns([3, 1])
        with col_r:
            bar_indicator = st.selectbox(
                "Indicator",
                list(INDICATOR_OPTIONS.keys()),
                index=0,
                key="bar_ind",
            )
            bar_ind_col = INDICATOR_OPTIONS[bar_indicator]
            n_countries_gc = st.slider("Show top N countries", 10, 40, 20, key="gc_n_countries")

        with col_l:
            fig_gc_bar = burden_bar(filtered_snap, bar_ind_col, n=n_countries_gc, height=max(400, n_countries_gc * 22))
            st.plotly_chart(fig_gc_bar, use_container_width=True)

    # ── Co-occurrence ────────────────────────────────────────────────────────
    elif _gc_view == "Co-occurrence":
        st.markdown(
            "**The key cross-cutting question:** Where do nutritional deficiencies co-occur with "
            "high infectious disease burden? Countries in the upper-right quadrant face a double "
            "burden — and are where integrated investments will have the highest leverage."
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            x_label = st.selectbox("X-axis", list(INDICATOR_OPTIONS.keys()), index=0, key="sc_x")
            x_col = INDICATOR_OPTIONS[x_label]
        with col_b:
            y_options = [k for k in INDICATOR_OPTIONS.keys() if INDICATOR_OPTIONS[k] != x_col]
            default_y = next((k for k, v in INDICATOR_OPTIONS.items() if v == "tb_incidence_per100k"), y_options[0])
            y_label = st.selectbox("Y-axis", y_options, index=y_options.index(default_y) if default_y in y_options else 0, key="sc_y")
            y_col = INDICATOR_OPTIONS[y_label]
        with col_c:
            size_options = ["(none)"] + [k for k in INDICATOR_OPTIONS.keys() if INDICATOR_OPTIONS[k] not in (x_col, y_col)]
            default_size = next((k for k, v in INDICATOR_OPTIONS.items() if v == "stunting_pct_who"), "(none)")
            size_label = st.selectbox("Bubble size", size_options, index=size_options.index(default_size) if default_size in size_options else 0, key="sc_sz")
            size_col = INDICATOR_OPTIONS.get(size_label)

        # Highlight presets
        highlight_preset = st.radio(
            "Highlight countries",
            ["None", "South Asia", "Sub-Saharan Africa", "East Africa"],
            horizontal=True,
            key="gc_highlight",
        )
        highlight = PRIORITY_COUNTRIES.get(highlight_preset, [])

        fig_gc_scatter = cooccurrence_scatter(
            filtered_snap, x_col, y_col,
            size_indicator=size_col,
            highlight_iso3=highlight,
            height=540,
        )
        st.plotly_chart(fig_gc_scatter, use_container_width=True)

        st.markdown(
            "_Dashed lines show global medians. Bubble size = stunting prevalence (where selected). "
            "Countries in the upper-right quadrant face double burden._"
        )

    # ── Trends ───────────────────────────────────────────────────────────────
    elif _gc_view == "Trends":
        st.markdown("Track how indicators have changed over time in selected countries.")

        col_l2, col_r2 = st.columns([3, 1])
        with col_r2:
            trend_indicator = st.selectbox(
                "Indicator",
                list(INDICATOR_OPTIONS.keys()),
                index=0,
                key="trend_ind",
            )
            trend_col = INDICATOR_OPTIONS[trend_indicator]

            # Country selector
            available_countries = (
                filtered_panel[["iso3", "country_name"]]
                .dropna(subset=["country_name"])
                .drop_duplicates()
                .sort_values("country_name")
            )
            country_options = {row["country_name"]: row["iso3"] for _, row in available_countries.iterrows()}

            default_countries = ["Pakistan", "Bangladesh", "India", "Nigeria", "Ethiopia"]
            default_sel = [c for c in default_countries if c in country_options]

            selected_names = st.multiselect(
                "Countries",
                list(country_options.keys()),
                default=default_sel,
                key="gc_trend_countries",
            )
            selected_iso3 = [country_options[n] for n in selected_names if n in country_options]

        with col_l2:
            if selected_iso3:
                fig_gc_trend = trend_lines(filtered_panel, trend_col, selected_iso3, height=480)
                st.plotly_chart(fig_gc_trend, use_container_width=True)
            else:
                st.info("Select at least one country to view trends.")

    # ── Causal Pathways ─────────────────────────────────────────────────────
    elif _gc_view == "Causal Pathways":
        st.markdown(
            "**Cross-indicator hypothesis testing** — 11 hypotheses across the Integrated Nutrition "
            "Impact Framework causal chain, from upstream food systems drivers through nutritional "
            "status, disease burden, and downstream human capital outcomes. "
            "All correlations are Spearman rank; priority countries (India, Pakistan, Bangladesh, "
            "Nigeria, Ethiopia, DRC) are outlined in each figure."
        )

        # Hypothesis registry: label → (function, one-line finding, detail text)
        HYPOTHESES = {
            "H1 — Vaccination gaps predict measles burden": (
                h1_vaccination_measles,
                "MCV dropout (r = +0.39***) is a stronger predictor than first-dose coverage alone.",
                "Countries that achieve good MCV1 but fail to retain children through MCV2 face "
                "outsized measles risk — likely reflecting health system continuity failures. "
                "The dropout gap is actionable: it points to the mid-childhood contact window "
                "(where DTP3, PCV3, and RotaC are also delivered) as the high-leverage intervention point.",
            ),
            "H2 — ANC coverage predicts better birth outcomes": (
                h2_anc_birth_outcomes,
                "ANC4+ coverage predicts lower LBW (r = −0.47***) and preterm birth (r = −0.44***).",
                "ANC4 is both a healthcare coverage metric and the primary delivery platform for "
                "iron/folate supplementation, nutrition counseling, and early complication detection. "
                "Countries with <50% ANC4 coverage face compounded risk across birth outcomes.",
            ),
            "H3 — Malaria amplifies anaemia beyond iron deficiency": (
                h3_malaria_anaemia,
                "Malaria–anaemia correlation (r = +0.79***) is unchanged after controlling for iron deficiency.",
                "Malaria drives anaemia through hemolysis and bone marrow suppression independently of "
                "nutritional iron status. In high-malaria settings, treating iron deficiency alone "
                "will not resolve child anaemia — co-intervention with malaria control is required.",
            ),
            "H4 — HIV–TB syndemic": (
                h4_hiv_tb,
                "HIV prevalence strongly predicts log(TB incidence) (r = +0.50***).",
                "The relationship is concentrated in Sub-Saharan Africa, where both burdens are "
                "co-located alongside high undernutrition — a triple syndemic. Bubble size = stunting "
                "prevalence, revealing that high-HIV/high-TB countries also carry high malnutrition burden.",
            ),
            "H5 — Health system reach vs. nutrition burden": (
                h5_system_vs_burden,
                "Health system composite vs. nutrition burden composite: r = −0.29***.",
                "The relationship is statistically robust but moderate — health system coverage is "
                "necessary but not sufficient. Many countries with moderate coverage still carry high "
                "burden, pointing to upstream food systems and income constraints. The 'crisis quadrant' "
                "(low coverage + high burden) identifies ~25 priority countries.",
            ),
            "H6 — LSFF intervention gap": (
                h6_lsff_gap,
                "20–30 SSA countries show high iron deficiency burden with minimal LSFF coverage.",
                "Iron deficiency and wheat flour fortification coverage are poorly correlated at country "
                "level — the relationship is diffuse, but the gap countries (lower-right quadrant) "
                "represent the clearest LSFF scale-up targets.",
            ),
            "H7 — Vitamin A deficiency vs. measles (null result)": (
                h7_vitamin_a_measles,
                "No country-level correlation between vitamin A deficiency and measles burden (r ≈ 0, n.s.).",
                "This is a meaningful null result, not evidence against vitamin A supplementation. "
                "The individual-level mechanism is well-established, but country-level aggregates are "
                "the wrong unit of analysis: high-burden countries often run active supplementation "
                "programs that suppress observed case counts, obscuring the relationship.",
            ),
            "H8 — Maternal anaemia and ANC → maternal mortality": (
                h8_maternal_anaemia_mortality,
                "Pregnant anaemia (r = +0.75***) and ANC4 (r = −0.74***) are the strongest MMR predictors.",
                "This triangulation supports the causal mechanism: inadequate ANC → untreated anaemia "
                "→ maternal hemorrhage → death. Closing the ANC4 coverage gap would likely drive the "
                "largest single-domain reduction in maternal mortality ratio.",
            ),
            "H9 — Stunting predicts child mortality": (
                h9_undernutrition_child_mortality,
                "Stunting predicts U5MR (r = +0.81***); composite burden reaches r = +0.83***.",
                "Stunting alone explains roughly 66% of under-5 mortality variance at country level. "
                "The nutrition burden composite (adding anaemia, iron deficiency, LBW) adds incremental "
                "power. Countries in the upper-right quadrant are the clearest integrated investment targets.",
            ),
            "H10 — Nutrition burden → human capital and GDP": (
                h10_nutrition_human_capital,
                "Burden vs. HCI: r = −0.85***; burden vs. log(GDP/capita): r = −0.80***.",
                "The strongest relationships in the entire dataset. Nutrition burden predicts HCI and "
                "GDP per capita more strongly than any individual disease or mortality indicator — "
                "the empirical backbone of the 'nutrition as human capital investment' argument.",
            ),
            "H11 — Food insecurity → stunting → child mortality": (
                h11_food_insecurity_pathway,
                "All three causal links confirmed: food insecurity → stunting (r = +0.77***), stunting → U5MR (r = +0.82***).",
                "The full causal chain is empirically traceable at country level. The reduced-form "
                "food insecurity → U5MR correlation (r = +0.83***) is as strong as the mediated path, "
                "consistent with food insecurity operating through multiple nutritional pathways "
                "simultaneously. This supports the upstream food systems investment framing.",
            ),
            "Burden heatmap — top 40 countries across all indicators": (
                burden_heatmap,
                "Multi-domain heatmap: top 40 countries by composite nutrition burden score.",
                "Rows sorted by composite burden; columns span nutritional status, micronutrient "
                "deficiencies, infectious disease, coverage indicators. Red = high burden / low coverage; "
                "green = low burden / high coverage. Foundation priority countries are outlined.",
            ),
        }

        selected_h = st.selectbox(
            "Select hypothesis",
            list(HYPOTHESES.keys()),
            index=0,
            key="gc_insights_sel",
        )

        fn, finding, detail = HYPOTHESES[selected_h]

        # Finding callout
        st.markdown(
            f'<div class="context-box"><b>Key finding:</b> {finding}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Interpretation and portfolio implications"):
            st.markdown(detail)

        # Generate figure — cache by hypothesis name so switching is fast
        @st.cache_data(show_spinner=False)
        def _get_insight_fig(h_name, _df_hash):
            fn_map = {k: v[0] for k, v in HYPOTHESES.items()}
            return fn_map[h_name](_insights_df, show=False)

        with st.spinner("Generating figure..."):
            fig_insight = _get_insight_fig(selected_h, len(_insights_df))

        st.plotly_chart(fig_insight, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "_All correlations are Spearman rank. Partial correlations (H3, H7) use "
            "residualization on stated confounders. Data: most recent available year per country "
            "(2010–2023 window). Full statistical detail: `docs/insights_summary.md`._"
        )

    # ── Country Profile ──────────────────────────────────────────────────────
    elif _gc_view == "Country Profile":
        # Build ordered country list: ★-prefixed priority countries first, then alpha
        _all_ctry = (
            snap[["iso3", "country_name"]].dropna(subset=["country_name"])
            .drop_duplicates().sort_values("country_name")
        )
        _iso_to_name = dict(zip(_all_ctry["iso3"], _all_ctry["country_name"]))
        _name_to_iso = dict(zip(_all_ctry["country_name"], _all_ctry["iso3"]))
        _priority_display = [f"★ {_iso_to_name[i]}" for i in PRIORITY_ISO3 if i in _iso_to_name]
        _other_display    = [n for n in _all_ctry["country_name"]
                             if n not in [_iso_to_name.get(i, "") for i in PRIORITY_ISO3]]

        col_cs, _, _ = st.columns([2, 1, 1])
        with col_cs:
            _selected_display = st.selectbox(
                "Select country",
                _priority_display + _other_display,
                index=0,
                key="profile_country_sel",
                help="★ = Foundation priority country",
            )

        _clean_name  = _selected_display.lstrip("★").strip()
        _profile_iso = _name_to_iso.get(_clean_name)

        if not _profile_iso:
            st.warning("Country not found in dataset.")
        else:
            _pr  = snap[snap["iso3"] == _profile_iso].iloc[0]
            _pname  = _pr.get("country_name", _clean_name)
            _region = _pr.get("who_region", "—")
            _income = _pr.get("income_level", "—")
            _is_priority = _profile_iso in PRIORITY_ISO3

            # Data coverage count
            _num_cols = [c for c in INDICATOR_CONFIG if c in snap.columns]
            _n_avail  = sum(pd.notna(_pr.get(c)) for c in _num_cols)

            # ── Country header ────────────────────────────────────────────────────
            _hdr_cols = st.columns([6, 1])
            with _hdr_cols[0]:
                _badge = (
                    ' <span style="background:#E87722;color:white;padding:2px 10px;'
                    'border-radius:12px;font-size:0.78rem;font-weight:700;">⭐ Priority</span>'
                    if _is_priority else ""
                )
                st.markdown(f"## {_pname}{_badge}", unsafe_allow_html=True)
                st.markdown(
                    f'<span style="color:#555E6E;font-size:0.9rem;">'
                    f'{_region} &nbsp;·&nbsp; {_income} &nbsp;·&nbsp; '
                    f'{_n_avail} of {len(_num_cols)} indicators available</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")

            # ── Key metric cards ──────────────────────────────────────────────────
            _region_snap = snap[snap["who_region"] == _region]
            _mcols = st.columns(len(PROFILE_KEY_METRICS))
            for _i, (_col, _label, _unit, _hib) in enumerate(PROFILE_KEY_METRICS):
                _val  = _pr.get(_col)
                _gmed = snap[_col].median() if _col in snap.columns else np.nan
                with _mcols[_i]:
                    if pd.notna(_val) and pd.notna(_gmed):
                        _delta = _val - _gmed
                        st.metric(
                            _label,
                            f"{_val:.1f}{_unit}",
                            delta=f"{_delta:+.1f}{_unit} vs. global",
                            delta_color="normal" if _hib else "inverse",
                            help=f"Global median: {_gmed:.1f}{_unit}",
                        )
                    else:
                        st.metric(_label, f"{_val:.1f}{_unit}" if pd.notna(_val) else "—")

            st.markdown("---")

            # ── Radar + domain bars ───────────────────────────────────────────────
            _domain_scores = compute_domain_scores(filtered_snap)

            col_radar, col_bars = st.columns([1, 1])

            with col_radar:
                st.markdown(
                    "**Burden profile by domain**  \n"
                    "<small style='color:#666'>0 = best globally &nbsp;·&nbsp; "
                    "100 = worst globally &nbsp;·&nbsp; outward = higher burden</small>",
                    unsafe_allow_html=True,
                )
                _radar_fig = _build_radar_fig(_profile_iso, _domain_scores, filtered_snap)
                if _radar_fig:
                    st.plotly_chart(_radar_fig, use_container_width=True)
                else:
                    st.info("Insufficient data for radar chart.")

            with col_bars:
                st.markdown("**Indicator breakdown vs. benchmarks**")
                _sel_domain = st.selectbox(
                    "Domain",
                    list(PROFILE_DOMAINS.keys()),
                    key="profile_domain_sel",
                )
                _bars_fig = _build_domain_bars(_profile_iso, filtered_snap, _sel_domain)
                if _bars_fig:
                    st.plotly_chart(_bars_fig, use_container_width=True)
                else:
                    st.info("No data for this domain.")

            st.markdown("---")

            # ── Trend section ─────────────────────────────────────────────────────
            st.markdown("**Trends over time**")
            # Only include indicators with ≥ 3 data points for this country
            _trend_opts = {
                INDICATOR_CONFIG[c][0]: c
                for c in INDICATOR_CONFIG
                if c in panel.columns
                and panel[(panel["iso3"] == _profile_iso) & panel[c].notna()].shape[0] >= 3
            }
            if _trend_opts:
                _def_trends = [l for l in _trend_opts if any(
                    kw in l.lower() for kw in ["stunt", "anaemia", "mortality", "hci"]
                )][:2]
                _sel_trends = st.multiselect(
                    "Select indicators",
                    list(_trend_opts.keys()),
                    default=_def_trends if _def_trends else list(_trend_opts.keys())[:2],
                    key="profile_trend_sel",
                )
                if _sel_trends:
                    _trend_fig = go.Figure()
                    _colors = ["#E87722", "#003366", "#009999", "#CC3333", "#6B8E23", "#666699"]
                    for _ti, _tlbl in enumerate(_sel_trends):
                        _tc   = _trend_opts[_tlbl]
                        _unit = INDICATOR_CONFIG[_tc][2]
                        _pd   = (panel[(panel["iso3"] == _profile_iso) & panel[_tc].notna()]
                                 .sort_values("year"))
                        if not _pd.empty:
                            _trend_fig.add_trace(go.Scatter(
                                x=_pd["year"], y=_pd[_tc],
                                mode="lines+markers",
                                name=f"{_tlbl}" + (f" ({_unit})" if _unit else ""),
                                line=dict(width=2.5, color=_colors[_ti % len(_colors)]),
                                marker=dict(size=6),
                            ))
                    _trend_fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(showgrid=True, gridcolor="#EEEEEE", title="Year",
                                   tickformat="d"),
                        yaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    font=dict(size=11)),
                        height=360,
                        margin=dict(l=60, r=40, t=50, b=40),
                        font=dict(family="Arial, sans-serif", color="#1A1A2E"),
                    )
                    st.plotly_chart(_trend_fig, use_container_width=True)
            else:
                st.info("No time series data available for this country.")

            st.markdown("---")

            # ── Nigeria subnational section ───────────────────────────────────────
            if _profile_iso == "NGA" and _nga_df is not None:
                st.markdown("#### 🇳🇬 Subnational breakdown — Nigeria states (NDHS 2018)")
                st.markdown(
                    "Nigeria is the only country in this dataset with state-level data. "
                    "Select a view below to explore the within-country distribution."
                )
                _nga_sub_view = st.radio(
                    "Subnational view",
                    ["State choropleth", "Zone comparison", "Coverage vs. burden scatter"],
                    horizontal=True,
                    key="profile_nga_view",
                )
                if _nga_sub_view == "State choropleth":
                    _nga_ind_lbl = st.selectbox(
                        "Indicator",
                        [v[0] for v in NGA_INDICATOR_CONFIG.values()],
                        key="profile_nga_ind",
                    )
                    _nga_ind_col = next(k for k, v in NGA_INDICATOR_CONFIG.items() if v[0] == _nga_ind_lbl)
                    st.plotly_chart(
                        nigeria_choropleth(_nga_df, _nga_geojson, _nga_ind_col, height=500),
                        use_container_width=True,
                    )
                elif _nga_sub_view == "Zone comparison":
                    st.plotly_chart(nigeria_zone_bars(_nga_df, height=400), use_container_width=True)
                else:
                    st.plotly_chart(nigeria_scatter(_nga_df, height=480), use_container_width=True)
                st.markdown("---")

            # ── Full indicator table ──────────────────────────────────────────────
            with st.expander("View all indicators"):
                _tbl_rows = []
                for _c, (_lbl, _, _unit) in INDICATOR_CONFIG.items():
                    if _c not in snap.columns:
                        continue
                    _val  = _pr.get(_c)
                    _gmed = snap[_c].median()
                    _rmed = _region_snap[_c].median() if _c in _region_snap.columns else np.nan
                    # Percentile rank (higher = higher value; interpret with direction)
                    _pctile = (
                        snap[_c].rank(pct=True).get(_pr.name) * 100
                        if _c in snap.columns and pd.notna(_val) else np.nan
                    )
                    _yr = _pr.get(f"{_c}_year", "")
                    _tbl_rows.append({
                        "Indicator": _lbl,
                        "Value": f"{_val:.1f} {_unit}".strip() if pd.notna(_val) else "—",
                        "Global median": f"{_gmed:.1f} {_unit}".strip() if pd.notna(_gmed) else "—",
                        f"{_region} median": f"{_rmed:.1f} {_unit}".strip() if pd.notna(_rmed) else "—",
                        "Global percentile": f"{_pctile:.0f}th" if pd.notna(_pctile) else "—",
                        "Data year": int(_yr) if pd.notna(_yr) and str(_yr) != "" else "—",
                    })
                st.dataframe(
                    pd.DataFrame(_tbl_rows),
                    use_container_width=True,
                    hide_index=True,
                )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:0.78rem;color:#555E6E;text-align:center;'>"
    "Malnutrition Data Commons | Sprint C Preview | June 2026 Learning Session | "
    "226 countries · 33 indicators · 11 hypotheses | IDM, Bill &amp; Melinda Gates Foundation"
    "</div>",
    unsafe_allow_html=True,
)
