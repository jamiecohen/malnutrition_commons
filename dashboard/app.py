"""
Malnutrition Data Commons — Interactive Dashboard
June 2026 Learning Session Preview

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
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
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { color: #003366; font-family: Arial, sans-serif; }
    h2, h3 { color: #003366; }
    .stMetric label { font-size: 0.82rem; color: #555; }
    .stMetric [data-testid="metric-container"] { background: white; border-radius: 8px; padding: 12px; }
    .sprint-badge {
        background: #003366; color: white; padding: 3px 10px;
        border-radius: 12px; font-size: 0.78rem; font-weight: 600;
        display: inline-block; margin-bottom: 0.5rem;
    }
    .context-box {
        background: #EBF4FF; border-left: 4px solid #003366;
        padding: 10px 16px; border-radius: 0 6px 6px 0;
        font-size: 0.88rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    snap = pd.read_csv(ROOT / "data" / "processed" / "commons_snapshot.csv")
    panel = pd.read_csv(ROOT / "data" / "processed" / "commons_panel.csv")
    return snap, panel


snap, panel = load_data()

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
- **WHO GHO** — anaemia, stunting, wasting, TB
- **World Bank / JME** — child malnutrition
- *FAO, UNICEF coming in Phase 1*
    """)
    st.markdown("---")
    st.markdown(
        '<div class="sprint-badge">Sprint C Preview</div>'
        '<div style="font-size:0.78rem;color:#555;">June 2026 Learning Session<br>Full build: LTE-led, Fall 2026</div>',
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
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Countries", f"{filtered_snap['iso3'].nunique()}")
with col2:
    val = filtered_snap["anaemia_children_pct"].median()
    st.metric("Median Anaemia (children)", f"{val:.1f}%" if pd.notna(val) else "—")
with col3:
    val = filtered_snap["anaemia_pregnant_women_pct"].median()
    st.metric("Median Anaemia (pregnant)", f"{val:.1f}%" if pd.notna(val) else "—")
with col4:
    val = filtered_snap["tb_incidence_per100k"].median()
    st.metric("Median TB Incidence", f"{val:.0f}/100k" if pd.notna(val) else "—")
with col5:
    val = filtered_snap["stunting_pct_who"].median()
    st.metric("Median Stunting", f"{val:.1f}%" if pd.notna(val) else "—")

st.markdown("---")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Geographic Map",
    "🔗 Co-Occurrence",
    "📊 Country Rankings",
    "📈 Trends",
])


# ── Tab 1: Choropleth map ─────────────────────────────────────────────────────
with tab1:
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
        fig = choropleth_map(filtered_snap, map_ind_col, height=520)
        st.plotly_chart(fig, use_container_width=True)

    # Data table
    with st.expander("View data table"):
        display_cols = ["country_name", "iso3", "who_region", map_ind_col]
        display_cols = [c for c in display_cols if c in filtered_snap.columns]
        tbl = filtered_snap[display_cols].dropna(subset=[map_ind_col]).sort_values(map_ind_col, ascending=False)
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True)


# ── Tab 2: Co-occurrence scatter ──────────────────────────────────────────────
with tab2:
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
    )
    highlight = PRIORITY_COUNTRIES.get(highlight_preset, [])

    fig2 = cooccurrence_scatter(
        filtered_snap, x_col, y_col,
        size_indicator=size_col,
        highlight_iso3=highlight,
        height=540,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "_Dashed lines show global medians. Bubble size = stunting prevalence (where selected). "
        "Countries in the upper-right quadrant face double burden._"
    )


# ── Tab 3: Country rankings ───────────────────────────────────────────────────
with tab3:
    col_l, col_r = st.columns([3, 1])
    with col_r:
        bar_indicator = st.selectbox(
            "Indicator",
            list(INDICATOR_OPTIONS.keys()),
            index=0,
            key="bar_ind",
        )
        bar_ind_col = INDICATOR_OPTIONS[bar_indicator]
        n_countries = st.slider("Show top N countries", 10, 40, 20)

    with col_l:
        fig3 = burden_bar(filtered_snap, bar_ind_col, n=n_countries, height=max(400, n_countries * 22))
        st.plotly_chart(fig3, use_container_width=True)


# ── Tab 4: Trends ─────────────────────────────────────────────────────────────
with tab4:
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
        )
        selected_iso3 = [country_options[n] for n in selected_names if n in country_options]

    with col_l2:
        if selected_iso3:
            fig4 = trend_lines(filtered_panel, trend_col, selected_iso3, height=480)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Select at least one country to view trends.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:0.78rem;color:#888;text-align:center;'>"
    "Malnutrition Data Commons | Sprint C Preview | June 2026 Learning Session | "
    "Data: WHO GHO, World Bank/JME | IDM, Bill & Melinda Gates Foundation"
    "</div>",
    unsafe_allow_html=True,
)
