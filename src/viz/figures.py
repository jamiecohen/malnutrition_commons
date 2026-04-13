"""
Core visualization functions for the Malnutrition Data Commons.

Produces two primary figures for the June 2026 Learning Session preview:
  1. choropleth_map()  — country-level choropleth for any indicator
  2. cooccurrence_scatter() — scatter of two indicators with bubble sizing
  3. burden_bar()       — ranked bar chart of top-burden countries

All functions return Plotly figures (interactive for dashboard, exportable for slides).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette / style ──────────────────────────────────────────────────────────

FOUNDATION_BLUE = "#003366"
ACCENT_ORANGE   = "#E87722"
ACCENT_TEAL     = "#009999"
ACCENT_RED      = "#CC3333"
BG_LIGHT        = "#F8F9FA"

# WHO region color map
REGION_COLORS = {
    "Africa":           "#E87722",
    "Americas":         "#003366",
    "South-East Asia":  "#CC3333",
    "Europe":           "#666699",
    "Eastern Mediterranean": "#009999",
    "Western Pacific":  "#6B8E23",
}

# Indicator display config: {col: (label, colorscale, unit)}
INDICATOR_CONFIG = {
    # ── Anaemia ────────────────────────────────────────────────────────────────
    "anaemia_children_pct":       ("Anaemia in children <5 (%)",              "Reds",    "%"),
    "anaemia_pregnant_women_pct": ("Anaemia in pregnant women (%)",            "Oranges", "%"),
    "anaemia_women_repro_pct":    ("Anaemia in women 15–49 (%)",               "YlOrRd",  "%"),
    # ── Child malnutrition ────────────────────────────────────────────────────
    "stunting_pct_who":           ("Stunting prevalence <5 (%)",               "YlOrBr",  "%"),
    "wasting_pct":                ("Wasting prevalence <5 (%)",                "OrRd",    "%"),
    "underweight_pct":            ("Underweight prevalence <5 (%)",            "PuRd",    "%"),
    # ── Infectious disease ────────────────────────────────────────────────────
    "tb_incidence_per100k":       ("TB incidence (per 100k pop.)",             "Blues",   "per 100k"),
    "hiv_prevalence_pct":         ("HIV prevalence, adults 15–49 (%)",         "Purples", "%"),
    "malaria_incidence_per1000":  ("Malaria incidence (per 1000 at risk)",     "Greens",  "per 1000"),
    # ── Micronutrient deficiencies (OWID / GBD) ───────────────────────────────
    "vitamin_a_deficiency_pct":   ("Vitamin A deficiency, children <5 (%)",   "YlOrRd",  "%"),
    "zinc_deficiency_pct":        ("Zinc deficiency prevalence (%)",           "BuGn",    "%"),
    "iron_deficiency_pct":        ("Iron deficiency prevalence (%)",           "Reds",    "%"),
    "iodine_deficiency_pct":      ("Iodine deficiency prevalence (%)",         "Blues",   "%"),
    # ── Food fortification ────────────────────────────────────────────────────
    "lsff_coverage_proxy_pct":    ("LSFF wheat flour coverage, proxy (%)",     "YlGn",    "%"),
}


def _prep_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Clean snapshot for visualization: drop rows missing both key coords."""
    return df.copy()


# ── 1. Choropleth map ────────────────────────────────────────────────────────

def choropleth_map(
    df: pd.DataFrame,
    indicator: str,
    title: str | None = None,
    height: int = 480,
) -> go.Figure:
    cfg = INDICATOR_CONFIG.get(indicator, (indicator, "Blues", ""))
    label, colorscale, unit = cfg
    title = title or f"{label} — Most Recent Available Year"

    hover_cols = ["country_name", indicator, "who_region"]
    hover_data = {c: True for c in hover_cols if c in df.columns}
    hover_data[indicator] = ":.1f"

    fig = px.choropleth(
        df.dropna(subset=[indicator]),
        locations="iso3",
        color=indicator,
        color_continuous_scale=colorscale,
        hover_name="country_name",
        hover_data=hover_data,
        labels={indicator: f"{label} ({unit})", "who_region": "WHO Region"},
        title=title,
        height=height,
    )
    fig.update_layout(
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor=BG_LIGHT,
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        title=dict(font=dict(size=15, color=FOUNDATION_BLUE), x=0.02),
        coloraxis_colorbar=dict(
            title=dict(text=unit, font=dict(size=11)),
            thickness=14,
            len=0.6,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#CCCCCC",
            showland=True,
            landcolor="#EEEEEE",
            showocean=True,
            oceancolor="#E8F4F8",
            projection_type="natural earth",
        ),
    )
    return fig


# ── 2. Co-occurrence scatter ─────────────────────────────────────────────────

def cooccurrence_scatter(
    df: pd.DataFrame,
    x_indicator: str,
    y_indicator: str,
    size_indicator: str | None = "stunting_pct_who",
    color_by: str = "who_region",
    highlight_iso3: list[str] | None = None,
    title: str | None = None,
    height: int = 520,
) -> go.Figure:
    x_cfg = INDICATOR_CONFIG.get(x_indicator, (x_indicator, "Blues", ""))
    y_cfg = INDICATOR_CONFIG.get(y_indicator, (y_indicator, "Reds", ""))
    x_label, _, x_unit = x_cfg
    y_label, _, y_unit = y_cfg

    plot_df = df.dropna(subset=[x_indicator, y_indicator]).copy()
    if size_indicator and size_indicator in plot_df.columns:
        plot_df = plot_df.dropna(subset=[size_indicator])
        size_col = size_indicator
        size_cfg = INDICATOR_CONFIG.get(size_indicator, (size_indicator, "", ""))
        size_label = size_cfg[0]
    else:
        plot_df["_size"] = 8
        size_col = "_size"
        size_label = ""

    # Color by WHO region
    color_sequence = list(REGION_COLORS.values())

    title = title or f"{x_label} × {y_label}"

    fig = px.scatter(
        plot_df,
        x=x_indicator,
        y=y_indicator,
        size=size_col,
        color=color_by if color_by in plot_df.columns else None,
        color_discrete_sequence=color_sequence,
        hover_name="country_name" if "country_name" in plot_df.columns else "iso3",
        hover_data={
            x_indicator: ":.1f",
            y_indicator: ":.1f",
            size_col: ":.1f" if size_col != "_size" else False,
            color_by: True if color_by in plot_df.columns else False,
        },
        labels={
            x_indicator: f"{x_label} ({x_unit})",
            y_indicator: f"{y_label} ({y_unit})",
            size_col: size_label,
            color_by: "WHO Region",
        },
        title=title,
        height=height,
        size_max=30,
        opacity=0.75,
    )

    # Highlight specific countries with outline
    if highlight_iso3:
        hi = plot_df[plot_df["iso3"].isin(highlight_iso3)]
        for _, row in hi.iterrows():
            fig.add_trace(go.Scatter(
                x=[row[x_indicator]],
                y=[row[y_indicator]],
                mode="markers+text",
                marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=FOUNDATION_BLUE, width=2)),
                text=[row.get("country_name", row["iso3"])],
                textposition="top center",
                textfont=dict(size=10, color=FOUNDATION_BLUE),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Quadrant lines at medians
    x_med = plot_df[x_indicator].median()
    y_med = plot_df[y_indicator].median()
    for val, axis, orientation in [(x_med, x_indicator, "v"), (y_med, y_indicator, "h")]:
        fig.add_shape(
            type="line",
            x0=val if orientation == "v" else plot_df[x_indicator].min(),
            x1=val if orientation == "v" else plot_df[x_indicator].max(),
            y0=y_med if orientation == "h" else plot_df[y_indicator].min(),
            y1=y_med if orientation == "h" else plot_df[y_indicator].max(),
            line=dict(color="#AAAAAA", dash="dot", width=1),
        )

    fig.update_layout(
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        title=dict(font=dict(size=15, color=FOUNDATION_BLUE), x=0.02),
        legend=dict(title="WHO Region", orientation="v", x=1.02, y=0.98),
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        margin=dict(l=60, r=160, t=60, b=60),
    )
    return fig


# ── 3. Top-burden bar chart ──────────────────────────────────────────────────

def burden_bar(
    df: pd.DataFrame,
    indicator: str,
    n: int = 20,
    color_by_region: bool = True,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    cfg = INDICATOR_CONFIG.get(indicator, (indicator, "Blues", ""))
    label, _, unit = cfg
    title = title or f"Top {n} Countries by {label}"

    plot_df = (
        df[["iso3", "country_name", indicator, "who_region"]]
        .dropna(subset=[indicator, "country_name"])
        .nlargest(n, indicator)
    )

    color_map = REGION_COLORS if color_by_region else None
    fig = px.bar(
        plot_df,
        x=indicator,
        y="country_name",
        orientation="h",
        color="who_region" if color_by_region else None,
        color_discrete_map=color_map,
        labels={indicator: f"{label} ({unit})", "country_name": "", "who_region": "WHO Region"},
        title=title,
        height=height,
        hover_data={"who_region": True, indicator: ":.1f"},
    )
    fig.update_layout(
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        title=dict(font=dict(size=15, color=FOUNDATION_BLUE), x=0.02),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(title="WHO Region"),
        margin=dict(l=150, r=40, t=60, b=40),
    )
    return fig


# ── 4. Time series ───────────────────────────────────────────────────────────

def trend_lines(
    panel: pd.DataFrame,
    indicator: str,
    iso3_list: list[str],
    title: str | None = None,
    height: int = 400,
) -> go.Figure:
    cfg = INDICATOR_CONFIG.get(indicator, (indicator, "Blues", ""))
    label, _, unit = cfg
    title = title or f"{label} Over Time — Selected Countries"

    plot_df = panel[
        (panel["iso3"].isin(iso3_list)) & panel[indicator].notna()
    ].sort_values("year")

    fig = px.line(
        plot_df,
        x="year",
        y=indicator,
        color="iso3",
        hover_name="country_name" if "country_name" in plot_df.columns else "iso3",
        labels={indicator: f"{label} ({unit})", "year": "Year", "iso3": "Country"},
        title=title,
        height=height,
        markers=True,
    )
    fig.update_layout(
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        title=dict(font=dict(size=15, color=FOUNDATION_BLUE), x=0.02),
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig
