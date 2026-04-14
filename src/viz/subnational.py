"""
Subnational visualization functions — Nigeria state-level choropleth maps.

Data source: NDHS 2018 via DHS Program public API
Boundaries: geoBoundaries Nigeria ADM1

Primary function:
  nigeria_choropleth(df, geojson, indicator, ...)  — single-indicator state map
  nigeria_multi_map(df, geojson)                   — 2×3 grid of headline indicators
  nigeria_north_south_bars(df)                     — zone-level grouped bar comparison

All functions return Plotly figures.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT    = Path(__file__).parents[2]
GEO_DIR = ROOT / "data" / "raw" / "geo"
SUB_DIR = ROOT / "data" / "processed" / "subnational"

# ── Styling ───────────────────────────────────────────────────────────────────
FOUNDATION_BLUE = "#003366"
ACCENT_ORANGE   = "#E87722"
BG_LIGHT        = "#F8F9FA"
FONT            = dict(family="Arial, sans-serif", color="#1A1A2E")

ZONE_COLORS = {
    "North West":  "#CC3333",
    "North East":  "#E87722",
    "North Central":"#F4A261",
    "South West":  "#2A9D8F",
    "South East":  "#457B9D",
    "South South": "#1D6FA4",
}

# Indicator display config: col → (label, colorscale, unit, higher_is_bad)
INDICATOR_CONFIG = {
    "stunting_pct":         ("Stunting prevalence <5 (%)",       "YlOrBr", "%", True),
    "wasting_pct":          ("Wasting prevalence <5 (%)",        "OrRd",   "%", True),
    "underweight_pct":      ("Underweight prevalence <5 (%)",    "PuRd",   "%", True),
    "anaemia_children_pct": ("Anaemia in children 6–59 mo (%)", "Reds",   "%", True),
    "anc4_coverage_pct":    ("ANC 4+ visits coverage (%)",       "Blues",  "%", False),
    "dtp3_coverage_pct":    ("DTP3 vaccination coverage (%)",    "Greens", "%", False),
    "mcv1_coverage_pct":    ("Measles vaccination coverage (%)", "BuGn",   "%", False),
    "low_birthweight_pct":  ("Low birthweight prevalence (%)",   "PuRd",   "%", True),
}


def load_nigeria_data() -> tuple[pd.DataFrame, dict]:
    """Load the wide-format state data and GeoJSON boundary."""
    df = pd.read_csv(SUB_DIR / "nga_states_wide.csv")
    with open(GEO_DIR / "nga_adm1.geojson") as f:
        geojson = json.load(f)
    return df, geojson


def nigeria_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    indicator: str = "stunting_pct",
    title: str | None = None,
    height: int = 520,
    show_labels: bool = True,
) -> go.Figure:
    """
    State-level choropleth map for a single indicator.

    Args:
        df:         Wide-format Nigeria state DataFrame (from load_nigeria_data)
        geojson:    Nigeria ADM1 GeoJSON (from load_nigeria_data)
        indicator:  Column name from INDICATOR_CONFIG
        title:      Override chart title
        height:     Figure height in pixels
        show_labels: Annotate each state with its value
    """
    cfg = INDICATOR_CONFIG.get(indicator, (indicator, "Blues", "%", True))
    label, colorscale, unit, higher_is_bad = cfg

    if not title:
        title = f"Nigeria — {label}<br><sup>NDHS 2018 · State level</sup>"

    plot_df = df.dropna(subset=[indicator]).copy()

    # Reverse colorscale for coverage indicators (high = good = green)
    cs = colorscale + "_r" if not higher_is_bad else colorscale

    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        featureidkey="properties.shapeName",
        locations="state_name",
        color=indicator,
        color_continuous_scale=cs,
        hover_name="state_name",
        hover_data={
            indicator: ":.1f",
            "zone": True,
        },
        labels={indicator: f"{label} ({unit})", "zone": "Zone"},
        title=title,
        height=height,
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showframe=False,
    )

    # State value annotations
    if show_labels:
        _add_state_labels(fig, plot_df, geojson, indicator, unit)

    fig.update_layout(
        paper_bgcolor=BG_LIGHT,
        font=FONT,
        title=dict(font=dict(size=15, color=FOUNDATION_BLUE), x=0.02),
        coloraxis_colorbar=dict(
            title=dict(text=unit, font=dict(size=11)),
            thickness=14,
            len=0.6,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def nigeria_multi_map(
    df: pd.DataFrame,
    geojson: dict,
    height: int = 900,
) -> go.Figure:
    """
    2 × 4 grid of choropleth maps — one per headline indicator.
    Uses Plotly subplots with individual geo axes.
    """
    indicators = list(INDICATOR_CONFIG.keys())
    n_cols = 4
    n_rows = 2

    subplot_titles = []
    for ind in indicators:
        cfg = INDICATOR_CONFIG[ind]
        subplot_titles.append(f"{cfg[0].split('(')[0].strip()}")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "choropleth"}] * n_cols for _ in range(n_rows)],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for i, indicator in enumerate(indicators):
        row = i // n_cols + 1
        col = i % n_cols + 1
        cfg = INDICATOR_CONFIG[indicator]
        label, colorscale, unit, higher_is_bad = cfg
        cs = colorscale + "_r" if not higher_is_bad else colorscale

        plot_df = df.dropna(subset=[indicator])
        vmin = plot_df[indicator].min()
        vmax = plot_df[indicator].max()

        choropleth = go.Choropleth(
            geojson=geojson,
            featureidkey="properties.shapeName",
            locations=plot_df["state_name"],
            z=plot_df[indicator],
            colorscale=cs,
            zmin=vmin,
            zmax=vmax,
            text=plot_df["state_name"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{label}: %{{z:.1f}} {unit}<extra></extra>"
            ),
            showscale=False,
        )
        fig.add_trace(choropleth, row=row, col=col)

        geo_key = f"geo{'' if (row == 1 and col == 1) else (i + 1)}"
        fig.update_layout(**{
            geo_key: dict(
                fitbounds="locations",
                visible=False,
                showframe=False,
            )
        })

    fig.update_layout(
        title=dict(
            text="Nigeria — Subnational nutrition and health profile (NDHS 2018)",
            font=dict(size=16, color=FOUNDATION_BLUE),
            x=0.02,
        ),
        paper_bgcolor=BG_LIGHT,
        font=FONT,
        height=height,
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def nigeria_zone_bars(
    df: pd.DataFrame,
    height: int = 420,
) -> go.Figure:
    """
    Grouped bar chart comparing zone-level medians for key nutritional indicators.
    Highlights the stark North–South divide.
    """
    burden_cols  = ["stunting_pct", "wasting_pct", "underweight_pct", "anaemia_children_pct"]
    coverage_cols = ["anc4_coverage_pct", "dtp3_coverage_pct", "mcv1_coverage_pct"]

    zone_df = (
        df.groupby("zone")[burden_cols + coverage_cols]
        .median()
        .reset_index()
    )
    zones = sorted(zone_df["zone"].tolist(),
                   key=lambda z: zone_df.loc[zone_df["zone"] == z, "stunting_pct"].values[0],
                   reverse=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Burden indicators (%) — higher = worse",
                        "Coverage indicators (%) — higher = better"],
        horizontal_spacing=0.12,
    )

    short_labels = {
        "stunting_pct": "Stunting",
        "wasting_pct": "Wasting",
        "underweight_pct": "Underweight",
        "anaemia_children_pct": "Anaemia",
        "anc4_coverage_pct": "ANC4+",
        "dtp3_coverage_pct": "DTP3",
        "mcv1_coverage_pct": "Measles vacc.",
    }

    for zone in zones:
        color = ZONE_COLORS.get(zone, "#999999")
        row_data = zone_df[zone_df["zone"] == zone].iloc[0]

        fig.add_trace(go.Bar(
            x=[short_labels[c] for c in burden_cols],
            y=[row_data[c] for c in burden_cols],
            name=zone,
            marker_color=color,
            legendgroup=zone,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=[short_labels[c] for c in coverage_cols],
            y=[row_data[c] for c in coverage_cols],
            name=zone,
            marker_color=color,
            legendgroup=zone,
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Nigeria — Geopolitical zone comparison (NDHS 2018 medians)<br>"
                 "<sup>Stark North–South gradient: North carries 2–3× higher burden "
                 "with significantly lower coverage</sup>",
            font=dict(size=15, color=FOUNDATION_BLUE),
            x=0.02,
        ),
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor="white",
        font=FONT,
        height=height,
        legend=dict(title="Zone", orientation="v"),
        xaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", range=[0, 100]),
        yaxis2=dict(showgrid=True, gridcolor="#EEEEEE", range=[0, 100]),
        margin=dict(l=50, r=150, t=80, b=50),
    )
    return fig


def nigeria_scatter(
    df: pd.DataFrame,
    x_col: str = "anc4_coverage_pct",
    y_col: str = "stunting_pct",
    size_col: str | None = "anaemia_children_pct",
    height: int = 500,
) -> go.Figure:
    """
    State-level scatter with zone coloring. Good for showing coverage→outcome relationship.
    """
    x_cfg = INDICATOR_CONFIG.get(x_col, (x_col, "", "%", True))
    y_cfg = INDICATOR_CONFIG.get(y_col, (y_col, "", "%", True))
    x_label = x_cfg[0].split("(")[0].strip()
    y_label = y_cfg[0].split("(")[0].strip()

    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    traces = []
    for zone, color in ZONE_COLORS.items():
        mask = plot_df["zone"] == zone
        sub = plot_df[mask]
        if sub.empty:
            continue
        sizes = sub[size_col].fillna(sub[size_col].median()) if size_col else pd.Series([12] * len(sub))
        # Normalize bubble sizes
        s_min, s_max = sizes.min(), sizes.max()
        scaled = 8 + 22 * (sizes - s_min) / (s_max - s_min + 1e-9)

        hover = (
            "<b>%{text}</b><br>"
            f"{x_label}: %{{x:.1f}}%<br>"
            f"{y_label}: %{{y:.1f}}%<br>"
        )
        if size_col:
            hover += f"{INDICATOR_CONFIG.get(size_col, (size_col,'','%',True))[0].split('(')[0].strip()}: %{{customdata:.1f}}%<br>"
        hover += "<extra></extra>"

        traces.append(go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            mode="markers+text",
            name=zone,
            marker=dict(color=color, size=scaled, opacity=0.8,
                        line=dict(color="white", width=0.8)),
            text=sub["state_name"],
            textposition="top center",
            textfont=dict(size=9),
            customdata=sub[size_col].values if size_col else None,
            hovertemplate=hover,
        ))

    fig = go.Figure(traces)

    # Trend line
    valid = plot_df[[x_col, y_col]].dropna()
    if len(valid) > 3:
        from scipy import stats as _stats
        m, b, *_ = _stats.linregress(valid[x_col], valid[y_col])
        xs = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=xs, y=m * xs + b, mode="lines",
            line=dict(color="#888", dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        r, p = _stats.spearmanr(valid[x_col], valid[y_col])
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        fig.add_annotation(
            x=valid[x_col].max(), y=valid[y_col].max(),
            text=f"Spearman r = {r:.2f} {stars}  (n={len(valid)} states)",
            font=dict(size=11, color="#555"),
            showarrow=False, xanchor="right",
            bgcolor="rgba(255,255,255,0.8)", borderpad=4,
        )

    size_note = ""
    if size_col:
        size_note = f" · Bubble size = {INDICATOR_CONFIG.get(size_col,(size_col,'','%',True))[0].split('(')[0].strip()}"

    fig.update_layout(
        title=dict(
            text=f"Nigeria states — {x_label} vs. {y_label}{size_note}<br>"
                 f"<sup>NDHS 2018 · colored by geopolitical zone</sup>",
            font=dict(size=15, color=FOUNDATION_BLUE), x=0.02,
        ),
        xaxis=dict(
            title=dict(text=f"{x_label} (%)", font=FONT),
            showgrid=True, gridcolor="#EEEEEE",
        ),
        yaxis=dict(
            title=dict(text=f"{y_label} (%)", font=FONT),
            showgrid=True, gridcolor="#EEEEEE",
        ),
        plot_bgcolor="white",
        paper_bgcolor=BG_LIGHT,
        font=FONT,
        legend=dict(title="Zone"),
        height=height,
        margin=dict(l=60, r=160, t=80, b=60),
    )
    return fig


def _add_state_labels(fig, df, geojson, indicator, unit):
    """Overlay state name + value as text annotations on the choropleth."""
    # Build centroid lookup from GeoJSON
    centroids = {}
    for feat in geojson["features"]:
        name = feat["properties"]["shapeName"]
        coords = feat["geometry"]["coordinates"]
        geom_type = feat["geometry"]["type"]
        try:
            if geom_type == "Polygon":
                all_coords = coords[0]
            else:  # MultiPolygon — pick largest ring
                all_coords = max(coords, key=lambda p: len(p[0]))[0]
            lons = [c[0] for c in all_coords]
            lats = [c[1] for c in all_coords]
            centroids[name] = (sum(lons) / len(lons), sum(lats) / len(lats))
        except Exception:
            pass

    for _, row in df.iterrows():
        state = row["state_name"]
        val = row.get(indicator)
        if pd.isna(val) or state not in centroids:
            continue
        lon, lat = centroids[state]
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat],
            mode="text",
            text=[f"{val:.0f}"],
            textfont=dict(size=8, color="#1A1A2E"),
            showlegend=False,
            hoverinfo="skip",
        ))
