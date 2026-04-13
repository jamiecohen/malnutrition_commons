"""
Triple Burden Visualizations — June 2026 Learning Session Slide

Three figures illustrating the geographic co-occurrence of:
  - Anaemia in children <5
  - TB incidence
  - HIV prevalence
  - Malaria incidence

Figures:
  1. composite_burden_map()   — choropleth of normalized composite score
  2. cooccurrence_scatter()   — anaemia × TB, sized by malaria, colored by HIV
  3. burden_profile_bars()    — top-N countries, all 4 indicators side-by-side

Export:
  export_slide_figures()      — saves PNG + HTML for each figure to outputs/slides/
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).parents[2]

# ── Design constants ─────────────────────────────────────────────────────────

FOUNDATION_BLUE  = "#003366"
BG_SLIDE         = "#FFFFFF"
BG_LIGHT         = "#F8F9FA"
GRID_COLOR       = "#E8E8E8"

# Per-indicator brand colors
INDICATOR_COLORS = {
    "anaemia_children_pct":      "#D44D2A",   # warm red-orange
    "tb_incidence_per100k":      "#1A5B8F",   # foundation blue
    "hiv_prevalence_pct":        "#6B3FA0",   # purple
    "malaria_incidence_per1000": "#1A8754",   # green
}

INDICATOR_LABELS = {
    "anaemia_children_pct":      "Anaemia, children <5 (%)",
    "tb_incidence_per100k":      "TB incidence (per 100k)",
    "hiv_prevalence_pct":        "HIV prevalence, adults 15–49 (%)",
    "malaria_incidence_per1000": "Malaria incidence (per 1,000 at risk)",
}

# Foundation priority countries for annotation
PRIORITY_ISO3 = {
    "PAK": "Pakistan", "BGD": "Bangladesh", "IND": "India",
    "ETH": "Ethiopia", "NGA": "Nigeria",    "COD": "DRC",
    "MOZ": "Mozambique", "ZMB": "Zambia",   "MWI": "Malawi",
    "UGA": "Uganda",
}

SLIDE_FONT = dict(family="Arial, Helvetica, sans-serif")


# ── Shared prep ───────────────────────────────────────────────────────────────

BURDEN_INDICATORS = [
    "anaemia_children_pct",
    "tb_incidence_per100k",
    "hiv_prevalence_pct",
    "malaria_incidence_per1000",
]


def prep_burden(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize indicators 0–1 and compute composite burden score."""
    out = df.copy()
    norm_cols = []
    for col in BURDEN_INDICATORS:
        if col not in out.columns:
            continue
        mn, mx = out[col].min(), out[col].max()
        out[f"{col}_norm"] = (out[col] - mn) / (mx - mn) if mx > mn else 0.0
        norm_cols.append(f"{col}_norm")
    out["composite_burden"] = out[norm_cols].mean(axis=1)
    return out


# ── Figure 1: Composite burden choropleth ────────────────────────────────────

def composite_burden_map(df: pd.DataFrame, height: int = 500) -> go.Figure:
    """
    Choropleth where each country is shaded by its composite burden score
    (equal-weighted normalised average of anaemia, TB, HIV, malaria).
    """
    plot_df = prep_burden(df).dropna(subset=["composite_burden"])

    # Custom hover text
    def hover(row):
        lines = [f"<b>{row.get('country_name', row['iso3'])}</b>"]
        for col in BURDEN_INDICATORS:
            if pd.notna(row.get(col)):
                label = INDICATOR_LABELS[col]
                lines.append(f"  {label}: {row[col]:.1f}")
        lines.append(f"  <b>Composite score: {row['composite_burden']:.2f}</b>")
        return "<br>".join(lines)

    plot_df["hover_text"] = plot_df.apply(hover, axis=1)

    fig = go.Figure(go.Choropleth(
        locations=plot_df["iso3"],
        z=plot_df["composite_burden"],
        text=plot_df["hover_text"],
        hoverinfo="text",
        colorscale=[
            [0.0,  "#F7F4F4"],
            [0.15, "#F5C5A3"],
            [0.35, "#E88060"],
            [0.55, "#C94040"],
            [0.75, "#8B1A1A"],
            [1.0,  "#4A0000"],
        ],
        zmin=0, zmax=1,
        colorbar=dict(
            title=dict(text="Composite<br>burden score", font=dict(size=11, **SLIDE_FONT)),
            thickness=14, len=0.6,
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["Low", "", "Medium", "", "High"],
            tickfont=dict(size=10),
        ),
        marker_line_color="#CCCCCC",
        marker_line_width=0.4,
    ))

    # Annotate top-10 burden countries by name
    top10 = plot_df.nlargest(10, "composite_burden")
    for _, row in top10.iterrows():
        if pd.notna(row.get("country_name")):
            fig.add_annotation(
                text=f"  {row['country_name']}",
                showarrow=False,
                font=dict(size=8, color="#333333", **SLIDE_FONT),
                # Geo annotations aren't directly supported; we use a legend-style note instead
            )

    fig.update_layout(
        title=dict(
            text="<b>Integrated Burden Landscape</b><br>"
                 "<span style='font-size:12px;color:#555'>Composite score: anaemia · TB · HIV · malaria (equal-weighted, normalized)</span>",
            font=dict(size=17, color=FOUNDATION_BLUE, **SLIDE_FONT),
            x=0.02, y=0.97,
        ),
        paper_bgcolor=BG_SLIDE,
        font=dict(**SLIDE_FONT),
        height=height,
        margin=dict(l=0, r=0, t=80, b=10),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#CCCCCC",
            showland=True,
            landcolor="#F0F0F0",
            showocean=True,
            oceancolor="#EAF3FB",
            projection_type="natural earth",
            showlakes=False,
        ),
    )
    return fig


# ── Figure 2: Co-occurrence scatter ──────────────────────────────────────────

def cooccurrence_scatter(df: pd.DataFrame, height: int = 560) -> go.Figure:
    """
    Scatter plot: Anaemia (x) × TB incidence (y)
    Bubble size = malaria incidence
    Bubble color = HIV prevalence (continuous)
    Annotate Foundation priority countries + top burden countries
    """
    plot_df = prep_burden(df).dropna(
        subset=["anaemia_children_pct", "tb_incidence_per100k"]
    ).copy()

    # Size: malaria where available, otherwise fixed size
    if "malaria_incidence_per1000" in plot_df.columns:
        plot_df["_size"] = plot_df["malaria_incidence_per1000"].fillna(5).clip(lower=5)
    else:
        plot_df["_size"] = 5.0
    plot_df["_size_scaled"] = np.sqrt(plot_df["_size"]) * 1.5  # sqrt scaling for visual

    # HIV color: use a 0–25 range to keep the scale readable
    plot_df["_hiv"] = plot_df["hiv_prevalence_pct"].fillna(0) if "hiv_prevalence_pct" in plot_df.columns else 0.0

    # Hover
    def hover(row):
        name = row.get("country_name", row["iso3"])
        lines = [f"<b>{name}</b>"]
        lines.append(f"Anaemia (children <5): {row['anaemia_children_pct']:.1f}%")
        lines.append(f"TB incidence: {row['tb_incidence_per100k']:.0f} per 100k")
        if pd.notna(row.get("hiv_prevalence_pct")):
            lines.append(f"HIV prevalence: {row['hiv_prevalence_pct']:.1f}%")
        if pd.notna(row.get("malaria_incidence_per1000")):
            lines.append(f"Malaria incidence: {row['malaria_incidence_per1000']:.0f} per 1k")
        lines.append(f"Composite burden: {row['composite_burden']:.2f}")
        return "<br>".join(lines)

    plot_df["hover_text"] = plot_df.apply(hover, axis=1)

    fig = go.Figure()

    # Main scatter — all countries
    fig.add_trace(go.Scatter(
        x=plot_df["anaemia_children_pct"],
        y=plot_df["tb_incidence_per100k"],
        mode="markers",
        marker=dict(
            size=plot_df["_size_scaled"],
            color=plot_df["_hiv"],
            colorscale=[
                [0.0,  "#D8EAF8"],
                [0.15, "#9EC8E8"],
                [0.35, "#C09FD0"],
                [0.6,  "#8844AA"],
                [1.0,  "#3D1560"],
            ],
            cmin=0, cmax=25,
            colorbar=dict(
                title=dict(text="HIV prevalence<br>adults 15–49 (%)", font=dict(size=10, **SLIDE_FONT)),
                thickness=12, len=0.55, x=1.01,
                tickvals=[0, 5, 10, 15, 20, 25],
                tickfont=dict(size=9),
            ),
            line=dict(color="rgba(255,255,255,0.6)", width=0.5),
            opacity=0.82,
            sizemode="diameter",
        ),
        text=plot_df["hover_text"],
        hoverinfo="text",
        showlegend=False,
    ))

    # Annotate Foundation priority countries + top 12 composite
    top_composite = plot_df.nlargest(12, "composite_burden")["iso3"].tolist()
    annotate_iso3 = set(PRIORITY_ISO3.keys()) | set(top_composite)

    annotated = plot_df[plot_df["iso3"].isin(annotate_iso3)].dropna(
        subset=["anaemia_children_pct", "tb_incidence_per100k"]
    )

    for _, row in annotated.iterrows():
        is_priority = row["iso3"] in PRIORITY_ISO3
        name = row.get("country_name", row["iso3"])
        # Shorten some long names
        short = {"Congo, Dem. Rep.": "DRC", "Central African Republic": "CAR",
                 "Papua New Guinea": "PNG", "Sierra Leone": "S. Leone",
                 "Cote d'Ivoire": "Côte d'Ivoire"}
        name = short.get(name, name)

        fig.add_annotation(
            x=row["anaemia_children_pct"],
            y=row["tb_incidence_per100k"],
            text=name,
            showarrow=False,
            xshift=8,
            yshift=5,
            font=dict(
                size=9 if not is_priority else 10,
                color=FOUNDATION_BLUE if is_priority else "#444444",
                family="Arial, Helvetica, sans-serif",
            ),
        )
        if is_priority:
            # Add a bold ring around priority countries
            fig.add_trace(go.Scatter(
                x=[row["anaemia_children_pct"]],
                y=[row["tb_incidence_per100k"]],
                mode="markers",
                marker=dict(
                    size=row["_size_scaled"] + 6,
                    color="rgba(0,0,0,0)",
                    line=dict(color=FOUNDATION_BLUE, width=2),
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Quadrant lines at global medians
    x_med = plot_df["anaemia_children_pct"].median()
    y_med = plot_df["tb_incidence_per100k"].median()
    x_max = plot_df["anaemia_children_pct"].max() * 1.05
    y_max = plot_df["tb_incidence_per100k"].max() * 1.05

    for x0, x1, y0, y1 in [
        (x_med, x_med, 0, y_max),       # vertical
        (0, x_max, y_med, y_med),        # horizontal
    ]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color="#BBBBBB", dash="dot", width=1))

    # Quadrant labels
    pad_x, pad_y = x_max * 0.03, y_max * 0.03
    fig.add_annotation(x=x_max - pad_x, y=y_max - pad_y,
                       text="<b>High nutrition<br>+ High infectious</b>",
                       showarrow=False, xanchor="right", yanchor="top",
                       font=dict(size=9, color="#AA3333", **SLIDE_FONT),
                       bgcolor="rgba(255,240,240,0.8)", borderpad=4)
    fig.add_annotation(x=pad_x, y=pad_y,
                       text="<b>Low burden</b>",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=9, color="#338833", **SLIDE_FONT),
                       bgcolor="rgba(240,255,240,0.8)", borderpad=4)

    # Bubble size legend (manual)
    for size_val, label in [(50, "50"), (200, "200"), (500, "500+")]:
        s = np.sqrt(size_val) * 1.5
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=s, color="#AAAAAA", line=dict(color="#888888", width=1),
                        sizemode="diameter"),
            name=f"Malaria {label}/1k",
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Nutritional Burden × Infectious Disease Co-Occurrence</b><br>"
                 "<span style='font-size:12px;color:#555'>"
                 "X = Anaemia prevalence (children <5)  ·  Y = TB incidence  ·  "
                 "Bubble size = Malaria incidence  ·  Color = HIV prevalence"
                 "</span>",
            font=dict(size=16, color=FOUNDATION_BLUE, **SLIDE_FONT),
            x=0.02, y=0.97,
        ),
        paper_bgcolor=BG_SLIDE,
        plot_bgcolor="white",
        font=dict(**SLIDE_FONT, size=11),
        height=height,
        xaxis=dict(
            title=dict(text="Anaemia prevalence, children <5 (%)", font=dict(size=12)),
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[0, x_max],
        ),
        yaxis=dict(
            title=dict(text="TB incidence (per 100,000 population)", font=dict(size=12)),
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[0, y_max],
        ),
        legend=dict(
            title=dict(text="Malaria incidence", font=dict(size=10)),
            orientation="v", x=1.12, y=0.3,
            font=dict(size=9),
        ),
        margin=dict(l=70, r=180, t=90, b=70),
    )
    return fig


# ── Figure 3: Burden profile bar chart ───────────────────────────────────────

def burden_profile_bars(df: pd.DataFrame, n: int = 15, height: int = 560) -> go.Figure:
    """
    Horizontal grouped bar chart for top-N countries.
    Each indicator shown as a separate bar (normalized 0–1), side by side.
    Countries ordered by composite score.
    """
    plot_df = prep_burden(df).dropna(subset=["composite_burden", "country_name"]).copy()
    top = plot_df.nlargest(n, "composite_burden").sort_values("composite_burden", ascending=True)

    # Shorten long country names
    short_names = {
        "Congo, Dem. Rep.": "DRC", "Central African Republic": "CAR",
        "Papua New Guinea": "PNG", "Sierra Leone": "S. Leone",
        "Cote d'Ivoire": "Côte d'Ivoire", "South Sudan": "S. Sudan",
    }
    top["display_name"] = top["country_name"].replace(short_names)

    # Mark Foundation priority countries
    top["is_priority"] = top["iso3"].isin(PRIORITY_ISO3)

    fig = go.Figure()

    for col in BURDEN_INDICATORS:
        norm_col = f"{col}_norm"
        if norm_col not in top.columns:
            continue
        # Raw value for hover
        raw = top[col].round(1).astype(str).where(top[col].notna(), "N/A")
        label_short = INDICATOR_LABELS[col].split(" (")[0]  # strip units for legend

        fig.add_trace(go.Bar(
            y=top["display_name"],
            x=top[norm_col].fillna(0),
            name=label_short,
            orientation="h",
            marker_color=INDICATOR_COLORS[col],
            opacity=0.88,
            customdata=raw,
            hovertemplate=f"<b>%{{y}}</b><br>{INDICATOR_LABELS[col]}: %{{customdata}}<extra></extra>",
        ))

    # Composite score line (secondary axis simulation via scatter)
    fig.add_trace(go.Scatter(
        y=top["display_name"],
        x=top["composite_burden"],
        mode="markers+lines",
        name="Composite score",
        marker=dict(color="#333333", size=6, symbol="diamond"),
        line=dict(color="#333333", width=1.5, dash="dot"),
        hovertemplate="<b>%{y}</b><br>Composite: %{x:.2f}<extra></extra>",
    ))

    # Bold priority country y-tick labels via annotation hack
    for _, row in top[top["is_priority"]].iterrows():
        fig.add_annotation(
            x=-0.01, y=row["display_name"],
            text="★",
            showarrow=False, xanchor="right",
            font=dict(size=10, color=FOUNDATION_BLUE),
            xref="paper",
        )

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"<b>Multi-Indicator Burden Profile — Top {n} Countries</b><br>"
                 "<span style='font-size:12px;color:#555'>"
                 "Bars = normalized burden (0–1 scale within each indicator)  ·  "
                 "◆ = composite score  ·  ★ = Foundation priority country"
                 "</span>",
            font=dict(size=16, color=FOUNDATION_BLUE, **SLIDE_FONT),
            x=0.02, y=0.98,
        ),
        paper_bgcolor=BG_SLIDE,
        plot_bgcolor="white",
        font=dict(**SLIDE_FONT, size=11),
        height=height,
        xaxis=dict(
            title=dict(text="Normalized burden score (0 = lowest, 1 = highest globally)", font=dict(size=11)),
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[0, 1.05],
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        legend=dict(
            orientation="h", x=0.02, y=-0.12,
            font=dict(size=10),
            traceorder="normal",
        ),
        bargap=0.25,
        bargroupgap=0.05,
        margin=dict(l=110, r=40, t=90, b=100),
    )
    return fig


# ── Export ────────────────────────────────────────────────────────────────────

def export_slide_figures(df: pd.DataFrame, output_dir: Path | None = None):
    """
    Export all three figures as both PNG (1920×height) and interactive HTML.
    Requires kaleido: pip install kaleido
    """
    import plotly.io as pio

    out = output_dir or ROOT / "outputs" / "slides"
    out.mkdir(parents=True, exist_ok=True)

    figures = {
        "01_composite_burden_map":   composite_burden_map(df, height=520),
        "02_cooccurrence_scatter":   cooccurrence_scatter(df, height=580),
        "03_burden_profile_bars":    burden_profile_bars(df, n=15, height=580),
    }

    for name, fig in figures.items():
        # Interactive HTML
        html_path = out / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"  [HTML] {html_path.name}")

        # Static PNG
        try:
            png_path = out / f"{name}.png"
            fig.write_image(str(png_path), width=1400, scale=2)
            print(f"  [PNG]  {png_path.name}")
        except Exception as e:
            print(f"  [PNG]  FAILED for {name}: {e}")

    print(f"\nSlide figures saved to: {out}")
    return figures


if __name__ == "__main__":
    snap = pd.read_csv(ROOT / "data" / "processed" / "commons_snapshot.csv")
    print("Exporting slide figures...")
    export_slide_figures(snap)
