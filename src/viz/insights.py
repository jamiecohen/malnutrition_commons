"""
Hypothesis-driven insights analysis for the Malnutrition Data Commons.

Tests six cross-indicator hypotheses and generates publication-ready figures
for the June 2026 IDM Learning Session.

Hypotheses tested:
  H1  Low vaccination coverage → higher measles burden
  H2  Lower ANC coverage → worse birth outcomes (LBW, preterm)
  H3  Malaria intensity amplifies child anaemia beyond iron deficiency alone
  H4  HIV–TB syndemic: HIV prevalence as a multiplier of TB incidence
  H5  Health-system reach composite vs. nutrition burden composite
  H6  LSFF intervention gap: high iron deficiency + low fortification coverage

Usage:
  python src/viz/insights.py            # saves figures to outputs/slides/insights/
  python src/viz/insights.py --show     # also opens figures interactively
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parents[2]
PROC    = ROOT / "data" / "processed"
OUT     = ROOT / "outputs" / "slides" / "insights"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
REGION_COLORS = {
    "Sub-Saharan Africa":        "#E63946",
    "South Asia":                "#F4A261",
    "East Asia & Pacific":       "#2A9D8F",
    "Latin America & Caribbean": "#457B9D",
    "Middle East & North Africa":"#A8DADC",
    "Europe & Central Asia":     "#A8A8A8",
    "North America":             "#CCCCCC",
    "":                          "#999999",
}

PRIORITY = ["IND", "PAK", "BGD", "NGA", "ETH", "COD"]
PRIORITY_STYLE = dict(
    marker_symbol="circle-open",
    marker_size=14,
    marker_line_width=2.5,
    marker_line_color="#1A1A2E",
)

FONT = dict(family="Arial, sans-serif", color="#1A1A2E")

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    snap = pd.read_csv(PROC / "commons_snapshot.csv")
    pop  = pd.read_csv(PROC / "population.csv")[["iso3", "population"]]
    df   = snap.merge(pop, on="iso3", how="left")
    # Measles per 100k population
    df["measles_per100k"] = (
        df["measles_reported_cases"] / df["population"] * 100_000
    )
    # MCV dropout rate (MCV1 → MCV2 gap): positive = losing kids between doses
    df["mcv_dropout_pct"] = df["mcv1_coverage_pct"] - df["mcv2_coverage_pct"]
    # Health-system reach composite (higher = better coverage)
    cov_cols = ["anc4_coverage_pct", "mcv1_coverage_pct", "dtp3_coverage_pct",
                "pcv3_coverage_pct", "rotac_coverage_pct"]
    df["health_system_score"] = _norm_composite(df, cov_cols, higher_is_better=True)
    # Nutrition-burden composite (higher = worse burden)
    burden_cols = ["anaemia_children_pct", "stunting_pct_who",
                   "iron_deficiency_pct", "low_birthweight_pct"]
    df["nutrition_burden_score"] = _norm_composite(df, burden_cols, higher_is_better=False)
    return df


def _norm_composite(df, cols, higher_is_better=True):
    """Min-max normalise each column, average across non-null, return 0-1 score.

    higher_is_better=True  (coverage): higher raw → higher score → better system
    higher_is_better=False (burden):   higher raw → higher score → worse burden
    In both cases the score direction is preserved: higher always means MORE of
    the underlying thing (more coverage OR more burden). Callers interpret accordingly.
    """
    parts = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].copy()
        rng = s.max() - s.min()
        if rng == 0:
            continue
        normed = (s - s.min()) / rng   # 0 = min value, 1 = max value
        parts.append(normed)
    if not parts:
        return np.nan
    return pd.concat(parts, axis=1).mean(axis=1)


def _region(df, iso):
    row = df[df["iso3"] == iso]
    return row["region"].iloc[0] if not row.empty else ""


def _corr_label(r, p, n):
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return f"r = {r:.2f} {stars}  (n={n})"


def _base_layout(title, xlab, ylab, **kwargs):
    return dict(
        title=dict(text=title, font=dict(size=16, **FONT), x=0.05),
        xaxis=dict(title=dict(text=xlab, font=FONT), tickfont=FONT, gridcolor="#EEEEEE"),
        yaxis=dict(title=dict(text=ylab, font=FONT), tickfont=FONT, gridcolor="#EEEEEE"),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        legend=dict(font=FONT, bgcolor="rgba(255,255,255,0.8)"),
        font=FONT,
        **kwargs,
    )


def _annotate_priority(fig, df, xcol, ycol, row=None, col=None):
    """Add labelled rings around the six Foundation priority countries."""
    kw = dict(row=row, col=col) if row else {}
    for iso in PRIORITY:
        sub = df[(df["iso3"] == iso) & df[xcol].notna() & df[ycol].notna()]
        if sub.empty:
            continue
        name = sub["country_name"].iloc[0]
        x, y = sub[xcol].iloc[0], sub[ycol].iloc[0]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(symbol="circle-open", size=16, line=dict(width=2.5, color="#1A1A2E"),
                        color="rgba(0,0,0,0)"),
            text=[name.split(",")[0]], textposition="top center",
            textfont=dict(size=10, color="#1A1A2E", family="Arial"),
            showlegend=False, hoverinfo="skip",
        ), **kw)


def _save(fig, name, show=False):
    html_path = OUT / f"{name}.html"
    png_path  = OUT / f"{name}.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=1400, height=900, scale=2)
        print(f"  Saved {name}.png + .html")
    except Exception as e:
        print(f"  Saved {name}.html  (PNG failed: {e})")
    if show:
        fig.show()


# ═══════════════════════════════════════════════════════════════════════════════
# H1  Vaccination coverage vs. measles burden
# ═══════════════════════════════════════════════════════════════════════════════

def h1_vaccination_measles(df, show=False):
    """
    H1: Countries with lower MCV1 coverage carry a disproportionately higher
    measles burden, compounded by high MCV1→MCV2 dropout.

    Two panels:
      Left  — MCV1 coverage (%) vs measles cases per 100k (log scale)
      Right — MCV dropout (MCV1-MCV2 gap) vs measles cases per 100k
    """
    sub = df[df["measles_per100k"].notna() & df["mcv1_coverage_pct"].notna()
             & (df["measles_per100k"] > 0)].copy()
    sub["log_measles"] = np.log10(sub["measles_per100k"])

    r1, p1, _ = stats.spearmanr(sub["mcv1_coverage_pct"], sub["log_measles"],
                                 nan_policy="omit")[:3] + (len(sub),)
    r1, p1 = stats.spearmanr(sub["mcv1_coverage_pct"], sub["log_measles"],
                              nan_policy="omit")
    n1 = len(sub)

    sub2 = df[df["measles_per100k"].notna() & df["mcv_dropout_pct"].notna()
              & (df["measles_per100k"] > 0)].copy()
    sub2["log_measles"] = np.log10(sub2["measles_per100k"])
    r2, p2 = stats.spearmanr(sub2["mcv_dropout_pct"], sub2["log_measles"],
                              nan_policy="omit")
    n2 = len(sub2)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"MCV1 coverage vs. measles cases per 100k<br>"
                            f"<sup>Spearman {_corr_label(r1, p1, n1)}</sup>",
                            f"MCV1→MCV2 dropout vs. measles cases per 100k<br>"
                            f"<sup>Spearman {_corr_label(r2, p2, n2)}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        mask1 = sub["region"] == region
        if mask1.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub.loc[mask1, "mcv1_coverage_pct"],
            y=sub.loc[mask1, "log_measles"],
            mode="markers",
            name=region or "Other",
            marker=dict(color=color, size=7, opacity=0.75),
            customdata=sub.loc[mask1, ["country_name", "measles_per100k",
                                        "mcv1_coverage_pct"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "MCV1: %{customdata[2]:.0f}%<br>"
                          "Measles/100k: %{customdata[1]:.1f}<extra></extra>",
            legendgroup=region,
        ), row=1, col=1)

        mask2 = sub2["region"] == region
        if mask2.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub2.loc[mask2, "mcv_dropout_pct"],
            y=sub2.loc[mask2, "log_measles"],
            mode="markers",
            name=region or "Other",
            marker=dict(color=color, size=7, opacity=0.75),
            customdata=sub2.loc[mask2, ["country_name", "measles_per100k",
                                         "mcv_dropout_pct"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "MCV dropout: %{customdata[2]:.0f}pp<br>"
                          "Measles/100k: %{customdata[1]:.1f}<extra></extra>",
            showlegend=False,
            legendgroup=region,
        ), row=1, col=2)

    _annotate_priority(fig, sub, "mcv1_coverage_pct", "log_measles", row=1, col=1)
    _annotate_priority(fig, sub2, "mcv_dropout_pct", "log_measles", row=1, col=2)

    # Trend lines
    for panel_sub, xcol, rc, pc in [
        (sub,  "mcv1_coverage_pct", r1, p1),
        (sub2, "mcv_dropout_pct",   r2, p2),
    ]:
        col_n = 1 if xcol == "mcv1_coverage_pct" else 2
        xs = np.linspace(panel_sub[xcol].min(), panel_sub[xcol].max(), 100)
        m, b, *_ = stats.linregress(panel_sub[xcol].dropna(),
                                     panel_sub.loc[panel_sub[xcol].notna(), "log_measles"])
        fig.add_trace(go.Scatter(
            x=xs, y=m * xs + b, mode="lines",
            line=dict(color="#888", dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_n)

    yticks = [0.01, 0.1, 1, 10, 100, 1000]
    ytick_vals = [np.log10(v) for v in yticks]

    fig.update_yaxes(tickvals=ytick_vals, ticktext=[str(v) for v in yticks],
                     title_text="Measles cases per 100k pop. (log scale)")
    fig.update_xaxes(title_text="MCV1 coverage (%)", row=1, col=1)
    fig.update_xaxes(title_text="MCV1 → MCV2 dropout (percentage points)", row=1, col=2)
    fig.update_layout(
        title=dict(text="H1 — Vaccination gaps predict measles burden",
                   font=dict(size=18, **FONT), x=0.05),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        font=FONT, height=550, legend=dict(font=FONT),
    )

    print(f"\nH1 Findings:")
    print(f"  MCV1 vs log(measles/100k):  r={r1:.3f}, p={p1:.3e}, n={n1}")
    print(f"  MCV dropout vs log(measles): r={r2:.3f}, p={p2:.3e}, n={n2}")
    top_burden = sub.nlargest(5, "measles_per100k")[["country_name","mcv1_coverage_pct","measles_per100k"]]
    print(f"  Top 5 measles burden countries:\n{top_burden.to_string(index=False)}")

    _save(fig, "h1_vaccination_measles", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H2  ANC coverage → birth outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def h2_anc_birth_outcomes(df, show=False):
    """
    H2: Higher ANC4 coverage is associated with lower low-birthweight prevalence
    and lower preterm birth rates. The mechanism runs through iron/folate
    supplementation, risk detection, and maternal nutrition support.
    """
    sub = df[df["anc4_coverage_pct"].notna() &
             (df["low_birthweight_pct"].notna() | df["preterm_birth_rate_pct"].notna())].copy()

    sub_lbw = sub.dropna(subset=["low_birthweight_pct"])
    sub_pre = sub.dropna(subset=["preterm_birth_rate_pct"])

    r_lbw, p_lbw = stats.spearmanr(sub_lbw["anc4_coverage_pct"],
                                     sub_lbw["low_birthweight_pct"])
    r_pre, p_pre = stats.spearmanr(sub_pre["anc4_coverage_pct"],
                                     sub_pre["preterm_birth_rate_pct"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"ANC4 coverage vs. low birthweight prevalence<br>"
                            f"<sup>Spearman {_corr_label(r_lbw, p_lbw, len(sub_lbw))}</sup>",
                            f"ANC4 coverage vs. preterm birth rate<br>"
                            f"<sup>Spearman {_corr_label(r_pre, p_pre, len(sub_pre))}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        for col_n, ss, ycol in [(1, sub_lbw, "low_birthweight_pct"),
                                 (2, sub_pre, "preterm_birth_rate_pct")]:
            mask = ss["region"] == region
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=ss.loc[mask, "anc4_coverage_pct"],
                y=ss.loc[mask, ycol],
                mode="markers",
                name=region or "Other",
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=ss.loc[mask, ["country_name", "anc4_coverage_pct", ycol]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "ANC4: %{customdata[1]:.0f}%<br>"
                              f"{ycol}: %{{customdata[2]:.1f}}%<extra></extra>",
                showlegend=(col_n == 1),
                legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub_lbw, "anc4_coverage_pct", "low_birthweight_pct", row=1, col=1)
    _annotate_priority(fig, sub_pre, "anc4_coverage_pct", "preterm_birth_rate_pct", row=1, col=2)

    for col_n, ss, ycol in [(1, sub_lbw, "low_birthweight_pct"),
                             (2, sub_pre, "preterm_birth_rate_pct")]:
        valid = ss[["anc4_coverage_pct", ycol]].dropna()
        if len(valid) > 5:
            m, b, *_ = stats.linregress(valid["anc4_coverage_pct"], valid[ycol])
            xs = np.linspace(valid["anc4_coverage_pct"].min(),
                             valid["anc4_coverage_pct"].max(), 100)
            fig.add_trace(go.Scatter(
                x=xs, y=m * xs + b, mode="lines",
                line=dict(color="#888", dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col_n)

    fig.update_xaxes(title_text="ANC 4+ visits coverage (%)")
    fig.update_yaxes(title_text="Low birthweight prevalence (%)", row=1, col=1)
    fig.update_yaxes(title_text="Preterm birth rate (%)", row=1, col=2)
    fig.update_layout(
        title=dict(text="H2 — Antenatal care coverage predicts better birth outcomes",
                   font=dict(size=18, **FONT), x=0.05),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        font=FONT, height=550, legend=dict(font=FONT),
    )

    print(f"\nH2 Findings:")
    print(f"  ANC4 vs LBW:    r={r_lbw:.3f}, p={p_lbw:.3e}, n={len(sub_lbw)}")
    print(f"  ANC4 vs preterm: r={r_pre:.3f}, p={p_pre:.3e}, n={len(sub_pre)}")
    # South Asia highlight
    sa = sub_lbw[sub_lbw["region"] == "South Asia"][
        ["country_name", "anc4_coverage_pct", "low_birthweight_pct"]].dropna()
    if not sa.empty:
        print(f"  South Asia outliers (high LBW despite moderate ANC):\n{sa.to_string(index=False)}")

    _save(fig, "h2_anc_birth_outcomes", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H3  Malaria amplifies child anaemia beyond iron deficiency
# ═══════════════════════════════════════════════════════════════════════════════

def h3_malaria_anaemia(df, show=False):
    """
    H3: Malaria destroys red blood cells independent of nutritional iron status.
    In high-malaria settings, child anaemia exceeds what iron deficiency alone
    would predict — illustrating why nutrition interventions alone are insufficient.

    Approach: show the 'anaemia residual' (observed anaemia minus iron-deficiency-
    predicted anaemia) as a function of malaria incidence.
    """
    sub = df[df["anaemia_children_pct"].notna() &
             df["iron_deficiency_pct"].notna() &
             df["malaria_incidence_per1000"].notna()].copy()

    # Fit simple linear model: anaemia ~ iron_deficiency
    m, b, r_fe, p_fe, _ = stats.linregress(
        sub["iron_deficiency_pct"], sub["anaemia_children_pct"]
    )
    sub["predicted_anaemia"] = m * sub["iron_deficiency_pct"] + b
    sub["anaemia_residual"]  = sub["anaemia_children_pct"] - sub["predicted_anaemia"]

    r_res, p_res = stats.spearmanr(sub["malaria_incidence_per1000"],
                                    sub["anaemia_residual"])

    # Also direct malaria → anaemia
    r_dir, p_dir = stats.spearmanr(sub["malaria_incidence_per1000"],
                                    sub["anaemia_children_pct"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Malaria incidence vs. child anaemia<br>"
                            f"<sup>Spearman {_corr_label(r_dir, p_dir, len(sub))}</sup>",
                            f"Malaria incidence vs. anaemia beyond iron deficiency<br>"
                            f"<sup>Spearman {_corr_label(r_res, p_res, len(sub))}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        mask = sub["region"] == region
        if mask.sum() == 0:
            continue
        for col_n, ycol, ytitle in [
            (1, "anaemia_children_pct", "Child anaemia prevalence (%)"),
            (2, "anaemia_residual",     "Anaemia residual after iron deficiency (pp)"),
        ]:
            fig.add_trace(go.Scatter(
                x=sub.loc[mask, "malaria_incidence_per1000"],
                y=sub.loc[mask, ycol],
                mode="markers",
                name=region or "Other",
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=sub.loc[mask, ["country_name",
                                           "malaria_incidence_per1000",
                                           "anaemia_children_pct",
                                           "iron_deficiency_pct"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "Malaria: %{customdata[1]:.0f}/1k<br>"
                              "Anaemia: %{customdata[2]:.0f}%<br>"
                              "Iron def: %{customdata[3]:.0f}%<extra></extra>",
                showlegend=(col_n == 1),
                legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub, "malaria_incidence_per1000", "anaemia_children_pct", row=1, col=1)
    _annotate_priority(fig, sub, "malaria_incidence_per1000", "anaemia_residual", row=1, col=2)

    # Zero line on residual panel
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=1, col=2)
    fig.add_annotation(x=sub["malaria_incidence_per1000"].max() * 0.85,
                       y=2, text="Above line = malaria driving extra anaemia",
                       font=dict(size=10, color="#555"), showarrow=False, row=1, col=2)

    for col_n, ycol in [(1, "anaemia_children_pct"), (2, "anaemia_residual")]:
        valid = sub[["malaria_incidence_per1000", ycol]].dropna()
        if len(valid) > 5:
            m2, b2, *_ = stats.linregress(valid["malaria_incidence_per1000"], valid[ycol])
            xs = np.linspace(0, valid["malaria_incidence_per1000"].max(), 100)
            fig.add_trace(go.Scatter(
                x=xs, y=m2 * xs + b2, mode="lines",
                line=dict(color="#888", dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col_n)

    fig.update_xaxes(title_text="Malaria incidence (per 1,000 at risk)")
    fig.update_yaxes(title_text="Child anaemia prevalence (%)", row=1, col=1)
    fig.update_yaxes(title_text="Anaemia beyond iron deficiency (percentage points)", row=1, col=2)
    fig.update_layout(
        title=dict(text="H3 — Malaria amplifies child anaemia beyond nutritional iron deficiency",
                   font=dict(size=18, **FONT), x=0.05),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        font=FONT, height=550, legend=dict(font=FONT),
    )

    print(f"\nH3 Findings:")
    print(f"  Iron def (linear model) → anaemia: slope={m:.2f}, r²={r_fe**2:.2f}")
    print(f"  Malaria vs anaemia (direct): r={r_dir:.3f}, p={p_dir:.3e}")
    print(f"  Malaria vs anaemia RESIDUAL: r={r_res:.3f}, p={p_res:.3e}, n={len(sub)}")
    top_res = sub.nlargest(5, "anaemia_residual")[["country_name","malaria_incidence_per1000",
                                                    "anaemia_children_pct","anaemia_residual"]]
    print(f"  Countries with most malaria-driven excess anaemia:\n{top_res.to_string(index=False)}")

    _save(fig, "h3_malaria_anaemia", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H4  HIV–TB syndemic
# ═══════════════════════════════════════════════════════════════════════════════

def h4_hiv_tb(df, show=False):
    """
    H4: HIV prevalence is a strong multiplier of TB incidence. Countries with
    high HIV burden carry disproportionate TB incidence, reflecting immune
    compromise. This also illustrates the triple-burden overlap: HIV/TB
    countries tend to have high malnutrition.
    """
    sub = df[df["hiv_prevalence_pct"].notna() &
             df["tb_incidence_per100k"].notna()].copy()
    sub["log_tb"] = np.log10(sub["tb_incidence_per100k"].clip(lower=0.1))

    r, p = stats.spearmanr(sub["hiv_prevalence_pct"], sub["log_tb"])

    # Bubble chart: size = stunting prevalence
    sub["bubble_size"] = sub["stunting_pct_who"].fillna(
        sub["stunting_pct_who"].median()
    ).clip(lower=5)

    fig = go.Figure()

    for region, color in REGION_COLORS.items():
        mask = sub["region"] == region
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub.loc[mask, "hiv_prevalence_pct"],
            y=sub.loc[mask, "log_tb"],
            mode="markers",
            name=region or "Other",
            marker=dict(
                color=color,
                size=sub.loc[mask, "bubble_size"] ** 0.6,
                opacity=0.75,
                line=dict(width=0.5, color="white"),
            ),
            customdata=sub.loc[mask, ["country_name", "hiv_prevalence_pct",
                                       "tb_incidence_per100k",
                                       "stunting_pct_who"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "HIV: %{customdata[1]:.1f}%<br>"
                          "TB incidence: %{customdata[2]:.0f}/100k<br>"
                          "Stunting: %{customdata[3]:.0f}%<extra></extra>",
        ))

    _annotate_priority(fig, sub, "hiv_prevalence_pct", "log_tb")

    # Trend line
    m, b, *_ = stats.linregress(sub["hiv_prevalence_pct"], sub["log_tb"])
    xs = np.linspace(0, sub["hiv_prevalence_pct"].max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=m * xs + b, mode="lines",
        line=dict(color="#888", dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))

    yticks = [1, 5, 10, 50, 100, 300, 700]
    layout = _base_layout(
        f"H4 — HIV–TB syndemic  (Spearman {_corr_label(r, p, len(sub))})<br>"
        f"<sup>Bubble size = stunting prevalence — illustrates nutrition–infectious disease co-occurrence</sup>",
        "HIV prevalence, adults 15–49 (%)",
        "TB incidence (per 100k pop., log scale)",
    )
    layout["yaxis"].update(
        tickvals=[np.log10(v) for v in yticks],
        ticktext=[str(v) for v in yticks],
        title=dict(text="TB incidence (per 100k, log scale)", font=FONT),
    )
    fig.update_layout(**layout, height=600)

    print(f"\nH4 Findings:")
    print(f"  HIV vs log(TB incidence): r={r:.3f}, p={p:.3e}, n={len(sub)}")
    top = sub.nlargest(8, "hiv_prevalence_pct")[
        ["country_name", "hiv_prevalence_pct", "tb_incidence_per100k", "stunting_pct_who"]
    ]
    print(f"  Highest HIV burden countries:\n{top.to_string(index=False)}")

    _save(fig, "h4_hiv_tb_syndemic", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H5  Health-system reach vs. nutrition burden
# ═══════════════════════════════════════════════════════════════════════════════

def h5_system_vs_burden(df, show=False):
    """
    H5: Countries with weaker health system reach (composite of ANC4, MCV1,
    DTP3, PCV3, RotaC) carry a higher composite nutrition burden (stunting,
    anaemia, iron deficiency, LBW). The quadrant structure reveals four
    country archetypes and where Foundation investment is most needed.
    """
    sub = df[df["health_system_score"].notna() &
             df["nutrition_burden_score"].notna()].copy()

    r, p = stats.spearmanr(sub["health_system_score"],
                            sub["nutrition_burden_score"])

    # Quadrant medians
    x_mid = sub["health_system_score"].median()
    y_mid = sub["nutrition_burden_score"].median()

    fig = go.Figure()

    for region, color in REGION_COLORS.items():
        mask = sub["region"] == region
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub.loc[mask, "health_system_score"],
            y=sub.loc[mask, "nutrition_burden_score"],
            mode="markers",
            name=region or "Other",
            marker=dict(color=color, size=8, opacity=0.75),
            customdata=sub.loc[mask, ["country_name", "health_system_score",
                                       "nutrition_burden_score",
                                       "anaemia_children_pct",
                                       "mcv1_coverage_pct"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Health system: %{customdata[1]:.2f}<br>"
                          "Nutrition burden: %{customdata[2]:.2f}<br>"
                          "Anaemia: %{customdata[3]:.0f}%<br>"
                          "MCV1: %{customdata[4]:.0f}%<extra></extra>",
        ))

    _annotate_priority(fig, sub, "health_system_score", "nutrition_burden_score")

    # Quadrant dividers
    fig.add_vline(x=x_mid, line_dash="dot", line_color="#AAAAAA", line_width=1)
    fig.add_hline(y=y_mid, line_dash="dot", line_color="#AAAAAA", line_width=1)

    # Quadrant labels
    # x-axis = health_system_score (higher = stronger coverage)
    # y-axis = nutrition_burden_score (higher = worse burden)
    # Crisis = low coverage + high burden = left side, top half
    pad = 0.03
    for (qx, qy, label, color) in [
        (x_mid - pad, y_mid + pad, "High burden<br>Low coverage<br><b>Crisis</b>",       "#CC3333"),
        (x_mid + pad, y_mid + pad, "High burden<br>Strong coverage<br><b>Progressing</b>","#F0A500"),
        (x_mid - pad, y_mid - pad, "Low burden<br>Low coverage<br><b>Fragile gains</b>", "#888888"),
        (x_mid + pad, y_mid - pad, "Low burden<br>Strong coverage<br><b>Best position</b>","#1A8754"),
    ]:
        xanchor = "right" if qx < x_mid else "left"
        yanchor = "bottom" if qy > y_mid else "top"
        fig.add_annotation(
            x=qx, y=qy, text=label,
            font=dict(size=10, color=color), showarrow=False,
            xanchor=xanchor, yanchor=yanchor,
            bgcolor="rgba(255,255,255,0.7)", borderpad=4,
        )

    fig.update_layout(
        **_base_layout(
            f"H5 — Health-system reach vs. nutrition burden composite<br>"
            f"<sup>Spearman {_corr_label(r, p, len(sub))} | "
            f"Health score = mean normalised (ANC4, MCV1, DTP3, PCV3, RotaC) | "
            f"Burden score = mean normalised (stunting, anaemia, iron def., LBW)</sup>",
            "Health-system coverage score (higher = stronger)",
            "Nutrition burden score (higher = worse burden)",
        ),
        height=620,
    )

    print(f"\nH5 Findings:")
    print(f"  Health system vs burden: r={r:.3f}, p={p:.3e}, n={len(sub)}")
    # Crisis quadrant
    crisis = sub[(sub["health_system_score"] < x_mid) &
                 (sub["nutrition_burden_score"] > y_mid)].sort_values(
                     "nutrition_burden_score", ascending=False
                 ).head(10)
    print(f"  'Crisis' quadrant top 10:\n{crisis[['country_name','health_system_score','nutrition_burden_score']].to_string(index=False)}")

    _save(fig, "h5_system_vs_burden", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H6  LSFF intervention gap (updated with GBD iron deficiency)
# ═══════════════════════════════════════════════════════════════════════════════

def h6_lsff_gap(df, show=False):
    """
    H6: Countries with high iron deficiency prevalence and low LSFF coverage
    represent the clearest intervention opportunity for large-scale food
    fortification. The lower-right quadrant of the scatter is the priority zone.

    Now using GBD 2023 iron deficiency estimates (vs. anaemia proxy previously).
    """
    sub = df[df["iron_deficiency_pct"].notna() &
             df["lsff_coverage_proxy_pct"].notna()].copy()

    r, p = stats.spearmanr(sub["lsff_coverage_proxy_pct"],
                            sub["iron_deficiency_pct"])

    # LSFF status colour
    lsff_color_map = {
        "mandatory": "#1A8754",
        "voluntary": "#F0A500",
        "no_program": "#CC3333",
    }

    def _lsff_status(row):
        if row.get("lsff_any_mandatory"):
            return "mandatory"
        if row.get("lsff_any_program"):
            return "voluntary"
        return "no_program"

    sub["lsff_status"] = sub.apply(_lsff_status, axis=1)

    fig = go.Figure()

    for status, color in lsff_color_map.items():
        mask = sub["lsff_status"] == status
        if mask.sum() == 0:
            continue
        label = {"mandatory": "Mandatory fortification",
                 "voluntary": "Voluntary fortification",
                 "no_program": "No programme"}[status]
        fig.add_trace(go.Scatter(
            x=sub.loc[mask, "lsff_coverage_proxy_pct"],
            y=sub.loc[mask, "iron_deficiency_pct"],
            mode="markers",
            name=label,
            marker=dict(color=color, size=8, opacity=0.8,
                        line=dict(width=0.5, color="white")),
            customdata=sub.loc[mask, ["country_name", "iron_deficiency_pct",
                                       "lsff_coverage_proxy_pct",
                                       "anaemia_children_pct"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Iron deficiency: %{customdata[1]:.1f}%<br>"
                          "LSFF coverage: %{customdata[2]:.0f}%<br>"
                          "Child anaemia: %{customdata[3]:.0f}%<extra></extra>",
        ))

    _annotate_priority(fig, sub, "lsff_coverage_proxy_pct", "iron_deficiency_pct")

    # Intervention gap zone
    x_thresh = sub["lsff_coverage_proxy_pct"].quantile(0.4)
    y_thresh = sub["iron_deficiency_pct"].quantile(0.6)
    fig.add_shape(type="rect",
                  x0=0, y0=y_thresh,
                  x1=x_thresh, y1=sub["iron_deficiency_pct"].max() * 1.05,
                  fillcolor="rgba(204, 51, 51, 0.06)",
                  line=dict(color="#CC3333", dash="dot", width=1.5))
    fig.add_annotation(
        x=x_thresh / 2, y=sub["iron_deficiency_pct"].max() * 0.98,
        text="<b>Intervention gap zone</b><br>High iron deficiency, low LSFF",
        font=dict(size=11, color="#CC3333"), showarrow=False,
        bgcolor="rgba(255,255,255,0.8)", borderpad=4,
    )

    h6_layout = _base_layout(
        f"H6 — LSFF intervention gap: iron deficiency vs. fortification coverage<br>"
        f"<sup>Spearman {_corr_label(r, p, len(sub))} | "
        f"Iron deficiency: GBD 2023 estimates | LSFF: FFI 2023 mandatory/voluntary status</sup>",
        "LSFF coverage proxy (%) — 75% mandatory, 20% voluntary, 0% no programme",
        "Iron deficiency prevalence, GBD 2023 (%)",
    )
    h6_layout["legend"] = dict(font=FONT, title=dict(text="Fortification status"))
    fig.update_layout(**h6_layout, height=600)

    print(f"\nH6 Findings:")
    print(f"  LSFF coverage vs iron deficiency: r={r:.3f}, p={p:.3e}, n={len(sub)}")
    gap_countries = sub[
        (sub["lsff_coverage_proxy_pct"] <= x_thresh) &
        (sub["iron_deficiency_pct"] >= y_thresh)
    ].sort_values("iron_deficiency_pct", ascending=False)
    print(f"  Countries in intervention gap zone ({len(gap_countries)}):")
    print(gap_countries[["country_name","iron_deficiency_pct",
                           "lsff_coverage_proxy_pct","region"]].head(15).to_string(index=False))

    _save(fig, "h6_lsff_intervention_gap", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus: Country burden heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def burden_heatmap(df, show=False, n_countries=40):
    """
    Heatmap of all key indicators for the top-N highest-burden countries.
    Columns are clustered by domain; rows sorted by composite burden score.
    Immediately reveals which countries have co-occurring burdens across domains.
    """
    indicators = {
        # Label: column
        "Child anaemia":          "anaemia_children_pct",
        "Pregnant anaemia":       "anaemia_pregnant_women_pct",
        "Iron deficiency":        "iron_deficiency_pct",
        "Vitamin A deficiency":   "vitamin_a_deficiency_pct",
        "Zinc deficiency":        "zinc_deficiency_pct",
        "Stunting":               "stunting_pct_who",
        "Wasting":                "wasting_pct",
        "Low birthweight":        "low_birthweight_pct",
        "Preterm birth":          "preterm_birth_rate_pct",
        "TB incidence":           "tb_incidence_per100k",
        "HIV prevalence":         "hiv_prevalence_pct",
        "Malaria incidence":      "malaria_incidence_per1000",
        "Measles/100k":           "measles_per100k",
        "MCV1 coverage":          "mcv1_coverage_pct",
        "DTP3 coverage":          "dtp3_coverage_pct",
        "ANC4 coverage":          "anc4_coverage_pct",
        "LSFF coverage":          "lsff_coverage_proxy_pct",
    }

    avail = {label: col for label, col in indicators.items() if col in df.columns}
    sub = df.dropna(subset=["nutrition_burden_score"]).nlargest(n_countries, "nutrition_burden_score")

    # Normalize each column 0→1 for display (coverage columns flipped so red=bad)
    coverage_cols = {"MCV1 coverage", "DTP3 coverage", "ANC4 coverage", "LSFF coverage"}
    matrix = pd.DataFrame(index=sub["country_name"])
    for label, col in avail.items():
        s = sub[col].values.astype(float)
        rng = np.nanmax(s) - np.nanmin(s)
        if rng == 0:
            normed = np.zeros_like(s)
        else:
            normed = (s - np.nanmin(s)) / rng
            if label in coverage_cols:
                normed = 1 - normed   # flip: high coverage = green
        matrix[label] = normed

    # Colour: RdYlGn reversed for burden (red=high burden), but consistent direction
    # We'll use a diverging scale where 1.0 = worst outcome (red)
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        colorscale=[
            [0.0, "#1A8754"],   # green = good
            [0.5, "#F0A500"],   # amber
            [1.0, "#CC3333"],   # red = bad
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Normalised score<br>(red = worst outcome)",
                       font=dict(size=11)),
            tickvals=[0, 0.5, 1],
            ticktext=["Best", "Mid", "Worst"],
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
    ))

    # Highlight priority countries
    priority_names = sub[sub["iso3"].isin(PRIORITY)]["country_name"].tolist()
    for name in priority_names:
        if name in matrix.index:
            idx = list(matrix.index).index(name)
            fig.add_shape(type="rect",
                          x0=-0.5, x1=len(matrix.columns) - 0.5,
                          y0=idx - 0.5, y1=idx + 0.5,
                          line=dict(color="#1A1A2E", width=2))

    fig.update_layout(
        title=dict(
            text=f"Multi-domain burden heatmap — top {n_countries} countries by nutrition burden score<br>"
                 f"<sup>Outlined rows = Foundation priority countries | "
                 f"Red = high burden / low coverage; Green = low burden / high coverage</sup>",
            font=dict(size=16, **FONT), x=0.02,
        ),
        xaxis=dict(tickfont=dict(size=11), side="top"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        height=1000,
        width=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        margin=dict(l=160, r=80, t=120, b=40),
    )

    print(f"\nBurden Heatmap: top {n_countries} countries, {len(avail)} indicators")

    _save(fig, "burden_heatmap", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H7  Vitamin A deficiency amplifies measles burden (controlling for MCV1)
# ═══════════════════════════════════════════════════════════════════════════════

def h7_vitamin_a_measles(df, show=False):
    """
    H7: Countries with higher vitamin A deficiency carry a higher measles
    burden even after accounting for MCV1 vaccination coverage.
    Mechanism: vitamin A is essential for mucosal immunity and is used as
    adjunct treatment for measles; deficiency worsens outcomes.
    WHO/Cochrane evidence: vitamin A supplementation reduces measles
    mortality by >50%.

    Approach: partial correlation — residualise both VitA and log(measles)
    on MCV1 coverage, then correlate the residuals.
    """
    sub = df[
        df["vitamin_a_deficiency_pct"].notna() &
        df["measles_per100k"].notna() &
        df["mcv1_coverage_pct"].notna() &
        (df["measles_per100k"] > 0)
    ].copy()
    sub["log_measles"] = np.log10(sub["measles_per100k"])

    # Direct correlation
    r_direct, p_direct = stats.spearmanr(
        sub["vitamin_a_deficiency_pct"], sub["log_measles"]
    )

    # Partial correlation: residualise on MCV1
    def _resid(y_col, x_col, data):
        valid = data[[y_col, x_col]].dropna()
        m, b, *_ = stats.linregress(valid[x_col], valid[y_col])
        return data[y_col] - (m * data[x_col] + b)

    sub["vita_resid"]    = _resid("vitamin_a_deficiency_pct", "mcv1_coverage_pct", sub)
    sub["measles_resid"] = _resid("log_measles", "mcv1_coverage_pct", sub)
    r_partial, p_partial = stats.spearmanr(
        sub["vita_resid"], sub["measles_resid"], nan_policy="omit"
    )

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Vitamin A deficiency vs. measles/100k<br>"
                            f"<sup>Direct Spearman {_corr_label(r_direct, p_direct, len(sub))}</sup>",
                            f"Vitamin A vs. measles (both residualised on MCV1)<br>"
                            f"<sup>Partial Spearman {_corr_label(r_partial, p_partial, len(sub))}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        mask = sub["region"] == region
        if mask.sum() == 0:
            continue
        for col_n, xcol, ycol in [
            (1, "vitamin_a_deficiency_pct", "log_measles"),
            (2, "vita_resid",               "measles_resid"),
        ]:
            fig.add_trace(go.Scatter(
                x=sub.loc[mask, xcol], y=sub.loc[mask, ycol],
                mode="markers", name=region or "Other",
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=sub.loc[mask, ["country_name", "vitamin_a_deficiency_pct",
                                           "measles_per100k", "mcv1_coverage_pct"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "VitA def: %{customdata[1]:.1f}%<br>"
                              "Measles/100k: %{customdata[2]:.1f}<br>"
                              "MCV1: %{customdata[3]:.0f}%<extra></extra>",
                showlegend=(col_n == 1), legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub, "vitamin_a_deficiency_pct", "log_measles", row=1, col=1)
    _annotate_priority(fig, sub, "vita_resid", "measles_resid", row=1, col=2)

    yticks = [0.01, 0.1, 1, 10, 100, 1000]
    fig.update_yaxes(
        tickvals=[np.log10(v) for v in yticks],
        ticktext=[str(v) for v in yticks],
        title_text="Measles cases per 100k (log scale)", row=1, col=1,
    )
    fig.update_yaxes(title_text="Measles residual (after MCV1)", row=1, col=2)
    fig.update_xaxes(title_text="Vitamin A deficiency (%)", row=1, col=1)
    fig.update_xaxes(title_text="Vitamin A deficiency residual (after MCV1)", row=1, col=2)
    fig.update_layout(
        title=dict(text="H7 — Vitamin A deficiency amplifies measles burden beyond vaccination gaps",
                   font=dict(size=18, **FONT), x=0.05),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white", font=FONT, height=550,
    )

    print(f"\nH7 Findings:")
    print(f"  VitA vs log(measles/100k) direct:  r={r_direct:.3f}, p={p_direct:.3e}, n={len(sub)}")
    print(f"  VitA vs measles partial (|MCV1): r={r_partial:.3f}, p={p_partial:.3e}")

    _save(fig, "h7_vitamin_a_measles", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H8  Maternal anaemia → maternal mortality
# ═══════════════════════════════════════════════════════════════════════════════

def h8_maternal_anaemia_mortality(df, show=False):
    """
    H8: Maternal anaemia — particularly in pregnant women — is a major
    pathway to maternal mortality through postpartum haemorrhage and
    reduced surgical resilience. The one-pager cites Daru et al. (2018):
    anaemic women face significantly elevated maternal mortality risk.

    We test: anaemia_pregnant_women_pct → log(maternal_mortality_per100k)
    and compare SSA vs. South Asia as regional archetypes.
    """
    sub = df[
        df["anaemia_pregnant_women_pct"].notna() &
        df["maternal_mortality_per100k"].notna() &
        (df["maternal_mortality_per100k"] > 0)
    ].copy()
    sub["log_mmr"] = np.log10(sub["maternal_mortality_per100k"])

    r, p = stats.spearmanr(sub["anaemia_pregnant_women_pct"], sub["log_mmr"])

    # Also ANC4 → MMR (the care-access pathway)
    sub2 = df[
        df["anc4_coverage_pct"].notna() &
        df["maternal_mortality_per100k"].notna() &
        (df["maternal_mortality_per100k"] > 0)
    ].copy()
    sub2["log_mmr"] = np.log10(sub2["maternal_mortality_per100k"])
    r2, p2 = stats.spearmanr(sub2["anc4_coverage_pct"], sub2["log_mmr"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Maternal anaemia → maternal mortality<br>"
                            f"<sup>Spearman {_corr_label(r, p, len(sub))}</sup>",
                            f"ANC4 coverage → maternal mortality<br>"
                            f"<sup>Spearman {_corr_label(r2, p2, len(sub2))}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        for col_n, ss, xcol in [(1, sub, "anaemia_pregnant_women_pct"),
                                 (2, sub2, "anc4_coverage_pct")]:
            mask = ss["region"] == region
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=ss.loc[mask, xcol], y=ss.loc[mask, "log_mmr"],
                mode="markers", name=region or "Other",
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=ss.loc[mask, ["country_name", xcol,
                                          "maternal_mortality_per100k"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              f"{xcol}: %{{customdata[1]:.1f}}<br>"
                              "MMR: %{customdata[2]:.0f}/100k<extra></extra>",
                showlegend=(col_n == 1), legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub,  "anaemia_pregnant_women_pct", "log_mmr", row=1, col=1)
    _annotate_priority(fig, sub2, "anc4_coverage_pct",          "log_mmr", row=1, col=2)

    mmr_ticks = [10, 50, 100, 200, 500, 1000, 2000]
    for col_n in [1, 2]:
        fig.update_yaxes(
            tickvals=[np.log10(v) for v in mmr_ticks],
            ticktext=[str(v) for v in mmr_ticks],
            title_text="Maternal mortality ratio (per 100k, log scale)",
            row=1, col=col_n,
        )
    fig.update_xaxes(title_text="Anaemia prevalence, pregnant women (%)", row=1, col=1)
    fig.update_xaxes(title_text="ANC 4+ visits coverage (%)", row=1, col=2)
    fig.update_layout(
        title=dict(text="H8 — Maternal anaemia and ANC gaps predict maternal mortality",
                   font=dict(size=18, **FONT), x=0.05),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white", font=FONT, height=550,
    )

    print(f"\nH8 Findings:")
    print(f"  Pregnant anaemia vs log(MMR): r={r:.3f}, p={p:.3e}, n={len(sub)}")
    print(f"  ANC4 vs log(MMR):            r={r2:.3f}, p={p2:.3e}, n={len(sub2)}")
    top_mmr = sub.nlargest(8, "maternal_mortality_per100k")[
        ["country_name", "anaemia_pregnant_women_pct", "maternal_mortality_per100k", "region"]
    ]
    print(f"  Highest MMR countries:\n{top_mmr.to_string(index=False)}")

    _save(fig, "h8_maternal_anaemia_mortality", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H9  Stunting + wasting → under-5 mortality (the 45% claim)
# ═══════════════════════════════════════════════════════════════════════════════

def h9_undernutrition_child_mortality(df, show=False):
    """
    H9: Undernutrition — stunting and wasting — is the largest single
    risk factor for child mortality globally, estimated to contribute to
    ~45% of under-5 deaths (Black et al. 2013 Lancet). We test whether
    this signal is visible at country level cross-sectionally.

    Also test the composite: stunting + wasting + anaemia + LBW vs U5MR,
    using the nutrition_burden_score.
    """
    sub = df[
        df["stunting_pct_who"].notna() &
        df["u5_mortality_per1000"].notna()
    ].copy()
    sub["log_u5mr"] = np.log10(sub["u5_mortality_per1000"].clip(lower=0.5))

    sub2 = df[df["nutrition_burden_score"].notna() & df["u5_mortality_per1000"].notna()].copy()
    sub2["log_u5mr"] = np.log10(sub2["u5_mortality_per1000"].clip(lower=0.5))

    r_st,  p_st  = stats.spearmanr(sub["stunting_pct_who"],       sub["log_u5mr"])
    r_nb,  p_nb  = stats.spearmanr(sub2["nutrition_burden_score"], sub2["log_u5mr"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Stunting prevalence vs. under-5 mortality<br>"
                            f"<sup>Spearman {_corr_label(r_st, p_st, len(sub))}</sup>",
                            f"Composite nutrition burden vs. under-5 mortality<br>"
                            f"<sup>Spearman {_corr_label(r_nb, p_nb, len(sub2))}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        for col_n, ss, xcol in [(1, sub, "stunting_pct_who"),
                                 (2, sub2, "nutrition_burden_score")]:
            mask = ss["region"] == region
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=ss.loc[mask, xcol], y=ss.loc[mask, "log_u5mr"],
                mode="markers", name=region or "Other",
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=ss.loc[mask, ["country_name", xcol,
                                          "u5_mortality_per1000"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              f"{xcol}: %{{customdata[1]:.2f}}<br>"
                              "U5MR: %{customdata[2]:.1f}/1k<extra></extra>",
                showlegend=(col_n == 1), legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub,  "stunting_pct_who",       "log_u5mr", row=1, col=1)
    _annotate_priority(fig, sub2, "nutrition_burden_score",  "log_u5mr", row=1, col=2)

    u5_ticks = [2, 5, 10, 20, 50, 100, 200]
    for col_n in [1, 2]:
        fig.update_yaxes(
            tickvals=[np.log10(v) for v in u5_ticks],
            ticktext=[str(v) for v in u5_ticks],
            title_text="Under-5 mortality rate (per 1,000, log scale)",
            row=1, col=col_n,
        )
    fig.update_xaxes(title_text="Stunting prevalence, WHO model (%)", row=1, col=1)
    fig.update_xaxes(title_text="Composite nutrition burden score (0–1)", row=1, col=2)
    fig.update_layout(
        title=dict(
            text="H9 — Undernutrition predicts under-5 mortality: the 45% pathway, visible in data<br>"
                 "<sup>Stunting alone and composite burden (stunting + anaemia + iron def. + LBW) "
                 "vs. U5MR across 150+ countries</sup>",
            font=dict(size=16, **FONT), x=0.05,
        ),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white", font=FONT, height=560,
    )

    print(f"\nH9 Findings:")
    print(f"  Stunting vs log(U5MR):          r={r_st:.3f}, p={p_st:.3e}, n={len(sub)}")
    print(f"  Composite burden vs log(U5MR):  r={r_nb:.3f}, p={p_nb:.3e}, n={len(sub2)}")

    _save(fig, "h9_undernutrition_child_mortality", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H10  Nutrition burden → human capital
# ═══════════════════════════════════════════════════════════════════════════════

def h10_nutrition_human_capital(df, show=False):
    """
    H10: The one-pager's central human capital claim — undernutrition
    impairs brain development, schooling, and productivity. Iodine
    deficiency alone is associated with substantial IQ losses; stunting
    in early life correlates with ~0.5–1 year less schooling and ~20%
    lower adult earnings.

    We test: nutrition_burden_score → HCI and GDP per capita (log scale).
    This is the full downstream pathway: nutrition → human capital → economy.
    """
    sub_hci = df[df["nutrition_burden_score"].notna() & df["hci_score"].notna()].copy()
    sub_gdp = df[
        df["nutrition_burden_score"].notna() &
        df["gdp_per_capita_ppp"].notna() &
        (df["gdp_per_capita_ppp"] > 0)
    ].copy()
    sub_gdp["log_gdp"] = np.log10(sub_gdp["gdp_per_capita_ppp"])

    r_hci, p_hci = stats.spearmanr(sub_hci["nutrition_burden_score"], sub_hci["hci_score"])
    r_gdp, p_gdp = stats.spearmanr(sub_gdp["nutrition_burden_score"], sub_gdp["log_gdp"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Nutrition burden → Human Capital Index<br>"
                            f"<sup>Spearman {_corr_label(r_hci, p_hci, len(sub_hci))}</sup>",
                            f"Nutrition burden → GDP per capita (PPP)<br>"
                            f"<sup>Spearman {_corr_label(r_gdp, p_gdp, len(sub_gdp))}</sup>",
                        ],
                        horizontal_spacing=0.12)

    for region, color in REGION_COLORS.items():
        mask_hci = sub_hci["region"] == region
        mask_gdp = sub_gdp["region"] == region
        for col_n, ss, mask, ycol in [
            (1, sub_hci, mask_hci, "hci_score"),
            (2, sub_gdp, mask_gdp, "log_gdp"),
        ]:
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=ss.loc[mask, "nutrition_burden_score"],
                y=ss.loc[mask, ycol],
                mode="markers", name=region or "Other",
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=ss.loc[mask, ["country_name", "nutrition_burden_score",
                                          "hci_score" if col_n == 1 else "gdp_per_capita_ppp",
                                          "stunting_pct_who"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "Burden: %{customdata[1]:.2f}<br>"
                              f"{'HCI' if col_n==1 else 'GDP/cap'}: %{{customdata[2]:.{'2f' if col_n==1 else '0f'}}}<br>"
                              "Stunting: %{customdata[3]:.0f}%<extra></extra>",
                showlegend=(col_n == 1), legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub_hci, "nutrition_burden_score", "hci_score",  row=1, col=1)
    _annotate_priority(fig, sub_gdp, "nutrition_burden_score", "log_gdp",    row=1, col=2)

    gdp_ticks = [500, 1000, 2000, 5000, 10000, 25000, 60000]
    fig.update_yaxes(title_text="Human Capital Index (0 = no human capital, 1 = full)", row=1, col=1)
    fig.update_yaxes(
        tickvals=[np.log10(v) for v in gdp_ticks],
        ticktext=[f"${v:,}" for v in gdp_ticks],
        title_text="GDP per capita PPP (log scale)",
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Composite nutrition burden score (0–1, higher = worse)")
    fig.update_layout(
        title=dict(
            text="H10 — Nutrition burden is a powerful predictor of human capital and economic productivity",
            font=dict(size=17, **FONT), x=0.05,
        ),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white", font=FONT, height=560,
    )

    print(f"\nH10 Findings:")
    print(f"  Burden vs HCI:      r={r_hci:.3f}, p={p_hci:.3e}, n={len(sub_hci)}")
    print(f"  Burden vs log(GDP): r={r_gdp:.3f}, p={p_gdp:.3e}, n={len(sub_gdp)}")
    bottom_hci = sub_hci.nsmallest(8, "hci_score")[
        ["country_name", "hci_score", "nutrition_burden_score", "stunting_pct_who"]
    ]
    print(f"  Lowest HCI countries:\n{bottom_hci.to_string(index=False)}")

    _save(fig, "h10_nutrition_human_capital", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# H11  Food insecurity → stunting → child mortality (the upstream pathway)
# ═══════════════════════════════════════════════════════════════════════════════

def h11_food_insecurity_pathway(df, show=False):
    """
    H11: The one-pager's upstream pathway — structural food systems
    constraints (food insecurity) drive inadequate intake, which drives
    stunting, which drives child mortality. This is the foundational
    justification for food-systems and fortification investment.

    We test the chain: food_insecurity → stunting → U5MR
    using scatter plots and Spearman correlations at each link.
    """
    sub = df[
        df["food_insecurity_mod_sev_pct"].notna() &
        df["stunting_pct_who"].notna() &
        df["u5_mortality_per1000"].notna()
    ].copy()
    sub["log_u5mr"] = np.log10(sub["u5_mortality_per1000"].clip(lower=0.5))

    r_fs_st, p_fs_st = stats.spearmanr(sub["food_insecurity_mod_sev_pct"], sub["stunting_pct_who"])
    r_st_u5, p_st_u5 = stats.spearmanr(sub["stunting_pct_who"], sub["log_u5mr"])
    r_fs_u5, p_fs_u5 = stats.spearmanr(sub["food_insecurity_mod_sev_pct"], sub["log_u5mr"])

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[
                            f"Food insecurity → stunting<br>"
                            f"<sup>Spearman {_corr_label(r_fs_st, p_fs_st, len(sub))}</sup>",
                            f"Stunting → child mortality<br>"
                            f"<sup>Spearman {_corr_label(r_st_u5, p_st_u5, len(sub))}</sup>",
                            f"Food insecurity → child mortality (direct)<br>"
                            f"<sup>Spearman {_corr_label(r_fs_u5, p_fs_u5, len(sub))}</sup>",
                        ],
                        horizontal_spacing=0.09)

    u5_ticks = [2, 5, 10, 20, 50, 100, 200]

    for region, color in REGION_COLORS.items():
        mask = sub["region"] == region
        if mask.sum() == 0:
            continue
        for col_n, xcol, ycol in [
            (1, "food_insecurity_mod_sev_pct", "stunting_pct_who"),
            (2, "stunting_pct_who",            "log_u5mr"),
            (3, "food_insecurity_mod_sev_pct", "log_u5mr"),
        ]:
            fig.add_trace(go.Scatter(
                x=sub.loc[mask, xcol], y=sub.loc[mask, ycol],
                mode="markers", name=region or "Other",
                marker=dict(color=color, size=6, opacity=0.7),
                customdata=sub.loc[mask, ["country_name", xcol, ycol]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "%{customdata[1]:.1f} → %{customdata[2]:.1f}<extra></extra>",
                showlegend=(col_n == 1), legendgroup=region,
            ), row=1, col=col_n)

    _annotate_priority(fig, sub, "food_insecurity_mod_sev_pct", "stunting_pct_who", row=1, col=1)
    _annotate_priority(fig, sub, "stunting_pct_who", "log_u5mr", row=1, col=2)
    _annotate_priority(fig, sub, "food_insecurity_mod_sev_pct", "log_u5mr", row=1, col=3)

    fig.update_xaxes(title_text="Moderate/severe food insecurity (%)", row=1, col=1)
    fig.update_xaxes(title_text="Moderate/severe food insecurity (%)", row=1, col=3)
    fig.update_xaxes(title_text="Stunting prevalence (%)", row=1, col=2)
    fig.update_yaxes(title_text="Stunting prevalence (%)", row=1, col=1)
    for col_n in [2, 3]:
        fig.update_yaxes(
            tickvals=[np.log10(v) for v in u5_ticks],
            ticktext=[str(v) for v in u5_ticks],
            title_text="Under-5 mortality (per 1,000, log scale)",
            row=1, col=col_n,
        )

    fig.update_layout(
        title=dict(
            text="H11 — The upstream pathway: food insecurity → stunting → child mortality",
            font=dict(size=17, **FONT), x=0.05,
        ),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white", font=FONT, height=540,
        width=1400,
    )

    print(f"\nH11 Findings:")
    print(f"  Food insecurity → stunting: r={r_fs_st:.3f}, p={p_fs_st:.3e}")
    print(f"  Stunting → U5MR:            r={r_st_u5:.3f}, p={p_st_u5:.3e}")
    print(f"  Food insecurity → U5MR:     r={r_fs_u5:.3f}, p={p_fs_u5:.3e}, n={len(sub)}")

    _save(fig, "h11_food_insecurity_pathway", show)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(show=False):
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} countries loaded, population data for "
          f"{df['population'].notna().sum()} countries")

    print("\n" + "=" * 60)
    print("Running hypothesis tests")
    print("=" * 60)

    h1_vaccination_measles(df, show)
    h2_anc_birth_outcomes(df, show)
    h3_malaria_anaemia(df, show)
    h4_hiv_tb(df, show)
    h5_system_vs_burden(df, show)
    h6_lsff_gap(df, show)
    h7_vitamin_a_measles(df, show)
    h8_maternal_anaemia_mortality(df, show)
    h9_undernutrition_child_mortality(df, show)
    h10_nutrition_human_capital(df, show)
    h11_food_insecurity_pathway(df, show)
    burden_heatmap(df, show)

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {OUT.relative_to(ROOT)}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Open figures in browser interactively")
    args = parser.parse_args()
    run(show=args.show)
