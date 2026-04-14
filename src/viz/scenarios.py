"""
Intervention scenario calculator for the Malnutrition Data Commons.

Fits cross-country OLS regression models at runtime and projects the
expected outcome change when a coverage/intervention indicator is moved
from its current level to a user-specified target in a given country.

Framing: "Countries with X% coverage tend to have Y% outcome."
These are directional signals for portfolio prioritisation, not forecasts.

Usage:
    from src.viz.scenarios import INTERVENTION_CHAINS, fit_model, project_outcome, scenario_scatter
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Styling (mirrors insights.py) ─────────────────────────────────────────────
FOUNDATION_BLUE = "#003366"
ACCENT_ORANGE   = "#E87722"
BG_LIGHT        = "#F8F9FA"
FONT            = dict(family="Arial, sans-serif", color="#1A1A2E")

REGION_COLORS = {
    "Sub-Saharan Africa":         "#E63946",
    "South Asia":                 "#F4A261",
    "East Asia & Pacific":        "#2A9D8F",
    "Latin America & Caribbean":  "#457B9D",
    "Middle East & North Africa": "#A8DADC",
    "Europe & Central Asia":      "#A8A8A8",
    "North America":              "#CCCCCC",
    "":                           "#BBBBBB",
}

PRIORITY_ISO3 = ["IND", "PAK", "BGD", "NGA", "ETH", "COD"]

# Approximate crude birth rate per 1,000 population by WHO region
REGION_BIRTH_RATES: dict[str, float] = {
    "Sub-Saharan Africa":         37.0,
    "South Asia":                 21.0,
    "South-East Asia":            18.0,
    "Western Pacific":            12.0,
    "Eastern Mediterranean":      25.0,
    "Americas":                   16.0,
    "Europe":                     10.0,
    "_default":                   22.0,
}


# ── Intervention chain registry ───────────────────────────────────────────────
# Each chain defines one intervention input → one or more downstream outcomes.
# "steps" is an ordered list of {x_col, y_col, y_label, y_unit, log_y} dicts.
# For two-step chains (two_step=True) each step feeds sequentially.

INTERVENTION_CHAINS: dict[str, dict] = {
    "anc4_birth_outcomes": {
        "label":       "Scale up ANC4+ coverage",
        "description": (
            "Increase the share of pregnant women receiving 4+ antenatal care visits. "
            "ANC4 is the primary delivery platform for iron/folate supplementation, "
            "risk detection, and maternal nutrition counseling."
        ),
        "input_col":   "anc4_coverage_pct",
        "input_label": "ANC4+ coverage",
        "input_unit":  "%",
        "target_default": 80,
        "two_step": False,
        "steps": [
            {"x_col": "anc4_coverage_pct", "y_col": "low_birthweight_pct",
             "y_label": "Low birthweight prevalence", "y_unit": "%", "log_y": False},
            {"x_col": "anc4_coverage_pct", "y_col": "preterm_birth_rate_pct",
             "y_label": "Preterm birth rate", "y_unit": "%", "log_y": False},
            {"x_col": "anc4_coverage_pct", "y_col": "maternal_mortality_per100k",
             "y_label": "Maternal mortality ratio", "y_unit": "per 100k", "log_y": True},
        ],
        "pop_impact_type": "births",
        "evidence_note": "H2: r = −0.47*** (LBW), −0.44*** (preterm) | H8: r = −0.74*** (MMR)",
    },
    "mcv1_measles": {
        "label":       "Scale up MCV1 vaccination coverage",
        "description": (
            "Increase measles first-dose vaccination coverage. "
            "MCV1→MCV2 dropout amplifies risk; target should include "
            "a follow-through strategy to close the dropout gap."
        ),
        "input_col":   "mcv1_coverage_pct",
        "input_label": "MCV1 coverage",
        "input_unit":  "%",
        "target_default": 95,
        "two_step": False,
        "steps": [
            {"x_col": "mcv1_coverage_pct", "y_col": "measles_per100k",
             "y_label": "Measles cases", "y_unit": "per 100k", "log_y": True},
        ],
        "pop_impact_type": "population",
        "evidence_note": "H1: r = −0.27** (MCV1 vs log measles)",
    },
    "lsff_stunting": {
        "label":       "Scale up LSFF wheat flour fortification",
        "description": (
            "Increase large-scale food fortification coverage of wheat flour. "
            "This is a two-step pathway: fortification reduces iron deficiency, "
            "which in turn reduces stunting prevalence."
        ),
        "input_col":   "lsff_coverage_proxy_pct",
        "input_label": "LSFF coverage",
        "input_unit":  "%",
        "target_default": 75,
        "two_step": True,
        "steps": [
            {"x_col": "lsff_coverage_proxy_pct", "y_col": "iron_deficiency_pct",
             "y_label": "Iron deficiency prevalence", "y_unit": "%", "log_y": False},
            {"x_col": "iron_deficiency_pct", "y_col": "stunting_pct_who",
             "y_label": "Stunting prevalence <5", "y_unit": "%", "log_y": False},
        ],
        "pop_impact_type": "u5_population",
        "evidence_note": "H6: LSFF intervention gap analysis (moderate signal)",
    },
    "stunting_outcomes": {
        "label":       "Reduce stunting prevalence (direct)",
        "description": (
            "Model the downstream effect of a direct reduction in stunting — "
            "as would result from an integrated package of nutrition-sensitive "
            "and nutrition-specific interventions. "
            "This is the strongest empirical relationship in the dataset."
        ),
        "input_col":   "stunting_pct_who",
        "input_label": "Stunting prevalence <5",
        "input_unit":  "%",
        "target_default": None,   # computed as 20% relative reduction from current
        "invert_slider": True,    # lower is better → slider should move downward
        "two_step": False,
        "steps": [
            {"x_col": "stunting_pct_who", "y_col": "u5_mortality_per1000",
             "y_label": "Under-5 mortality rate", "y_unit": "per 1k", "log_y": False},
            {"x_col": "stunting_pct_who", "y_col": "hci_score",
             "y_label": "Human Capital Index", "y_unit": "score", "log_y": False},
        ],
        "pop_impact_type": "u5_population",
        "evidence_note": "H9: r = +0.81*** (stunting→U5MR) | H10: r = −0.85*** (burden→HCI)",
    },
}


# ── Core model fitting ────────────────────────────────────────────────────────

def fit_model(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    log_y: bool = False,
) -> dict | None:
    """
    Fit a simple OLS regression y ~ x on the full dataset.

    If log_y=True, fits on log10(y) and returns log-space coefficients;
    callers should use project_outcome() which handles back-transformation.

    Returns a dict with slope, intercept, stderr, r, p, n, and the
    x/y arrays used (for scatter plotting).
    """
    sub = df[[x_col, y_col]].dropna()
    if log_y:
        sub = sub[sub[y_col] > 0].copy()
        sub["_y"] = np.log10(sub[y_col])
    else:
        sub["_y"] = sub[y_col]

    if len(sub) < 10:
        return None

    slope, intercept, r, p, stderr = stats.linregress(sub[x_col], sub["_y"])
    return {
        "slope":      slope,
        "intercept":  intercept,
        "stderr":     stderr,
        "r":          r,
        "p":          p,
        "n":          len(sub),
        "log_y":      log_y,
        "x_col":      x_col,
        "y_col":      y_col,
        "x_data":     sub[x_col].values,
        "y_data":     sub[y_col].values,     # original scale
        "y_log_data": sub["_y"].values,      # log scale (same as y_data if log_y=False)
    }


def project_outcome(
    model: dict,
    x_current: float,
    x_target: float,
) -> dict:
    """
    Project the outcome value at x_target given a fitted model.

    For log-scale models, results are back-transformed to original scale.
    CI is approximate: ±1.96 × stderr × |Δx| (regression uncertainty only).

    Returns current_y, projected_y, delta_y, ci_lower, ci_upper (all in
    original scale), plus log-scale values for chart positioning.
    """
    log_y = model["log_y"]

    # Predictions in model space (log or linear)
    y_current_model   = model["intercept"] + model["slope"] * x_current
    y_projected_model = model["intercept"] + model["slope"] * x_target
    delta_model       = y_projected_model - y_current_model
    ci_half_model     = 1.96 * model["stderr"] * abs(x_target - x_current)

    if log_y:
        y_current   = 10 ** y_current_model
        y_projected = 10 ** y_projected_model
        # CI in original scale (asymmetric)
        ci_lower = 10 ** (y_projected_model - ci_half_model)
        ci_upper = 10 ** (y_projected_model + ci_half_model)
        delta_y  = y_projected - y_current
    else:
        y_current   = y_current_model
        y_projected = y_projected_model
        ci_lower    = y_projected - ci_half_model
        ci_upper    = y_projected + ci_half_model
        delta_y     = delta_model

    return {
        "y_current":         y_current,
        "y_projected":       y_projected,
        "delta_y":           delta_y,
        "ci_lower":          ci_lower,
        "ci_upper":          ci_upper,
        # Model-space values (for log-scale chart positioning)
        "y_current_model":   y_current_model,
        "y_projected_model": y_projected_model,
    }


def project_two_step(
    df: pd.DataFrame,
    chain: dict,
    x_current: float,
    x_target: float,
) -> dict:
    """
    Sequential two-step projection: input → intermediate → final outcome.

    Propagates uncertainty (RSS of both regression standard errors × Δx).
    Returns projections for both steps.
    """
    step1, step2 = chain["steps"][0], chain["steps"][1]
    model1 = fit_model(df, step1["x_col"], step1["y_col"], step1.get("log_y", False))
    model2 = fit_model(df, step2["x_col"], step2["y_col"], step2.get("log_y", False))

    if model1 is None or model2 is None:
        return {}

    proj1 = project_outcome(model1, x_current, x_target)
    # Use projected intermediate as input to step 2
    intermediate_current   = proj1["y_current"]
    intermediate_projected = proj1["y_projected"]
    proj2 = project_outcome(model2, intermediate_current, intermediate_projected)

    # Propagate CI: combined SE from both steps
    se_combined = np.sqrt(
        (model1["stderr"] * abs(x_target - x_current)) ** 2 +
        (model2["stderr"] * abs(intermediate_projected - intermediate_current)) ** 2
    ) * 1.96
    proj2["ci_lower"] = proj2["y_projected"] - se_combined
    proj2["ci_upper"] = proj2["y_projected"] + se_combined

    return {"step1": (model1, proj1), "step2": (model2, proj2)}


def population_impact(
    country_row: pd.Series,
    population: float,
    delta_outcome: float,
    impact_type: str,
) -> dict:
    """
    Estimate the approximate number of events averted per year.

    impact_type:
      "births"        — applies delta to annual births
      "population"    — applies per-100k rate to total population
      "u5_population" — applies to estimated U5 population (~17% of total)
    """
    region = country_row.get("who_region", "_default")
    birth_rate = REGION_BIRTH_RATES.get(region, REGION_BIRTH_RATES["_default"])
    annual_births = population * birth_rate / 1000
    u5_pop = population * 0.17

    if impact_type == "births":
        n_averted = annual_births * abs(delta_outcome) / 100
        denom_label = f"{annual_births/1e6:.1f}M births/yr"
    elif impact_type == "population":
        n_averted = population * abs(delta_outcome) / 100_000
        denom_label = f"{population/1e6:.0f}M population"
    else:  # u5_population
        n_averted = u5_pop * abs(delta_outcome) / 1000 if "mortality" in impact_type else \
                    u5_pop * abs(delta_outcome) / 100
        denom_label = f"{u5_pop/1e6:.1f}M children <5"

    direction = "fewer" if delta_outcome < 0 else "additional"
    return {
        "n_averted":    n_averted,
        "direction":    direction,
        "denom_label":  denom_label,
    }


# ── Figure builders ───────────────────────────────────────────────────────────

def scenario_scatter(
    df: pd.DataFrame,
    step: dict,
    model: dict,
    projection: dict,
    iso3: str,
    x_current: float,
    x_target: float,
    country_name: str,
    x_source_label: str = "Observed",
    height: int = 480,
) -> go.Figure:
    """
    Scatter of all countries on the x→y regression plane, with:
    - Regression line (dashed gray)
    - All countries colored by region (small, semi-transparent)
    - Priority countries outlined
    - Selected country as a filled orange dot
    - Orange arrow from current to projected position
    - Corr. annotation (r, p, n)
    """
    log_y = step.get("log_y", False)
    x_col = step["x_col"]
    y_col = step["y_col"]
    y_label = step["y_label"]
    y_unit  = step["y_unit"]
    x_label = INTERVENTION_CHAINS.get(
        next((k for k, v in INTERVENTION_CHAINS.items() if v["input_col"] == x_col), ""),
        {}
    ).get("input_label", x_col.replace("_", " ").title())

    plot_df = df[[x_col, y_col, "iso3", "country_name", "region"]].dropna()
    if log_y:
        plot_df = plot_df[plot_df[y_col] > 0].copy()

    fig = go.Figure()

    # ── All non-priority countries ────────────────────────────────────────────
    for region, color in REGION_COLORS.items():
        mask = (plot_df["region"] == region) & (~plot_df["iso3"].isin(PRIORITY_ISO3))
        sub = plot_df[mask]
        if sub.empty:
            continue
        y_vals = np.log10(sub[y_col]) if log_y else sub[y_col]
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=y_vals,
            mode="markers",
            name=region or "Other",
            marker=dict(color=color, size=6, opacity=0.30),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                f"{x_col.replace('_',' ')}: %{{x:.1f}}%<br>"
                f"{y_label}: %{{customdata[1]:.1f}} {y_unit}<extra></extra>"
            ),
            customdata=np.stack([sub["country_name"].values,
                                 sub[y_col].values], axis=1),
            legendgroup=region,
            showlegend=True,
        ))

    # ── Priority countries (except selected) ─────────────────────────────────
    for piso in PRIORITY_ISO3:
        if piso == iso3:
            continue
        row = plot_df[plot_df["iso3"] == piso]
        if row.empty:
            continue
        y_v = np.log10(float(row[y_col].iloc[0])) if log_y else float(row[y_col].iloc[0])
        pname = row["country_name"].iloc[0].split(",")[0]
        fig.add_trace(go.Scatter(
            x=[float(row[x_col].iloc[0])], y=[y_v],
            mode="markers+text",
            marker=dict(symbol="circle-open", size=12,
                        line=dict(width=2, color="#1A1A2E"), color="rgba(0,0,0,0)"),
            text=[pname], textposition="top center",
            textfont=dict(size=9, color="#1A1A2E"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Regression line ───────────────────────────────────────────────────────
    xs = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 120)
    ys_model = model["intercept"] + model["slope"] * xs
    ys_chart  = ys_model  # already in log space if log_y
    stars = "***" if model["p"] < 0.001 else "**" if model["p"] < 0.01 else "*" if model["p"] < 0.05 else "n.s."
    fig.add_trace(go.Scatter(
        x=xs, y=ys_chart, mode="lines",
        line=dict(color="#AAAAAA", dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
        name=f"r = {model['r']:.2f} {stars}, n={model['n']}",
    ))

    # ── Arrow: current → projected ────────────────────────────────────────────
    y_cur_chart  = projection["y_current_model"]   # log or linear
    y_proj_chart = projection["y_projected_model"]

    fig.add_annotation(
        x=x_target,        y=y_proj_chart,
        ax=x_current,      ay=y_cur_chart,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=4, arrowwidth=2.5, arrowcolor=ACCENT_ORANGE,
        text="", showarrow=True,
    )

    # ── Selected country — current position ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_current], y=[y_cur_chart],
        mode="markers",
        marker=dict(color=ACCENT_ORANGE, size=14, opacity=1.0,
                    line=dict(color="white", width=1.5)),
        name=country_name,
        text=[f"{country_name}<br>({x_source_label})"],
        hovertemplate="<b>%{text}</b><br>"
                      f"{x_col.replace('_',' ')}: %{{x:.1f}}%<extra></extra>",
        showlegend=False,
    ))

    # Target position (ghost dot)
    fig.add_trace(go.Scatter(
        x=[x_target], y=[y_proj_chart],
        mode="markers",
        marker=dict(color=ACCENT_ORANGE, size=14, opacity=0.35,
                    line=dict(color=ACCENT_ORANGE, width=1.5),
                    symbol="circle-open"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_annotation(
        x=x_target, y=y_proj_chart,
        text=f" Target: {x_target:.0f}%",
        font=dict(size=10, color=ACCENT_ORANGE),
        showarrow=False, xanchor="left",
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    # Country label at current position
    fig.add_annotation(
        x=x_current, y=y_cur_chart,
        text=f" {country_name}",
        font=dict(size=10, color="#1A1A2E"),
        showarrow=False, xanchor="left",
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    # r/p annotation
    fig.add_annotation(
        x=0.98, y=0.97, xref="paper", yref="paper",
        text=f"Spearman r = {model['r']:.2f} {stars}  (n={model['n']} countries)",
        font=dict(size=11, color="#555"),
        showarrow=False, xanchor="right",
        bgcolor="rgba(255,255,255,0.85)", borderpad=4,
    )

    # ── Axes ─────────────────────────────────────────────────────────────────
    if log_y:
        raw_y = plot_df[y_col]
        tick_vals_raw = [v for v in [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
                         if v >= raw_y.min() * 0.5 and v <= raw_y.max() * 2]
        fig.update_yaxes(
            tickvals=[np.log10(v) for v in tick_vals_raw],
            ticktext=[str(v) for v in tick_vals_raw],
            title_text=f"{y_label} ({y_unit}) — log scale",
        )
    else:
        fig.update_yaxes(title_text=f"{y_label} ({y_unit})")

    fig.update_xaxes(title_text=f"{x_label} (%)")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor=BG_LIGHT,
        font=FONT,
        height=height,
        legend=dict(font=dict(size=10), orientation="v", x=1.02),
        margin=dict(l=60, r=160, t=30, b=50),
    )
    return fig


def two_step_scatter(
    df: pd.DataFrame,
    chain: dict,
    iso3: str,
    x_current: float,
    x_target: float,
    two_step_result: dict,
    country_name: str,
    height: int = 420,
) -> go.Figure:
    """Two-panel scatter for a sequential two-step intervention chain."""
    model1, proj1 = two_step_result["step1"]
    model2, proj2 = two_step_result["step2"]
    step1, step2  = chain["steps"][0], chain["steps"][1]

    inter_current   = proj1["y_current"]
    inter_projected = proj1["y_projected"]

    titles = [
        f"Step 1: {chain['input_label']} → {step1['y_label'].split('<')[0].strip()}",
        f"Step 2: {step1['y_label'].split('<')[0].strip()} → {step2['y_label'].split('<')[0].strip()}",
    ]

    fig = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.14)

    for panel, (step, model, proj, xc, xt, x_col) in enumerate([
        (step1, model1, proj1, x_current, x_target, step1["x_col"]),
        (step2, model2, proj2, inter_current, inter_projected, step2["x_col"]),
    ], start=1):
        log_y = step.get("log_y", False)
        y_col = step["y_col"]
        plot_df = df[[x_col, y_col, "region"]].dropna()
        if log_y:
            plot_df = plot_df[plot_df[y_col] > 0].copy()

        for region, color in REGION_COLORS.items():
            mask = plot_df["region"] == region
            sub = plot_df[mask]
            if sub.empty:
                continue
            y_chart = np.log10(sub[y_col]) if log_y else sub[y_col]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=y_chart,
                mode="markers", name=region or "Other",
                marker=dict(color=color, size=5, opacity=0.30),
                showlegend=(panel == 1),
                legendgroup=region,
            ), row=1, col=panel)

        xs = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
        ys = model["intercept"] + model["slope"] * xs
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="#AAAAAA", dash="dash", width=1.5),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=panel)

        yc_chart = proj["y_current_model"]
        yp_chart = proj["y_projected_model"]

        fig.add_annotation(
            x=xt, y=yp_chart, ax=xc, ay=yc_chart,
            xref=f"x{'' if panel == 1 else panel}",
            yref=f"y{'' if panel == 1 else panel}",
            axref=f"x{'' if panel == 1 else panel}",
            ayref=f"y{'' if panel == 1 else panel}",
            arrowhead=4, arrowwidth=2.5, arrowcolor=ACCENT_ORANGE,
            text="", showarrow=True,
        )
        fig.add_trace(go.Scatter(
            x=[xc], y=[yc_chart], mode="markers",
            marker=dict(color=ACCENT_ORANGE, size=12,
                        line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=panel)

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor=BG_LIGHT, font=FONT,
        height=height,
        legend=dict(font=dict(size=9), orientation="v", x=1.02),
        margin=dict(l=50, r=140, t=60, b=50),
    )
    return fig
