"""
Visualization functions for the MUMTA Pakistan cohort data.

MUMTA is a multi-arm RCT in Pakistan with arms:
  A = Control, B = Maamta, C = Maamta + Azithromycin, D = Maamta + Choline + Nicotinamide

All functions return Plotly go.Figure objects for use in the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette / style ──────────────────────────────────────────────────────────

FOUNDATION_BLUE = "#003366"
ACCENT_ORANGE = "#E87722"
BG_LIGHT = "#F8F9FA"
FONT = dict(family="Arial, sans-serif", color="#1A1A2E")

ARM_COLORS = {
    "A": "#888888",  # Control — grey
    "B": "#E87722",  # Maamta — orange
    "C": "#2A9D8F",  # Maamta + Azithromycin — teal
    "D": "#7C3AED",  # Maamta + Choline + Nicotinamide — purple
}

ARM_LABELS = {
    "A": "Control",
    "B": "Maamta",
    "C": "Maamta + Azithro",
    "D": "Maamta + Choline + Nic",
}

ARM_ORDER = ["A", "B", "C", "D"]

# Timepoint ordering for longitudinal charts
TIMEPOINT_ORDER = ["19wk", "32wk", "1-2mo", "3-4mo", "5-6mo", "12mo"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_figure(msg="No data available", height=300):
    """Return a blank figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#999"),
    )
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=FONT,
    )
    return fig


def _is_empty(df):
    """Check if a DataFrame is None or empty."""
    return df is None or (isinstance(df, pd.DataFrame) and df.empty)


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a hex color like '#888888' to 'rgba(136,136,136,0.12)'."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    elif len(h) == 3:
        r, g, b = int(h[0]*2, 16), int(h[1]*2, 16), int(h[2]*2, 16)
    else:
        return f"rgba(128,128,128,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def _binomial_ci(k, n, z=1.96):
    """Wilson score 95 % CI for a binomial proportion, returns (lo, hi) as %."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    lo = max(0.0, centre - margin) * 100
    hi = min(1.0, centre + margin) * 100
    return (lo, hi)


def _base_layout(height=350, **kwargs):
    """Common layout options."""
    layout = dict(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=FONT,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    layout.update(kwargs)
    return layout


# ── 1. Cohort overview metrics ──────────────────────────────────────────────

def cohort_overview_metrics(cohort_df):
    """Return a dict of summary statistics for display as st.metric cards.

    Keys returned:
        n_enrolled, n_live_births, n_stillbirths, n_miscarriages,
        lbw_pct, preterm_pct, stunted_birth_pct,
        mean_birth_weight, mean_gestational_age,
        anaemia_19wk_pct, anaemia_32wk_pct,
        iron_def_19wk_pct, iron_def_32wk_pct
    """
    if _is_empty(cohort_df):
        return {}

    df = cohort_df.copy()
    n = len(df)
    if "birth_outcome" in df.columns:
        _bo = df["birth_outcome"].dropna().str.lower()
        live = int(_bo.str.contains("live", na=False).sum())
        still = int(_bo.str.contains("still", na=False).sum())
        miscarriage = int(_bo.str.contains("miscarriage", na=False).sum())
    else:
        live = still = miscarriage = 0

    def _pct(col):
        valid = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if valid.empty:
            return None
        return round(valid.mean() * 100, 1)

    def _mean(col, decimals=1):
        valid = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if valid.empty:
            return None
        return round(valid.mean(), decimals)

    return dict(
        n_enrolled=n,
        n_live_births=int(live),
        n_stillbirths=int(still),
        n_miscarriages=int(miscarriage),
        lbw_pct=_pct("lbw"),
        preterm_pct=_pct("preterm"),
        stunted_birth_pct=_pct("stunted_at_birth"),
        mean_birth_weight=_mean("birth_weight_g", 0),
        mean_gestational_age=_mean("gestational_age_weeks", 1),
        anaemia_19wk_pct=_pct("anaemic_19wk"),
        anaemia_32wk_pct=_pct("anaemic_32wk"),
        iron_def_19wk_pct=_pct("iron_deficient_19wk"),
        iron_def_32wk_pct=_pct("iron_deficient_32wk"),
    )


# ── 2. Birth outcomes by arm ────────────────────────────────────────────────

def birth_outcomes_by_arm(cohort_df, height=350):
    """Grouped bar chart: LBW %, preterm %, stunted at birth % by arm with 95 % CI."""
    if _is_empty(cohort_df):
        return _empty_figure(height=height)

    outcomes = [
        ("lbw", "Low birthweight"),
        ("preterm", "Preterm"),
        ("stunted_at_birth", "Stunted at birth"),
    ]

    fig = go.Figure()
    for arm in ARM_ORDER:
        arm_df = cohort_df[cohort_df["arm"] == arm]
        if arm_df.empty:
            continue
        n_arm = len(arm_df)
        pcts, lo_errs, hi_errs, x_labels = [], [], [], []
        for col, label in outcomes:
            if col not in arm_df.columns:
                pcts.append(0)
                lo_errs.append(0)
                hi_errs.append(0)
                x_labels.append(label)
                continue
            k = int(arm_df[col].sum())
            p = k / n_arm * 100
            ci_lo, ci_hi = _binomial_ci(k, n_arm)
            pcts.append(round(p, 1))
            lo_errs.append(round(p - ci_lo, 1))
            hi_errs.append(round(ci_hi - p, 1))
            x_labels.append(label)

        fig.add_trace(go.Bar(
            name=ARM_LABELS.get(arm, arm),
            x=x_labels,
            y=pcts,
            error_y=dict(type="data", symmetric=False, array=hi_errs, arrayminus=lo_errs),
            marker_color=ARM_COLORS.get(arm, "#333"),
            hovertemplate="%{x}: %{y:.1f}% (n=" + str(n_arm) + ")<extra>" + ARM_LABELS.get(arm, arm) + "</extra>",
        ))

    fig.update_layout(
        **_base_layout(height=height),
        barmode="group",
        title=dict(text="Birth Outcomes by Trial Arm", font=dict(size=15, color=FOUNDATION_BLUE)),
        yaxis=dict(title="Prevalence (%)", gridcolor="#eee"),
        xaxis=dict(title=""),
    )
    return fig


# ── 3. Maternal anaemia trajectory ──────────────────────────────────────────

def maternal_anemia_trajectory(anemia_df, height=320):
    """Two-row subplot: top = Hb trajectory (19wk→32wk), bottom = iron deficiency %."""
    if _is_empty(anemia_df):
        return _empty_figure(height=height)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15,
        subplot_titles=["Haemoglobin (g/dL)", "Iron deficiency (%)"],
    )

    timepoints = ["19wk", "32wk"]
    tp_x = {tp: i for i, tp in enumerate(timepoints)}

    for arm in ARM_ORDER:
        arm_df = anemia_df[anemia_df["arm"] == arm]
        if arm_df.empty:
            continue
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)

        # ── Top panel: Hb trajectory ──
        hb_means, hb_ses, xs = [], [], []
        for tp in timepoints:
            tp_df = arm_df[arm_df["timepoint"] == tp]
            hb = tp_df["hemoglobin"].dropna()
            if hb.empty:
                continue
            hb_means.append(hb.mean())
            hb_ses.append(hb.std() / np.sqrt(len(hb)))
            xs.append(tp)

        if hb_means:
            fig.add_trace(go.Scatter(
                x=xs, y=hb_means, mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                error_y=dict(type="data", array=hb_ses, visible=True),
                hovertemplate="Hb: %{y:.1f} ± %{error_y.array:.2f} g/dL<extra>" + label + "</extra>",
                legendgroup=arm,
            ), row=1, col=1)

        # ── Bottom panel: Iron deficiency % ──
        id_pcts, id_xs = [], []
        for tp in timepoints:
            tp_df = arm_df[arm_df["timepoint"] == tp]
            iron = tp_df["iron_deficient"].dropna()
            if iron.empty:
                continue
            id_pcts.append(iron.mean() * 100)
            id_xs.append(tp)

        if id_pcts:
            fig.add_trace(go.Scatter(
                x=id_xs, y=id_pcts, mode="lines+markers",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=8, color=color, symbol="square"),
                hovertemplate="Iron def: %{y:.1f}%<extra>" + label + "</extra>",
                legendgroup=arm,
                showlegend=False,
            ), row=2, col=1)

    # Anaemia threshold line on top panel
    fig.add_hline(y=11, line_dash="dash", line_color="#CC3333", line_width=1, row=1, col=1,
                  annotation_text="Anaemia threshold (11 g/dL)", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#CC3333")

    fig.update_layout(
        **_base_layout(height=height),
        title=dict(text="Maternal Anaemia Trajectory", font=dict(size=15, color=FOUNDATION_BLUE)),
    )
    fig.update_yaxes(title_text="Hb (g/dL)", gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(title_text="%", gridcolor="#eee", row=2, col=1)

    return fig


# ── 4. Infant growth curves ─────────────────────────────────────────────────

def infant_growth_curves(growth_df, metric="laz", height=350):
    """Line chart: mean z-score by month (0–8) per arm with shaded ±1 SE band."""
    if _is_empty(growth_df):
        return _empty_figure(height=height)

    metric_labels = {
        "laz": ("Length-for-age z-score (LAZ)", "Stunting"),
        "waz": ("Weight-for-age z-score (WAZ)", "Underweight"),
        "wlz": ("Weight-for-length z-score (WLZ)", "Wasting"),
    }
    y_label, threshold_label = metric_labels.get(metric, (metric.upper(), "Threshold"))

    fig = go.Figure()
    months = sorted(growth_df["month"].dropna().unique())

    for arm in ARM_ORDER:
        arm_df = growth_df[growth_df["arm"] == arm]
        if arm_df.empty:
            continue
        color = ARM_COLORS.get(arm, "#333")
        label = ARM_LABELS.get(arm, arm)

        means, ses, valid_months = [], [], []
        for m in months:
            vals = arm_df.loc[arm_df["month"] == m, metric].dropna()
            if vals.empty:
                continue
            means.append(vals.mean())
            ses.append(vals.std() / np.sqrt(len(vals)))
            valid_months.append(m)

        if not means:
            continue

        means = np.array(means)
        ses = np.array(ses)
        upper = means + ses
        lower = means - ses

        # Shaded SE band
        fig.add_trace(go.Scatter(
            x=list(valid_months) + list(reversed(valid_months)),
            y=list(upper) + list(reversed(lower)),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=arm,
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=valid_months, y=means, mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate="Month %{x}: %{y:.2f}<extra>" + label + "</extra>",
            legendgroup=arm,
        ))

    # Threshold line at -2
    fig.add_hline(
        y=-2, line_dash="dash", line_color="#CC3333", line_width=1,
        annotation_text=f"{threshold_label} threshold (z = −2)",
        annotation_position="bottom left",
        annotation_font_size=10, annotation_font_color="#CC3333",
    )

    fig.update_layout(
        **_base_layout(height=height),
        title=dict(text=f"Infant Growth: {y_label}", font=dict(size=15, color=FOUNDATION_BLUE)),
        xaxis=dict(title="Age (months)", dtick=1, gridcolor="#eee"),
        yaxis=dict(title="Z-score", gridcolor="#eee"),
    )
    return fig


# ── 5. B. infantis colonization ─────────────────────────────────────────────

def binfantis_colonization(binfantis_df, height=320):
    """B. infantis positivity rate over timepoints — two subplot rows: maternal / infant stool."""
    if _is_empty(binfantis_df):
        return _empty_figure(height=height)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        subplot_titles=["Maternal stool", "Infant stool"],
    )

    specimen_types = ["maternal", "infant"]
    # Filter to known timepoints and sort
    all_tps = [tp for tp in TIMEPOINT_ORDER if tp in binfantis_df["timepoint"].unique()]

    for row_idx, specimen in enumerate(specimen_types, start=1):
        spec_df = binfantis_df[binfantis_df["specimen_type"] == specimen]
        for arm in ARM_ORDER:
            arm_df = spec_df[spec_df["arm"] == arm]
            if arm_df.empty:
                continue
            color = ARM_COLORS.get(arm, "#333")
            label = ARM_LABELS.get(arm, arm)

            tps, rates = [], []
            for tp in all_tps:
                tp_df = arm_df[arm_df["timepoint"] == tp]
                pos_col = "b_infantis_positive"
                if pos_col not in tp_df.columns or tp_df[pos_col].dropna().empty:
                    continue
                vals = tp_df[pos_col].dropna()
                tps.append(tp)
                rates.append(vals.mean() * 100)

            if not tps:
                continue

            fig.add_trace(go.Scatter(
                x=tps, y=rates, mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>",
                legendgroup=arm,
                showlegend=(row_idx == 1),
            ), row=row_idx, col=1)

    fig.update_layout(
        **_base_layout(height=height),
        title=dict(text="B. infantis Colonization", font=dict(size=15, color=FOUNDATION_BLUE)),
    )
    fig.update_yaxes(title_text="Positivity (%)", gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(title_text="Positivity (%)", gridcolor="#eee", row=2, col=1)
    return fig


# ── 6. Gut inflammation trajectory ──────────────────────────────────────────

def gut_inflammation_trajectory(inflammation_df, marker="mpo", height=320):
    """Median MPO or LCN-2 over timepoints by arm — two rows: maternal / infant."""
    if _is_empty(inflammation_df):
        return _empty_figure(height=height)

    marker_col = marker.lower()
    if marker_col not in inflammation_df.columns:
        return _empty_figure(msg=f"Column '{marker_col}' not found", height=height)

    marker_label = {"mpo": "MPO (ng/mL)", "lcn2": "LCN-2 (ng/mL)"}.get(marker_col, marker_col)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        subplot_titles=["Maternal stool", "Infant stool"],
    )

    specimen_types = ["maternal", "infant"]
    all_tps = [tp for tp in TIMEPOINT_ORDER if tp in inflammation_df["timepoint"].unique()]

    for row_idx, specimen in enumerate(specimen_types, start=1):
        spec_df = inflammation_df[inflammation_df["specimen_type"] == specimen]
        for arm in ARM_ORDER:
            arm_df = spec_df[spec_df["arm"] == arm]
            if arm_df.empty:
                continue
            color = ARM_COLORS.get(arm, "#333")
            label = ARM_LABELS.get(arm, arm)

            tps, medians, ns = [], [], []
            for tp in all_tps:
                vals = arm_df.loc[arm_df["timepoint"] == tp, marker_col].dropna()
                if vals.empty:
                    continue
                tps.append(tp)
                medians.append(vals.median())
                ns.append(len(vals))

            if not tps:
                continue

            hover_text = [f"{tp}: median={m:.1f}, n={n}" for tp, m, n in zip(tps, medians, ns)]
            fig.add_trace(go.Scatter(
                x=tps, y=medians, mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                text=hover_text,
                hovertemplate="%{text}<extra>" + label + "</extra>",
                legendgroup=arm,
                showlegend=(row_idx == 1),
            ), row=row_idx, col=1)

    fig.update_layout(
        **_base_layout(height=height),
        title=dict(text=f"Gut Inflammation: {marker_label}", font=dict(size=15, color=FOUNDATION_BLUE)),
    )
    fig.update_yaxes(title_text=marker_label, gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(title_text=marker_label, gridcolor="#eee", row=2, col=1)
    return fig


# ── 7. Microbiome composition ───────────────────────────────────────────────

def microbiome_composition(microbiome_df, height=400):
    """Stacked bar: top 15 genera by mean relative abundance, per timepoint, optionally by arm."""
    if _is_empty(microbiome_df):
        return _empty_figure(height=height)

    df = microbiome_df.copy()

    # Identify genus columns (everything that isn't metadata)
    meta_cols = {"sample_id", "study_id", "timepoint", "arm"}
    genus_cols = [c for c in df.columns if c not in meta_cols]
    if not genus_cols:
        return _empty_figure(msg="No genus columns found", height=height)

    # Determine top 15 genera by overall mean abundance
    genus_means = df[genus_cols].mean().sort_values(ascending=False)
    top15 = genus_means.head(15).index.tolist()
    other_cols = [c for c in genus_cols if c not in top15]

    # Build a colour palette for genera
    genus_palette = [
        "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
        "#003366", "#E87722", "#7C3AED", "#1D3557", "#457B9D",
        "#A8DADC", "#F1FAEE", "#606C38", "#DDA15E", "#BC6C25",
        "#AAAAAA",  # "Other"
    ]

    timepoints = [tp for tp in ["19wk", "32wk"] if tp in df["timepoint"].unique()]
    arms_present = [a for a in ARM_ORDER if a in df["arm"].unique()]

    # Build x-axis categories: arm × timepoint
    x_labels = []
    for arm in arms_present:
        for tp in timepoints:
            x_labels.append(f"{ARM_LABELS.get(arm, arm)}<br>{tp}")

    fig = go.Figure()
    for i, genus in enumerate(top15):
        y_vals = []
        for arm in arms_present:
            for tp in timepoints:
                subset = df[(df["arm"] == arm) & (df["timepoint"] == tp)]
                if subset.empty or genus not in subset.columns:
                    y_vals.append(0)
                else:
                    y_vals.append(subset[genus].mean())
        fig.add_trace(go.Bar(
            name=genus,
            x=x_labels,
            y=y_vals,
            marker_color=genus_palette[i % len(genus_palette)],
            hovertemplate=genus + ": %{y:.1f}%<extra></extra>",
        ))

    # "Other" category
    if other_cols:
        y_vals = []
        for arm in arms_present:
            for tp in timepoints:
                subset = df[(df["arm"] == arm) & (df["timepoint"] == tp)]
                if subset.empty:
                    y_vals.append(0)
                else:
                    y_vals.append(subset[other_cols].sum(axis=1).mean())
        fig.add_trace(go.Bar(
            name="Other",
            x=x_labels,
            y=y_vals,
            marker_color=genus_palette[-1],
            hovertemplate="Other: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(height=height),
        barmode="stack",
        title=dict(text="Gut Microbiome Composition (Top 15 Genera)", font=dict(size=15, color=FOUNDATION_BLUE)),
        yaxis=dict(title="Relative abundance (%)", gridcolor="#eee"),
        xaxis=dict(title="", tickangle=-30),
        legend=dict(font=dict(size=9), orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    return fig


# ── 8. Model vs cohort comparison ───────────────────────────────────────────

def model_vs_cohort_comparison(cohort_df, snapshot_row, height=300):
    """Side-by-side horizontal bars: MUMTA cohort ground-truth vs commons modelled estimates."""
    if _is_empty(cohort_df) or snapshot_row is None:
        return _empty_figure(height=height)

    # If snapshot_row is a DataFrame, take the first row
    if isinstance(snapshot_row, pd.DataFrame):
        if snapshot_row.empty:
            return _empty_figure(height=height)
        snapshot_row = snapshot_row.iloc[0]

    # ── Compute cohort values ──
    def _cohort_pct(col):
        if col not in cohort_df.columns:
            return None
        vals = cohort_df[col].dropna()
        return round(vals.mean() * 100, 1) if not vals.empty else None

    indicators = []
    cohort_vals = []
    model_vals = []

    # LBW
    c_lbw = _cohort_pct("lbw")
    m_lbw = snapshot_row.get("low_birthweight_pct")
    if c_lbw is not None and m_lbw is not None:
        indicators.append("Low birthweight (%)")
        cohort_vals.append(c_lbw)
        model_vals.append(float(m_lbw))

    # Anaemia in pregnancy (32wk)
    c_ana = _cohort_pct("anaemic_32wk")
    m_ana = snapshot_row.get("anaemia_pregnant_women_pct")
    if c_ana is not None and m_ana is not None:
        indicators.append("Anaemia in pregnancy (%)")
        cohort_vals.append(c_ana)
        model_vals.append(float(m_ana))

    # Iron deficiency (32wk)
    c_iron = _cohort_pct("iron_deficient_32wk")
    m_iron = snapshot_row.get("iron_deficiency_pct")
    if c_iron is not None and m_iron is not None:
        indicators.append("Iron deficiency (%)")
        cohort_vals.append(c_iron)
        model_vals.append(float(m_iron))

    # Stunting at birth vs <5 stunting
    c_stunt = _cohort_pct("stunted_at_birth")
    m_stunt = snapshot_row.get("stunting_pct_who")
    if c_stunt is not None and m_stunt is not None:
        indicators.append("Stunting (%)*")
        cohort_vals.append(c_stunt)
        model_vals.append(float(m_stunt))

    if not indicators:
        return _empty_figure(msg="No matching indicators to compare", height=height)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MUMTA cohort",
        y=indicators,
        x=cohort_vals,
        orientation="h",
        marker_color=ACCENT_ORANGE,
        hovertemplate="%{y}: %{x:.1f}%<extra>MUMTA cohort</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Modelled estimate (GBD/WHO)",
        y=indicators,
        x=model_vals,
        orientation="h",
        marker_color=FOUNDATION_BLUE,
        hovertemplate="%{y}: %{x:.1f}%<extra>Modelled estimate</extra>",
    ))

    fig.update_layout(
        **_base_layout(height=height),
        barmode="group",
        title=dict(text="MUMTA Cohort vs Modelled Estimates (Pakistan)", font=dict(size=15, color=FOUNDATION_BLUE)),
        xaxis=dict(title="Prevalence (%)", gridcolor="#eee"),
        yaxis=dict(title=""),
    )

    # Add footnote for stunting comparison caveat
    if "Stunting (%)*" in indicators:
        fig.add_annotation(
            text="*Cohort = stunted at birth; modelled = stunting in children <5",
            xref="paper", yref="paper", x=0, y=-0.15,
            showarrow=False, font=dict(size=9, color="#888"),
        )

    return fig


# ── Birth outcomes by maternal risk factor ──────────────────────────────────

# Risk-factor color palette
_RISK_COLORS = {
    # Anemia status
    "Anaemic": "#CC3333",
    "Not anaemic": "#2A9D8F",
    # BMI categories
    "Underweight": "#CC3333",
    "Normal": "#2A9D8F",
    "Overweight": "#E87722",
    "Obese": "#7C3AED",
    # MUAC
    "Malnourished (<23cm)": "#CC3333",
    "Adequate (≥23cm)": "#2A9D8F",
    # Iron deficiency
    "Iron deficient": "#CC3333",
    "Iron replete": "#2A9D8F",
}

_RISK_ORDER = list(_RISK_COLORS.keys())


def _wilson_ci(count, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0, 0, 0
    p = count / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return centre, max(0, centre - margin), min(1, centre + margin)


def birth_outcomes_by_risk_factor(cohort_df, risk_factor="anaemia"):
    """
    Grouped bar chart showing birth outcome rates stratified by a maternal risk factor.

    Parameters
    ----------
    cohort_df : DataFrame
        MUMTA cohort summary with columns: birth_outcome, lbw, preterm,
        stunted_at_birth, wasted_at_birth, and the relevant risk factor columns.
    risk_factor : str
        One of: 'anaemia', 'bmi', 'muac', 'iron_deficiency'

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if _is_empty(cohort_df):
        return _empty_figure("No cohort data available")

    # Filter to live births for birth outcome analysis
    live = cohort_df[cohort_df["birth_outcome"].str.contains("live", case=False, na=False)].copy()

    if live.empty:
        return _empty_figure("No live births in data")

    # Define strata based on risk factor
    if risk_factor == "anaemia":
        live["_stratum"] = live["anaemic_19wk"].map({True: "Anaemic", False: "Not anaemic"})
        title = "Birth Outcomes by Maternal Anaemia Status (19 weeks)"
    elif risk_factor == "bmi":
        live["_stratum"] = live["bmi_category"].astype(str)
        live = live[live["_stratum"] != "nan"]
        title = "Birth Outcomes by Maternal BMI Category (enrollment)"
    elif risk_factor == "muac":
        live["_stratum"] = live["muac_malnourished"].map(
            {True: "Malnourished (<23cm)", False: "Adequate (≥23cm)"}
        )
        title = "Birth Outcomes by Maternal MUAC Status (enrollment)"
    elif risk_factor == "iron_deficiency":
        live["_stratum"] = live["iron_deficient_19wk"].map(
            {True: "Iron deficient", False: "Iron replete"}
        )
        title = "Birth Outcomes by Iron Deficiency Status (19 weeks)"
    else:
        return _empty_figure(f"Unknown risk factor: {risk_factor}")

    live = live.dropna(subset=["_stratum"])

    outcomes = [
        ("lbw", "Low birthweight"),
        ("preterm", "Preterm"),
        ("stunted_at_birth", "Stunted at birth"),
        ("wasted_at_birth", "Wasted at birth"),
    ]

    strata = sorted(live["_stratum"].unique(),
                    key=lambda x: _RISK_ORDER.index(x) if x in _RISK_ORDER else 99)

    fig = go.Figure()

    for stratum in strata:
        sub = live[live["_stratum"] == stratum]
        n_group = len(sub)
        rates = []
        ci_lo = []
        ci_hi = []
        labels = []

        for col, label in outcomes:
            count = sub[col].sum()
            p, lo, hi = _wilson_ci(count, n_group)
            rates.append(p * 100)
            ci_lo.append(lo * 100)
            ci_hi.append(hi * 100)
            labels.append(label)

        color = _RISK_COLORS.get(stratum, "#999999")

        fig.add_trace(go.Bar(
            name=f"{stratum} (n={n_group})",
            x=labels,
            y=rates,
            marker_color=color,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[hi - r for r, hi in zip(rates, ci_hi)],
                arrayminus=[r - lo for r, lo in zip(rates, ci_lo)],
                color="#555",
                thickness=1.5,
            ),
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
            textfont=dict(size=11),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Prevalence (%)",
            showgrid=True,
            gridcolor="#EEEEEE",
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=11),
        ),
        height=420,
        margin=dict(l=60, r=40, t=80, b=40),
        font=FONT,
    )

    return fig


def birth_weight_distribution(cohort_df, risk_factor="anaemia"):
    """
    Overlaid histograms of birth weight stratified by a maternal risk factor.

    Parameters
    ----------
    cohort_df : DataFrame
        MUMTA cohort summary with birth_weight_g and risk factor columns.
    risk_factor : str
        One of: 'anaemia', 'bmi', 'muac', 'iron_deficiency'

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if _is_empty(cohort_df):
        return _empty_figure("No cohort data available")

    live = cohort_df[cohort_df["birth_outcome"].str.contains("live", case=False, na=False)].copy()
    live = live.dropna(subset=["birth_weight_g"])

    if live.empty:
        return _empty_figure("No birth weight data available")

    # Define strata
    if risk_factor == "anaemia":
        live["_stratum"] = live["anaemic_19wk"].map({True: "Anaemic", False: "Not anaemic"})
        title = "Birth Weight Distribution by Anaemia Status"
    elif risk_factor == "bmi":
        live["_stratum"] = live["bmi_category"].astype(str)
        live = live[live["_stratum"] != "nan"]
        title = "Birth Weight Distribution by BMI Category"
    elif risk_factor == "muac":
        live["_stratum"] = live["muac_malnourished"].map(
            {True: "Malnourished (<23cm)", False: "Adequate (≥23cm)"}
        )
        title = "Birth Weight Distribution by MUAC Status"
    elif risk_factor == "iron_deficiency":
        live["_stratum"] = live["iron_deficient_19wk"].map(
            {True: "Iron deficient", False: "Iron replete"}
        )
        title = "Birth Weight Distribution by Iron Deficiency"
    else:
        return _empty_figure(f"Unknown risk factor: {risk_factor}")

    live = live.dropna(subset=["_stratum"])

    strata = sorted(live["_stratum"].unique(),
                    key=lambda x: _RISK_ORDER.index(x) if x in _RISK_ORDER else 99)

    fig = go.Figure()

    for stratum in strata:
        sub = live[live["_stratum"] == stratum]
        color = _RISK_COLORS.get(stratum, "#999999")
        mean_bw = sub["birth_weight_g"].mean()

        fig.add_trace(go.Histogram(
            x=sub["birth_weight_g"],
            name=f"{stratum} (n={len(sub)}, mean={mean_bw:.0f}g)",
            marker_color=_hex_to_rgba(color, 0.6),
            marker_line=dict(color=color, width=1),
            nbinsx=30,
            opacity=0.7,
        ))

    # Add LBW threshold line
    fig.add_vline(
        x=2500, line_dash="dash", line_color="#CC3333", line_width=2,
        annotation_text="LBW threshold (2500g)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#CC3333"),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Birth weight (g)", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=11),
        ),
        height=400,
        margin=dict(l=60, r=40, t=80, b=40),
        font=FONT,
    )

    return fig


def adverse_outcome_summary(cohort_df):
    """
    Summary table: adverse birth outcome rates by all maternal risk factors.

    Returns a DataFrame with columns: Risk factor, Group, N, LBW (%), Preterm (%),
    Stunted (%), Wasted (%), Mean BW (g).
    """
    if _is_empty(cohort_df):
        return pd.DataFrame()

    live = cohort_df[cohort_df["birth_outcome"].str.contains("live", case=False, na=False)].copy()

    strata_defs = {
        "Anaemia (19wk)": ("anaemic_19wk", {True: "Anaemic", False: "Not anaemic"}),
        "Anaemia (32wk)": ("anaemic_32wk", {True: "Anaemic", False: "Not anaemic"}),
        "Iron deficiency (19wk)": ("iron_deficient_19wk", {True: "Iron deficient", False: "Iron replete"}),
        "BMI category": ("bmi_category", None),
        "MUAC status": ("muac_malnourished", {True: "Malnourished (<23cm)", False: "Adequate (≥23cm)"}),
    }

    rows = []
    for rf_label, (col, mapping) in strata_defs.items():
        if col not in live.columns:
            continue
        if mapping:
            live["_s"] = live[col].map(mapping)
        else:
            live["_s"] = live[col].astype(str)
            live = live[live["_s"] != "nan"]

        for stratum, sub in live.groupby("_s"):
            n = len(sub)
            if n < 10:
                continue
            rows.append({
                "Risk factor": rf_label,
                "Group": stratum,
                "N": n,
                "LBW (%)": f"{sub['lbw'].mean() * 100:.1f}",
                "Preterm (%)": f"{sub['preterm'].mean() * 100:.1f}",
                "Stunted (%)": f"{sub['stunted_at_birth'].mean() * 100:.1f}",
                "Wasted (%)": f"{sub['wasted_at_birth'].mean() * 100:.1f}",
                "Mean BW (g)": f"{sub['birth_weight_g'].mean():.0f}",
            })

    return pd.DataFrame(rows)


# ── Enteric pathogen & gut dysfunction visualizations ───────────────────────

# Category colors for pathogen groups
_CATEGORY_COLORS = {
    "Bacteria": "#003366",
    "Diarrheagenic E. coli": "#E87722",
    "Virus": "#2A9D8F",
    "Parasite": "#7C3AED",
    "Helminth": "#CC3333",
}


def pathogen_detection_heatmap(tac_df, specimen_type="all"):
    """
    Heatmap of pathogen detection rates (%) by timepoint.

    Rows = pathogens (sorted by peak detection rate), columns = timepoints.
    Color intensity = detection rate. Filters to maternal or infant specimens.

    Parameters
    ----------
    tac_df : DataFrame
        Long-format TAC data from mumta_tac_pathogens.csv.
    specimen_type : str
        'maternal', 'infant', or 'all'.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if _is_empty(tac_df):
        return _empty_figure("No TAC data available")

    df = tac_df.copy()
    if specimen_type != "all":
        df = df[df["specimen_type"] == specimen_type]

    if df.empty:
        return _empty_figure(f"No {specimen_type} TAC data")

    # Compute detection rate per pathogen per timepoint
    rates = (
        df.groupby(["pathogen_label", "timepoint"])["detected"]
        .mean()
        .unstack(fill_value=0)
        * 100
    )

    # Sort by max detection rate across any timepoint
    rates["_max"] = rates.max(axis=1)
    rates = rates.sort_values("_max", ascending=True).drop(columns="_max")

    # Filter to pathogens detected at least once at >5%
    rates = rates[rates.max(axis=1) > 5]

    if rates.empty:
        return _empty_figure("No pathogens detected above 5% threshold")

    # Ensure timepoint column order
    tp_order = [tp for tp in TIMEPOINT_ORDER if tp in rates.columns]
    rates = rates[tp_order]

    # Build heatmap
    # Create labels combining timepoint + specimen type
    if specimen_type == "all":
        col_labels = tp_order
    else:
        col_labels = tp_order

    fig = go.Figure(data=go.Heatmap(
        z=rates.values,
        x=col_labels,
        y=rates.index.tolist(),
        colorscale=[
            [0.0, "#F8F9FA"],
            [0.15, "#FFF3E0"],
            [0.3, "#FFB74D"],
            [0.5, "#E87722"],
            [0.7, "#CC3333"],
            [1.0, "#7B1FA2"],
        ],
        colorbar=dict(title=dict(text="Detection<br>rate (%)", font=dict(size=11))),
        text=[[f"{v:.0f}%" if v > 0 else "" for v in row] for row in rates.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
    ))

    specimen_label = specimen_type.capitalize() if specimen_type != "all" else "All"
    fig.update_layout(
        title=dict(
            text=f"Enteric Pathogen Detection — {specimen_label} Specimens",
            font=dict(size=15),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Timepoint", side="bottom"),
        yaxis=dict(title="", automargin=True),
        height=max(350, len(rates) * 22 + 120),
        margin=dict(l=160, r=80, t=60, b=60),
        font=FONT,
    )

    return fig


def pathogen_burden_trajectory(tac_df):
    """
    Line chart: mean number of pathogens detected per person over time,
    split by maternal vs infant specimens, colored by treatment arm.

    Parameters
    ----------
    tac_df : DataFrame
        Long-format TAC data.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if _is_empty(tac_df):
        return _empty_figure("No TAC data available")

    # Count unique pathogens detected per person per timepoint per specimen
    burden = (
        tac_df[tac_df["detected"]]
        .groupby(["study_id", "arm", "timepoint", "specimen_type"])
        .size()
        .reset_index(name="n_pathogens")
    )

    # Also need to include people with 0 detections
    all_combos = (
        tac_df[["study_id", "arm", "timepoint", "specimen_type"]]
        .drop_duplicates()
    )
    burden = all_combos.merge(burden, on=["study_id", "arm", "timepoint", "specimen_type"], how="left")
    burden["n_pathogens"] = burden["n_pathogens"].fillna(0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Maternal stool", "Infant stool"],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col_idx, specimen in enumerate(["maternal", "infant"], 1):
        sub = burden[burden["specimen_type"] == specimen]
        if sub.empty:
            continue

        for arm in ARM_ORDER:
            arm_data = sub[sub["arm"] == arm]
            if arm_data.empty:
                continue

            # Mean and SE per timepoint
            stats = (
                arm_data.groupby("timepoint")["n_pathogens"]
                .agg(["mean", "sem", "count"])
                .reindex(TIMEPOINT_ORDER)
                .dropna(subset=["mean"])
            )

            fig.add_trace(
                go.Scatter(
                    x=stats.index.tolist(),
                    y=stats["mean"],
                    mode="lines+markers",
                    name=ARM_LABELS.get(arm, arm),
                    line=dict(color=ARM_COLORS.get(arm, "#999"), width=2),
                    marker=dict(size=6),
                    error_y=dict(
                        type="data",
                        array=stats["sem"].tolist(),
                        visible=True,
                        thickness=1,
                    ),
                    legendgroup=arm,
                    showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title=dict(text="Enteric Pathogen Burden Over Time by Arm", font=dict(size=15)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, font=dict(size=10)),
        font=FONT,
    )
    fig.update_yaxes(title_text="Mean pathogens detected", row=1, col=1,
                     showgrid=True, gridcolor="#EEEEEE", rangemode="tozero")
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", rangemode="tozero", row=1, col=2)
    fig.update_xaxes(showgrid=False)

    return fig


def gut_inflammation_vs_growth(inflammation_df, growth_df):
    """
    Scatter: gut inflammation (MPO) vs concurrent infant growth (LAZ),
    colored by treatment arm. Shows the EED-growth relationship.

    Parameters
    ----------
    inflammation_df : DataFrame
        Long-format gut inflammation data with mpo, timepoint, specimen_type.
    growth_df : DataFrame
        Long-format infant growth data with laz, month.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if _is_empty(inflammation_df) or _is_empty(growth_df):
        return _empty_figure("Insufficient data for inflammation–growth analysis")

    # Map inflammation timepoints to growth months for matching
    tp_to_month = {
        "1-2mo": 1,
        "3-4mo": 3,
        "5-6mo": 5,
        "12mo": 8,  # closest available growth month
    }

    # Get infant MPO only
    inf_mpo = inflammation_df[
        (inflammation_df["specimen_type"] == "infant") &
        (inflammation_df["mpo"].notna())
    ].copy()
    inf_mpo["month"] = inf_mpo["timepoint"].map(tp_to_month)
    inf_mpo = inf_mpo.dropna(subset=["month"])
    inf_mpo["month"] = inf_mpo["month"].astype(int)

    # Merge with growth data
    merged = inf_mpo.merge(
        growth_df[["study_id", "arm", "month", "laz"]].dropna(subset=["laz"]),
        on=["study_id", "month"],
        how="inner",
        suffixes=("", "_growth"),
    )

    if merged.empty or len(merged) < 20:
        return _empty_figure("Insufficient paired inflammation–growth data")

    # Log-transform MPO for visualization
    merged["log_mpo"] = np.log10(merged["mpo"].clip(lower=1))

    fig = go.Figure()

    for arm in ARM_ORDER:
        sub = merged[merged["arm"] == arm]
        if sub.empty:
            continue

        fig.add_trace(go.Scatter(
            x=sub["log_mpo"],
            y=sub["laz"],
            mode="markers",
            name=ARM_LABELS.get(arm, arm),
            marker=dict(
                color=_hex_to_rgba(ARM_COLORS.get(arm, "#999"), 0.6),
                size=6,
                line=dict(color=ARM_COLORS.get(arm, "#999"), width=0.5),
            ),
            hovertemplate=(
                f"<b>{ARM_LABELS.get(arm, arm)}</b><br>"
                "MPO: %{customdata[0]:.0f} ng/mL<br>"
                "LAZ: %{y:.2f}<br>"
                "Month: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.column_stack([sub["mpo"], sub["month"]]),
        ))

    # Add stunting threshold
    fig.add_hline(y=-2, line_dash="dash", line_color="#CC3333", line_width=1.5,
                  annotation_text="Stunting threshold (LAZ < -2)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#CC3333"))

    # Correlation annotation
    valid = merged[["log_mpo", "laz"]].dropna()
    if len(valid) > 10:
        try:
            from scipy import stats as sp_stats
            r, p = sp_stats.spearmanr(valid["log_mpo"], valid["laz"])
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            fig.add_annotation(
                text=f"Spearman r = {r:.2f}, {p_str} (n = {len(valid)})",
                xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=11, color="#333"),
                bgcolor="rgba(255,255,255,0.8)",
            )
        except ImportError:
            # Scipy not available — show n only
            fig.add_annotation(
                text=f"n = {len(valid)} paired observations",
                xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=11, color="#333"),
                bgcolor="rgba(255,255,255,0.8)",
            )

    fig.update_layout(
        title=dict(text="Gut Inflammation (MPO) vs. Infant Growth (LAZ)", font=dict(size=15)),
        xaxis=dict(title="log₁₀(MPO ng/mL)", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="Length-for-age Z-score (LAZ)", showgrid=True, gridcolor="#EEEEEE"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        height=450,
        margin=dict(l=60, r=40, t=80, b=60),
        font=FONT,
    )

    return fig


def top_pathogens_by_timepoint(tac_df, n_top=10):
    """
    Grouped bar chart: top N pathogens by detection rate at each timepoint,
    comparing maternal vs infant specimens.

    Parameters
    ----------
    tac_df : DataFrame
        Long-format TAC data.
    n_top : int
        Number of top pathogens to show.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if _is_empty(tac_df):
        return _empty_figure("No TAC data available")

    # Get overall top pathogens by max detection rate across any timepoint
    overall_rates = tac_df.groupby("pathogen_label")["detected"].mean() * 100
    top = overall_rates.nlargest(n_top).index.tolist()

    df = tac_df[tac_df["pathogen_label"].isin(top)].copy()

    # Split into maternal and infant timepoints
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Maternal stool", "Infant stool"],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col_idx, specimen in enumerate(["maternal", "infant"], 1):
        sub = df[df["specimen_type"] == specimen]
        if sub.empty:
            continue

        tp_order_sub = [tp for tp in TIMEPOINT_ORDER
                        if tp in sub["timepoint"].unique()]

        rates_pivot = (
            sub.groupby(["pathogen_label", "timepoint"])["detected"]
            .mean()
            .unstack(fill_value=0) * 100
        )
        rates_pivot = rates_pivot.reindex(columns=tp_order_sub, fill_value=0)
        # Sort by max rate
        rates_pivot = rates_pivot.loc[rates_pivot.max(axis=1).sort_values(ascending=False).index]

        tp_colors = {
            "19wk": "#003366", "32wk": "#336699",
            "1-2mo": "#E87722", "3-4mo": "#CC3333",
            "5-6mo": "#7C3AED", "12mo": "#2A9D8F",
        }

        for tp in tp_order_sub:
            fig.add_trace(
                go.Bar(
                    x=rates_pivot[tp].values,
                    y=rates_pivot.index.tolist(),
                    orientation="h",
                    name=tp,
                    marker_color=tp_colors.get(tp, "#999"),
                    legendgroup=tp,
                    showlegend=(col_idx == 1),
                    text=[f"{v:.0f}%" if v > 3 else "" for v in rates_pivot[tp].values],
                    textposition="outside",
                    textfont=dict(size=9),
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        title=dict(text=f"Top {n_top} Pathogen Detection Rates by Timepoint", font=dict(size=15)),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(400, n_top * 35 + 120),
        margin=dict(l=140, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, font=dict(size=10)),
        font=FONT,
    )
    fig.update_xaxes(title_text="Detection rate (%)", showgrid=True, gridcolor="#EEEEEE",
                     rangemode="tozero")
    fig.update_yaxes(automargin=True)

    return fig


# ── B. infantis deep dive ───────────────────────────────────────────────────

_BINF_POS_COLOR = "#2A9D8F"
_BINF_NEG_COLOR = "#CC3333"


def binfantis_colonization_corrected(binfantis_df):
    """
    B. infantis colonization trajectory — CORRECTED to show rates among
    tested participants only, not the full cohort.

    Shows colonization rates over time for maternal and infant specimens,
    with proper denominator (tested) and sample size annotations.

    Parameters
    ----------
    binfantis_df : DataFrame
        Long-format B. infantis qPCR data with 'tested' column.
    """
    if _is_empty(binfantis_df):
        return _empty_figure("No B. infantis data available")

    df = binfantis_df.copy()

    # Only include tested rows
    if "tested" not in df.columns:
        return _empty_figure("B. infantis data missing 'tested' column — re-run process_mumta.py")

    df = df[df["tested"] == True]

    if df.empty:
        return _empty_figure("No tested B. infantis data")

    fig = go.Figure()

    for specimen, dash in [("maternal", "dot"), ("infant", "solid")]:
        sub = df[df["specimen_type"] == specimen]
        if sub.empty:
            continue

        rates = sub.groupby("timepoint").agg(
            n_tested=("b_infantis_positive", "count"),
            n_positive=("b_infantis_positive", "sum"),
        )
        # Reorder chronologically
        rates = rates.reindex([tp for tp in TIMEPOINT_ORDER if tp in rates.index])
        rates["rate"] = rates["n_positive"] / rates["n_tested"] * 100

        fig.add_trace(go.Scatter(
            x=rates.index.tolist(),
            y=rates["rate"],
            mode="lines+markers+text",
            name=f"{specimen.capitalize()} (n={int(rates['n_tested'].median())})",
            line=dict(
                color=_BINF_POS_COLOR if specimen == "infant" else FOUNDATION_BLUE,
                width=3 if specimen == "infant" else 2,
                dash=dash,
            ),
            marker=dict(size=8 if specimen == "infant" else 6),
            text=[f"{r:.0f}%<br>(n={n})" for r, n in zip(rates["rate"], rates["n_tested"])],
            textposition="top center",
            textfont=dict(size=9),
        ))

    fig.update_layout(
        title=dict(
            text="B. infantis Colonization Trajectory (among tested participants)",
            font=dict(size=15),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Timepoint"),
        yaxis=dict(
            title="Colonization rate (%)",
            showgrid=True, gridcolor="#EEEEEE",
            range=[0, 105],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        height=420,
        margin=dict(l=60, r=40, t=80, b=40),
        font=FONT,
    )

    return fig


def binfantis_by_arm(binfantis_df):
    """
    B. infantis infant colonization rate by treatment arm at each timepoint.

    Parameters
    ----------
    binfantis_df : DataFrame
        Long-format B. infantis qPCR data with 'tested' column.
    """
    if _is_empty(binfantis_df):
        return _empty_figure("No B. infantis data available")

    df = binfantis_df.copy()
    if "tested" in df.columns:
        df = df[df["tested"] == True]

    # Infant only
    df = df[df["specimen_type"] == "infant"]
    if df.empty:
        return _empty_figure("No infant B. infantis data")

    fig = go.Figure()

    for arm in ARM_ORDER:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue

        rates = sub.groupby("timepoint").agg(
            n_tested=("b_infantis_positive", "count"),
            n_positive=("b_infantis_positive", "sum"),
        )
        rates = rates.reindex([tp for tp in TIMEPOINT_ORDER if tp in rates.index])
        rates = rates[rates["n_tested"] >= 5]  # Require minimum sample
        rates["rate"] = rates["n_positive"] / rates["n_tested"] * 100

        fig.add_trace(go.Scatter(
            x=rates.index.tolist(),
            y=rates["rate"],
            mode="lines+markers",
            name=f"{ARM_LABELS.get(arm, arm)} (n≈{int(rates['n_tested'].median())})",
            line=dict(color=ARM_COLORS.get(arm, "#999"), width=2.5),
            marker=dict(size=7),
            hovertemplate=(
                f"<b>{ARM_LABELS.get(arm, arm)}</b><br>"
                "%{x}: %{y:.0f}%<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Infant B. infantis Colonization by Treatment Arm", font=dict(size=15)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Timepoint"),
        yaxis=dict(
            title="Colonization rate (%)",
            showgrid=True, gridcolor="#EEEEEE",
            range=[0, 105],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        height=400,
        margin=dict(l=60, r=40, t=80, b=40),
        font=FONT,
    )

    return fig


def binfantis_vs_pathogens(binfantis_df, tac_df):
    """
    Compare pathogen detection rates in B. infantis-positive vs -negative infants.

    Grouped bar chart: for each pathogen with >5% detection, show the detection
    rate in B. infantis+ vs B. infantis- infants (pooled across timepoints).

    Parameters
    ----------
    binfantis_df : DataFrame
        Long-format B. infantis qPCR data with 'tested' column.
    tac_df : DataFrame
        Long-format TAC pathogen data.
    """
    if _is_empty(binfantis_df) or _is_empty(tac_df):
        return _empty_figure("Insufficient data")

    # Get infant B. infantis status (tested only)
    binf = binfantis_df.copy()
    if "tested" in binf.columns:
        binf = binf[binf["tested"] == True]
    binf = binf[binf["specimen_type"] == "infant"]
    binf = binf[binf["b_infantis_positive"].notna()]

    # Get infant TAC data
    tac = tac_df[tac_df["specimen_type"] == "infant"].copy()

    # Merge on study_id + timepoint
    merged = tac.merge(
        binf[["study_id", "timepoint", "b_infantis_positive"]],
        on=["study_id", "timepoint"],
        how="inner",
    )

    if merged.empty or len(merged) < 50:
        return _empty_figure(f"Insufficient overlap (n={len(merged)})")

    # Compute detection rate per pathogen, split by B. infantis status
    results = []
    for pathogen_label, grp in merged.groupby("pathogen_label"):
        pos = grp[grp["b_infantis_positive"] == True]
        neg = grp[grp["b_infantis_positive"] == False]
        if len(pos) < 5 or len(neg) < 5:
            continue
        pos_rate = pos["detected"].mean() * 100
        neg_rate = neg["detected"].mean() * 100
        results.append({
            "pathogen": pathogen_label,
            "binf_pos_rate": pos_rate,
            "binf_neg_rate": neg_rate,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "diff": pos_rate - neg_rate,
        })

    if not results:
        return _empty_figure("No pathogens with sufficient data in both groups")

    res_df = pd.DataFrame(results).sort_values("diff")

    # Filter to pathogens with >5% detection in either group
    res_df = res_df[(res_df["binf_pos_rate"] > 5) | (res_df["binf_neg_rate"] > 5)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"B. infantis + (n≈{int(res_df['n_pos'].median())})",
        y=res_df["pathogen"],
        x=res_df["binf_pos_rate"],
        orientation="h",
        marker_color=_BINF_POS_COLOR,
        text=[f"{v:.0f}%" for v in res_df["binf_pos_rate"]],
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.add_trace(go.Bar(
        name=f"B. infantis − (n≈{int(res_df['n_neg'].median())})",
        y=res_df["pathogen"],
        x=res_df["binf_neg_rate"],
        orientation="h",
        marker_color=_BINF_NEG_COLOR,
        text=[f"{v:.0f}%" for v in res_df["binf_neg_rate"]],
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.update_layout(
        title=dict(
            text="Pathogen Detection: B. infantis+ vs B. infantis− Infants",
            font=dict(size=15),
        ),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Detection rate (%)",
            showgrid=True, gridcolor="#EEEEEE",
            rangemode="tozero",
        ),
        yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        height=max(400, len(res_df) * 28 + 120),
        margin=dict(l=160, r=60, t=80, b=40),
        font=FONT,
    )

    return fig


def binfantis_vs_inflammation(binfantis_df, inflammation_df):
    """
    Box plot comparing gut inflammation (MPO) in B. infantis+ vs − infants.

    Parameters
    ----------
    binfantis_df : DataFrame
        Long-format B. infantis qPCR data with 'tested' column.
    inflammation_df : DataFrame
        Long-format gut inflammation data.
    """
    if _is_empty(binfantis_df) or _is_empty(inflammation_df):
        return _empty_figure("Insufficient data")

    binf = binfantis_df.copy()
    if "tested" in binf.columns:
        binf = binf[binf["tested"] == True]
    binf = binf[binf["specimen_type"] == "infant"]
    binf = binf[binf["b_infantis_positive"].notna()]

    inflam = inflammation_df[
        (inflammation_df["specimen_type"] == "infant") &
        (inflammation_df["mpo"].notna())
    ]

    merged = binf.merge(
        inflam[["study_id", "timepoint", "mpo"]],
        on=["study_id", "timepoint"],
        how="inner",
    )

    if merged.empty or len(merged) < 20:
        return _empty_figure("Insufficient paired data")

    merged["log_mpo"] = np.log10(merged["mpo"].clip(lower=1))
    merged["status"] = merged["b_infantis_positive"].map(
        {True: "B. infantis +", False: "B. infantis −"}
    )

    fig = go.Figure()

    for status, color in [("B. infantis +", _BINF_POS_COLOR), ("B. infantis −", _BINF_NEG_COLOR)]:
        sub = merged[merged["status"] == status]
        if sub.empty:
            continue

        for tp in [t for t in TIMEPOINT_ORDER if t in sub["timepoint"].values]:
            tp_data = sub[sub["timepoint"] == tp]
            fig.add_trace(go.Box(
                x=[tp] * len(tp_data),
                y=tp_data["log_mpo"],
                name=status,
                legendgroup=status,
                showlegend=(tp == sub["timepoint"].unique()[0]),
                marker_color=color,
                boxmean=True,
                line=dict(color=color),
            ))

    fig.update_layout(
        title=dict(text="Gut Inflammation (MPO) by B. infantis Status", font=dict(size=15)),
        boxmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Timepoint"),
        yaxis=dict(
            title="log₁₀(MPO ng/mL)",
            showgrid=True, gridcolor="#EEEEEE",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        height=420,
        margin=dict(l=60, r=40, t=80, b=40),
        font=FONT,
    )

    return fig


def binfantis_vs_growth(binfantis_df, growth_df):
    """
    Compare infant growth (LAZ) trajectories by B. infantis status.

    Line chart: mean LAZ over time for B. infantis+ vs − infants.

    Parameters
    ----------
    binfantis_df : DataFrame
        Long-format B. infantis qPCR data with 'tested' column.
    growth_df : DataFrame
        Long-format infant growth data.
    """
    if _is_empty(binfantis_df) or _is_empty(growth_df):
        return _empty_figure("Insufficient data")

    binf = binfantis_df.copy()
    if "tested" in binf.columns:
        binf = binf[binf["tested"] == True]
    binf = binf[binf["specimen_type"] == "infant"]
    binf = binf[binf["b_infantis_positive"].notna()]

    # Map B. infantis timepoints to growth months
    tp_to_months = {
        "1-2mo": [1, 2],
        "3-4mo": [3, 4],
        "5-6mo": [5, 6],
    }

    records = []
    for tp, months in tp_to_months.items():
        binf_tp = binf[binf["timepoint"] == tp][["study_id", "b_infantis_positive"]]
        for m in months:
            growth_m = growth_df[(growth_df["month"] == m) & growth_df["laz"].notna()]
            merged = growth_m.merge(binf_tp, on="study_id", how="inner")
            if not merged.empty:
                merged["_month"] = m
                records.append(merged[["study_id", "_month", "laz", "waz", "b_infantis_positive"]])

    if not records:
        return _empty_figure("No paired B. infantis + growth data")

    all_data = pd.concat(records, ignore_index=True)
    all_data["status"] = all_data["b_infantis_positive"].map(
        {True: "B. infantis +", False: "B. infantis −"}
    )

    fig = go.Figure()

    for status, color, dash in [
        ("B. infantis +", _BINF_POS_COLOR, "solid"),
        ("B. infantis −", _BINF_NEG_COLOR, "dash"),
    ]:
        sub = all_data[all_data["status"] == status]
        if sub.empty:
            continue

        stats = sub.groupby("_month")["laz"].agg(["mean", "sem", "count"]).reset_index()
        stats = stats[stats["count"] >= 5]

        fig.add_trace(go.Scatter(
            x=stats["_month"],
            y=stats["mean"],
            mode="lines+markers",
            name=f"{status} (n≈{int(stats['count'].median())})",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=7),
            error_y=dict(
                type="data",
                array=stats["sem"].tolist(),
                visible=True,
                thickness=1.5,
            ),
        ))

    # Stunting threshold
    fig.add_hline(y=-2, line_dash="dot", line_color="#999", line_width=1,
                  annotation_text="Stunting (LAZ < −2)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#999"))

    fig.update_layout(
        title=dict(text="Infant Growth (LAZ) by B. infantis Status", font=dict(size=15)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Age (months)", showgrid=True, gridcolor="#EEEEEE", dtick=1),
        yaxis=dict(title="Length-for-age Z-score (LAZ)", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        height=400,
        margin=dict(l=60, r=40, t=80, b=40),
        font=FONT,
    )

    return fig
