"""
Visualization functions for the AMANHI cohort data.

AMANHI is a multi-country observational cohort (Pakistan, Bangladesh, Tanzania)
studying the maternal gut microbiome and neonatal outcomes.

Tier 1 views:
  - B. infantis cross-cohort comparison (AMANHI vs MUMTA)
  - Bioanalytes: CRP and ferritin vs birth outcomes
  - Maternal enteropathogens (TAC) — comparison with MUMTA

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

SITE_COLORS = {
    "PAK": "#2A9D8F",   # teal
    "BGD": "#E87722",   # orange
    "TZA": "#7C3AED",   # purple
}
SITE_LABELS = {
    "PAK": "Pakistan",
    "BGD": "Bangladesh",
    "TZA": "Tanzania",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_figure(msg="No data available", height=300):
    """Return a blank figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#999"),
    )
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False), font=FONT,
    )
    return fig


def _wilson_ci(n_pos, n_total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0, 0, 0
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return centre, max(0, centre - margin), min(1, centre + margin)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: B. infantis cross-country and cross-cohort
# ═════════════════════════════════════════════════════════════════════════════

def binfantis_by_site(neonatal_df):
    """Bar chart of B. infantis colonization rate by country with Wilson CIs."""
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df.dropna(subset=["binfantis_positive"])
    sites = ["PAK", "BGD", "TZA"]
    rates, lowers, uppers, labels, colors, ns = [], [], [], [], [], []

    for site in sites:
        sub = df[df["site"] == site]
        n_pos = sub["binfantis_positive"].sum()
        n_total = len(sub)
        rate, lo, hi = _wilson_ci(n_pos, n_total)
        rates.append(rate * 100)
        lowers.append((rate - lo) * 100)
        uppers.append((hi - rate) * 100)
        labels.append(f"{SITE_LABELS[site]}<br>(n={n_total})")
        colors.append(SITE_COLORS[site])
        ns.append(n_total)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rates,
        error_y=dict(type="data", symmetric=False, array=uppers, arrayminus=lowers),
        marker_color=colors,
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Colonization: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Neonatal B. infantis Colonization by Country (AMANHI)"),
        yaxis=dict(title="Colonization rate (%)", range=[0, 80]),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, font=FONT, showlegend=False,
    )
    return fig


def binfantis_vs_outcomes(neonatal_df):
    """Compare birth outcomes (LBW, preterm, SGA) by B. infantis status.

    Grouped bars: B.inf+ vs B.inf− for each outcome, Pakistan focus.
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    pak = neonatal_df[
        (neonatal_df["site"] == "PAK") &
        neonatal_df["binfantis_positive"].notna()
    ].copy()

    if len(pak) < 10:
        return _empty_figure("Insufficient Pakistan data")

    outcomes = [
        ("lbw", "Low birth weight"),
        ("preterm", "Preterm"),
        ("sga", "SGA"),
    ]

    fig = go.Figure()
    for status, color, label in [
        (True, "#2A9D8F", "B. infantis +"),
        (False, "#E76F51", "B. infantis −"),
    ]:
        grp = pak[pak["binfantis_positive"] == status]
        rates, los, his, xs = [], [], [], []
        for col, name in outcomes:
            n_pos = grp[col].sum()
            n_tot = grp[col].notna().sum()
            rate, lo, hi = _wilson_ci(int(n_pos), int(n_tot))
            rates.append(rate * 100)
            los.append((rate - lo) * 100)
            his.append((hi - rate) * 100)
            xs.append(name)

        fig.add_trace(go.Bar(
            name=f"{label} (n={len(grp)})",
            x=xs, y=rates,
            error_y=dict(type="data", symmetric=False, array=his, arrayminus=los),
            marker_color=color,
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(text="Birth Outcomes by B. infantis Status — Pakistan (AMANHI)"),
        yaxis=dict(title="Prevalence (%)", range=[0, 70]),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def binfantis_vs_growth(neonatal_df):
    """Box plots of HAZ at birth and 6 months by B. infantis status (Pakistan)."""
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    pak = neonatal_df[
        (neonatal_df["site"] == "PAK") &
        neonatal_df["binfantis_positive"].notna()
    ].copy()

    if len(pak) < 10:
        return _empty_figure("Insufficient Pakistan data")

    fig = make_subplots(rows=1, cols=2, subplot_titles=["HAZ at Birth", "HAZ at 6 Months"])

    for col_idx, (col, title) in enumerate([("haz1", "Birth"), ("haz6", "6 Months")], 1):
        for status, color, label in [
            (True, "#2A9D8F", "B.inf +"),
            (False, "#E76F51", "B.inf −"),
        ]:
            vals = pak.loc[pak["binfantis_positive"] == status, col].dropna()
            fig.add_trace(go.Box(
                y=vals, name=label,
                marker_color=color, boxmean=True,
                legendgroup=label, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="Growth (HAZ) by B. infantis Status — Pakistan (AMANHI)"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="HAZ (z-score)", row=1, col=1)
    fig.update_yaxes(title_text="HAZ (z-score)", row=1, col=2)
    return fig


def binfantis_cross_cohort(neonatal_df, mumta_binfantis_df=None):
    """Compare B. infantis colonization rates between AMANHI and MUMTA (Pakistan).

    AMANHI: single neonatal timepoint
    MUMTA: longitudinal (multiple timepoints)
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No AMANHI data")

    # AMANHI Pakistan rate
    pak = neonatal_df[
        (neonatal_df["site"] == "PAK") &
        neonatal_df["binfantis_positive"].notna()
    ]
    amanhi_pos = int(pak["binfantis_positive"].sum())
    amanhi_n = len(pak)
    amanhi_rate, amanhi_lo, amanhi_hi = _wilson_ci(amanhi_pos, amanhi_n)

    fig = go.Figure()

    # AMANHI bar
    fig.add_trace(go.Bar(
        x=["AMANHI<br>(neonatal)"],
        y=[amanhi_rate * 100],
        error_y=dict(
            type="data", symmetric=False,
            array=[(amanhi_hi - amanhi_rate) * 100],
            arrayminus=[(amanhi_rate - amanhi_lo) * 100],
        ),
        marker_color="#264653",
        text=[f"{amanhi_rate*100:.0f}%"],
        textposition="outside",
        name=f"AMANHI (n={amanhi_n})",
        width=0.4,
    ))

    # MUMTA bars by timepoint (if available)
    if mumta_binfantis_df is not None and not mumta_binfantis_df.empty:
        mb = mumta_binfantis_df.copy()
        # Only infant specimens
        if "specimen_type" in mb.columns:
            mb = mb[mb["specimen_type"] == "infant"]
        # Only tested
        if "tested" in mb.columns:
            mb = mb[mb["tested"] == True]  # noqa: E712

        tp_order = ["1-2mo", "3-4mo", "5-6mo"]
        for tp in tp_order:
            sub = mb[mb["timepoint"] == tp]
            if len(sub) == 0:
                continue
            if "binfantis_positive" in sub.columns:
                pos_col = "binfantis_positive"
            elif "positive" in sub.columns:
                pos_col = "positive"
            else:
                continue
            n_pos = int(sub[pos_col].sum())
            n_tot = len(sub)
            rate, lo, hi = _wilson_ci(n_pos, n_tot)
            fig.add_trace(go.Bar(
                x=[f"MUMTA<br>({tp})"],
                y=[rate * 100],
                error_y=dict(
                    type="data", symmetric=False,
                    array=[(hi - rate) * 100],
                    arrayminus=[(rate - lo) * 100],
                ),
                marker_color="#2A9D8F",
                text=[f"{rate*100:.0f}%"],
                textposition="outside",
                name=f"MUMTA {tp} (n={n_tot})",
                width=0.4,
            ))

    fig.update_layout(
        title=dict(text="B. infantis Colonization: AMANHI vs MUMTA (Pakistan)"),
        yaxis=dict(title="Colonization rate (%)", range=[0, 110]),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Bioanalytes — CRP and ferritin vs outcomes
# ═════════════════════════════════════════════════════════════════════════════

def crp_distribution(bioanalytes_df):
    """Histogram of maternal CRP with elevated threshold line."""
    if bioanalytes_df is None or bioanalytes_df.empty:
        return _empty_figure("No bioanalytes data")

    crp = bioanalytes_df["crp_mg_dl"].dropna()
    # Cap at 99th percentile for display
    cap = crp.quantile(0.99)
    crp_display = crp[crp <= cap]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=crp_display, nbinsx=50,
        marker_color="#2A9D8F", opacity=0.8,
        name="CRP distribution",
    ))
    fig.add_vline(x=1.9, line_dash="dash", line_color="#E76F51", line_width=2,
                  annotation_text="Elevated (>1.9 mg/dL, WHO/BRINDA 2020)", annotation_position="top right")

    n_total = len(crp)
    n_elevated = (crp > 1.9).sum()
    fig.update_layout(
        title=dict(text=f"Maternal CRP Distribution — Pakistan (n={n_total}, {n_elevated/n_total:.0%} elevated)"),
        xaxis=dict(title="CRP (mg/dL)"),
        yaxis=dict(title="Count"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, font=FONT, showlegend=False,
    )
    return fig


def ferritin_distribution(bioanalytes_df):
    """Histogram of maternal ferritin with iron deficiency threshold."""
    if bioanalytes_df is None or bioanalytes_df.empty:
        return _empty_figure("No bioanalytes data")

    fer = bioanalytes_df["ferritin_ng_ml"].dropna()
    cap = fer.quantile(0.99)
    fer_display = fer[fer <= cap]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fer_display, nbinsx=50,
        marker_color="#E87722", opacity=0.8,
        name="Ferritin distribution",
    ))
    fig.add_vline(x=30, line_dash="dash", line_color="#E76F51", line_width=2,
                  annotation_text="Iron deficient at delivery (<30 ng/mL)", annotation_position="top right")

    n_total = len(fer)
    n_deficient = (fer < 30).sum()
    fig.update_layout(
        title=dict(text=f"Maternal Ferritin — Pakistan (n={n_total}, {n_deficient/n_total:.0%} iron deficient at delivery)"),
        xaxis=dict(title="Ferritin (ng/mL)"),
        yaxis=dict(title="Count"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, font=FONT, showlegend=False,
    )
    return fig


def crp_vs_birth_outcomes(bioanalytes_df, neonatal_df):
    """Compare birth outcomes (LBW, preterm, SGA) by CRP status.

    Links bioanalytes to neonatal outcomes via participant_id.
    """
    if bioanalytes_df is None or neonatal_df is None:
        return _empty_figure("Missing data for CRP vs outcomes")

    # Link on participant_id
    pak_neo = neonatal_df[neonatal_df["site"] == "PAK"].copy()
    merged = pak_neo.merge(
        bioanalytes_df[["participant_id", "crp_mg_dl", "crp_elevated"]],
        on="participant_id", how="inner",
    )
    if len(merged) < 10:
        return _empty_figure(f"Only {len(merged)} linked records — insufficient")

    outcomes = [("lbw", "Low Birth Weight"), ("preterm", "Preterm"), ("sga", "SGA")]
    fig = go.Figure()

    for elevated, color, label in [
        (False, "#2A9D8F", "CRP Normal (≤1.9 mg/dL)"),
        (True, "#E76F51", "CRP Elevated (>1.9 mg/dL)"),
    ]:
        grp = merged[merged["crp_elevated"] == elevated]
        rates, los, his, xs = [], [], [], []
        for col, name in outcomes:
            n_pos = grp[col].sum()
            n_tot = grp[col].notna().sum()
            rate, lo, hi = _wilson_ci(int(n_pos), int(n_tot))
            rates.append(rate * 100)
            los.append((rate - lo) * 100)
            his.append((hi - rate) * 100)
            xs.append(name)

        fig.add_trace(go.Bar(
            name=f"{label} (n={len(grp)})",
            x=xs, y=rates,
            error_y=dict(type="data", symmetric=False, array=his, arrayminus=los),
            marker_color=color,
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(text="Birth Outcomes by Maternal CRP Status — Pakistan (AMANHI)"),
        yaxis=dict(title="Prevalence (%)", range=[0, 70]),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def ferritin_vs_birth_outcomes(bioanalytes_df, neonatal_df):
    """Compare birth outcomes by iron deficiency status."""
    if bioanalytes_df is None or neonatal_df is None:
        return _empty_figure("Missing data for ferritin vs outcomes")

    pak_neo = neonatal_df[neonatal_df["site"] == "PAK"].copy()
    merged = pak_neo.merge(
        bioanalytes_df[["participant_id", "ferritin_ng_ml", "iron_deficient"]],
        on="participant_id", how="inner",
    )
    if len(merged) < 10:
        return _empty_figure(f"Only {len(merged)} linked records — insufficient")

    outcomes = [("lbw", "Low Birth Weight"), ("preterm", "Preterm"), ("sga", "SGA")]
    fig = go.Figure()

    for deficient, color, label in [
        (False, "#2A9D8F", "Iron Replete (≥30 ng/mL)"),
        (True, "#E76F51", "Iron Deficient (<30 ng/mL)"),
    ]:
        grp = merged[merged["iron_deficient"] == deficient]
        rates, los, his, xs = [], [], [], []
        for col, name in outcomes:
            n_pos = grp[col].sum()
            n_tot = grp[col].notna().sum()
            rate, lo, hi = _wilson_ci(int(n_pos), int(n_tot))
            rates.append(rate * 100)
            los.append((rate - lo) * 100)
            his.append((hi - rate) * 100)
            xs.append(name)

        fig.add_trace(go.Bar(
            name=f"{label} (n={len(grp)})",
            x=xs, y=rates,
            error_y=dict(type="data", symmetric=False, array=his, arrayminus=los),
            marker_color=color,
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(text="Birth Outcomes by Iron Status — Pakistan (AMANHI)"),
        yaxis=dict(title="Prevalence (%)", range=[0, 70]),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def crp_ferritin_scatter(bioanalytes_df):
    """Scatter plot of CRP vs ferritin with iron deficiency and inflammation zones."""
    if bioanalytes_df is None or bioanalytes_df.empty:
        return _empty_figure("No bioanalytes data")

    df = bioanalytes_df.dropna(subset=["crp_mg_dl", "ferritin_ng_ml"]).copy()
    # Cap for display
    df = df[(df["crp_mg_dl"] <= df["crp_mg_dl"].quantile(0.99)) &
            (df["ferritin_ng_ml"] <= df["ferritin_ng_ml"].quantile(0.99))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["crp_mg_dl"], y=df["ferritin_ng_ml"],
        mode="markers",
        marker=dict(size=5, color="#2A9D8F", opacity=0.5),
        hovertemplate="CRP: %{x:.2f} mg/dL<br>Ferritin: %{y:.1f} ng/mL<extra></extra>",
    ))

    # Reference lines
    fig.add_hline(y=30, line_dash="dash", line_color="#E76F51", line_width=1.5,
                  annotation_text="Iron deficient at delivery (<30 ng/mL)", annotation_position="bottom right")
    fig.add_vline(x=1.9, line_dash="dash", line_color="#999", line_width=1.5,
                  annotation_text="CRP elevated (>1.9 mg/dL)", annotation_position="top left")

    # Quadrant labels
    fig.add_annotation(x=0.15, y=5, text="Iron deficient<br>No inflammation",
                       showarrow=False, font=dict(size=9, color="#666"))
    fig.add_annotation(x=2.0, y=5, text="Iron deficient +<br>Inflammation",
                       showarrow=False, font=dict(size=9, color="#E76F51"))

    fig.update_layout(
        title=dict(text=f"CRP vs Ferritin — Pakistan (n={len(df)})"),
        xaxis=dict(title="CRP (mg/dL)"),
        yaxis=dict(title="Ferritin (ng/mL)"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, font=FONT, showlegend=False,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Maternal TAC pathogens
# ═════════════════════════════════════════════════════════════════════════════

def tac_top_pathogens(tac_df, n_top=20):
    """Horizontal bar chart of top detected maternal pathogens (Pakistan)."""
    if tac_df is None or tac_df.empty:
        return _empty_figure("No TAC data")

    n_mothers = tac_df["whowid"].nunique()
    top = (
        tac_df[tac_df["detected"] == 1]
        .groupby(["pathogen", "pathogen_category"])
        .size()
        .reset_index(name="n_detected")
        .sort_values("n_detected", ascending=True)
        .tail(n_top)
    )
    top["pct"] = (top["n_detected"] / n_mothers * 100).round(1)

    cat_colors = {"bacteria": "#2A9D8F", "virus": "#E87722", "parasite": "#7C3AED"}
    colors = [cat_colors.get(c, "#888") for c in top["pathogen_category"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["pathogen"], x=top["pct"],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%" for p in top["pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Detected in %{x:.1f}% of mothers<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Top {n_top} Maternal Enteropathogens — Pakistan AMANHI (n={n_mothers})"),
        xaxis=dict(title="Detection rate (%)", range=[0, max(top["pct"]) + 15]),
        yaxis=dict(title=""),
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(400, n_top * 25), font=FONT, showlegend=False,
    )
    return fig


def tac_pathogen_burden_vs_outcomes(tac_df):
    """Box plots of total maternal pathogen burden by birth outcome."""
    if tac_df is None or tac_df.empty:
        return _empty_figure("No TAC data")

    # Get per-mother pathogen burden + outcomes
    mothers = tac_df.drop_duplicates(subset=["whowid"])
    if "total_pathogens" not in mothers.columns:
        return _empty_figure("No pathogen burden data")

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Preterm", "SGA", "Low Birth Weight"])

    outcomes = [
        ("preterm_new", "Preterm"),
        ("sga_bin", "SGA"),
    ]

    # Derive LBW from wt0
    mothers = mothers.copy()
    if "wt0" in mothers.columns:
        mothers["lbw"] = mothers["wt0"] < 2.5  # kg
        outcomes.append(("lbw", "LBW"))

    for col_idx, (col, label) in enumerate(outcomes, 1):
        if col not in mothers.columns:
            continue
        for val, color, name in [(0, "#2A9D8F", "No"), (1, "#E76F51", "Yes")]:
            sub = mothers[mothers[col] == val]["total_pathogens"].dropna()
            fig.add_trace(go.Box(
                y=sub, name=name,
                marker_color=color, boxmean=True,
                legendgroup=name, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="Maternal Pathogen Burden by Birth Outcome — Pakistan AMANHI"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, font=FONT,
    )
    for i in range(1, 4):
        fig.update_yaxes(title_text="# Pathogens detected", row=1, col=i)
    return fig


def tac_cross_cohort_comparison(amanhi_tac_df, mumta_tac_df=None):
    """Side-by-side pathogen detection rates: AMANHI maternal vs MUMTA maternal.

    Compares overlapping pathogens between the two Pakistan cohorts.
    """
    if amanhi_tac_df is None or amanhi_tac_df.empty:
        return _empty_figure("No AMANHI TAC data")

    n_amanhi = amanhi_tac_df["whowid"].nunique()

    # AMANHI rates
    amanhi_rates = (
        amanhi_tac_df[amanhi_tac_df["detected"] == 1]
        .groupby("pathogen").size()
        .div(n_amanhi)
        .mul(100)
        .rename("amanhi_pct")
    )

    if mumta_tac_df is None or mumta_tac_df.empty:
        # Just show AMANHI
        top = amanhi_rates.sort_values(ascending=True).tail(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top.index, x=top.values,
            orientation="h", marker_color="#264653",
            text=[f"{v:.0f}%" for v in top.values],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(text=f"Maternal Pathogen Detection — AMANHI Pakistan (n={n_amanhi})"),
            xaxis=dict(title="Detection rate (%)"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=450, font=FONT, showlegend=False,
        )
        return fig

    # MUMTA rates — maternal specimens only
    mumta_mat = mumta_tac_df[mumta_tac_df["specimen_type"] == "maternal"] if "specimen_type" in mumta_tac_df.columns else mumta_tac_df
    n_mumta = mumta_mat["study_id"].nunique() if "study_id" in mumta_mat.columns else mumta_mat.iloc[:, 0].nunique()

    mumta_rates = (
        mumta_mat[mumta_mat["detected"] == True]  # noqa: E712
        .groupby("pathogen").size()
        .div(n_mumta)
        .mul(100)
        .rename("mumta_pct")
    )

    # Find overlapping pathogens
    both = pd.DataFrame({"amanhi_pct": amanhi_rates, "mumta_pct": mumta_rates}).fillna(0)
    both["max_pct"] = both.max(axis=1)
    both = both.sort_values("max_pct", ascending=True).tail(15)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=both.index, x=both["amanhi_pct"],
        orientation="h", marker_color="#264653", name=f"AMANHI (n={n_amanhi})",
    ))
    fig.add_trace(go.Bar(
        y=both.index, x=both["mumta_pct"],
        orientation="h", marker_color="#2A9D8F", name=f"MUMTA (n={n_mumta})",
    ))

    fig.update_layout(
        title=dict(text="Maternal Pathogen Detection: AMANHI vs MUMTA (Pakistan)"),
        xaxis=dict(title="Detection rate (%)"),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        height=500, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 (Tier 2): B. infantis + B. longum vs growth trajectories
# ═════════════════════════════════════════════════════════════════════════════

def growth_trajectory_by_binfantis(neonatal_df):
    """Slopegraph: mean HAZ from birth → 6 months by B. infantis status, all 3 countries.

    Shows whether B. infantis-colonized neonates experience less growth faltering.
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df.dropna(subset=["binfantis_positive", "haz1", "haz6"]).copy()
    df["bi"] = df["binfantis_positive"].astype(bool)

    sites = ["PAK", "BGD", "TZA"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[SITE_LABELS.get(s, s) for s in sites],
        shared_yaxes=True,
    )

    for col_idx, site in enumerate(sites, 1):
        sub = df[df["site"] == site]
        for status, color, dash, label in [
            (True, "#2A9D8F", "solid", "B. infantis +"),
            (False, "#E76F51", "dash", "B. infantis −"),
        ]:
            grp = sub[sub["bi"] == status]
            n = len(grp)
            if n == 0:
                continue
            mean_birth = grp["haz1"].mean()
            mean_6mo = grp["haz6"].mean()
            delta = mean_6mo - mean_birth

            fig.add_trace(go.Scatter(
                x=["Birth", "6 months"],
                y=[mean_birth, mean_6mo],
                mode="lines+markers+text",
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=10, color=color),
                text=[f"{mean_birth:.2f}", f"{mean_6mo:.2f}"],
                textposition="top center",
                name=f"{label} (n={n}, Δ={delta:+.2f})",
                legendgroup=label,
                showlegend=(col_idx == 1),
                hovertemplate=(
                    f"<b>{label}</b> (n={n})<br>"
                    "%{x}: HAZ = %{y:.2f}<extra></extra>"
                ),
            ), row=1, col=col_idx)

        # Add stunting threshold
        fig.add_hline(y=-2, line_dash="dot", line_color="#ccc", line_width=1,
                      row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="Growth Trajectory (HAZ): Birth → 6 Months by B. infantis Status"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="HAZ (z-score)", row=1, col=1)
    for i in range(1, 4):
        fig.update_yaxes(range=[-2.5, 0.5], row=1, col=i)
    return fig


def growth_trajectory_by_blongum(neonatal_df):
    """Slopegraph: mean HAZ from birth → 6 months by B. longum status, all 3 countries."""
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df.dropna(subset=["blongum_positive", "haz1", "haz6"]).copy()
    df["bl"] = df["blongum_positive"].astype(bool)

    sites = ["PAK", "BGD", "TZA"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[SITE_LABELS.get(s, s) for s in sites],
        shared_yaxes=True,
    )

    for col_idx, site in enumerate(sites, 1):
        sub = df[df["site"] == site]
        for status, color, dash, label in [
            (True, "#264653", "solid", "B. longum +"),
            (False, "#E9C46A", "dash", "B. longum −"),
        ]:
            grp = sub[sub["bl"] == status]
            n = len(grp)
            if n == 0:
                continue
            mean_birth = grp["haz1"].mean()
            mean_6mo = grp["haz6"].mean()
            delta = mean_6mo - mean_birth

            fig.add_trace(go.Scatter(
                x=["Birth", "6 months"],
                y=[mean_birth, mean_6mo],
                mode="lines+markers+text",
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=10, color=color),
                text=[f"{mean_birth:.2f}", f"{mean_6mo:.2f}"],
                textposition="top center",
                name=f"{label} (n={n}, Δ={delta:+.2f})",
                legendgroup=label,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        fig.add_hline(y=-2, line_dash="dot", line_color="#ccc", line_width=1,
                      row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="Growth Trajectory (HAZ): Birth → 6 Months by B. longum Status"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="HAZ (z-score)", row=1, col=1)
    for i in range(1, 4):
        fig.update_yaxes(range=[-2.5, 0.5], row=1, col=i)
    return fig


def growth_by_colonization_group(neonatal_df, site="PAK"):
    """Box plots of growth faltering (ΔHAZ birth→6mo) by 4-group colonization status.

    Groups: Both (B.inf+ & B.long+), B.inf only, B.long only, Neither.
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df[neonatal_df["site"] == site].copy()
    df = df.dropna(subset=["binfantis_positive", "blongum_positive", "haz1", "haz6"])
    if len(df) < 10:
        return _empty_figure(f"Insufficient data for {site}")

    bi = df["binfantis_positive"].astype(bool)
    bl = df["blongum_positive"].astype(bool)
    df["colon_group"] = "Neither"
    df.loc[bi & bl, "colon_group"] = "Both B.inf + B.long"
    df.loc[bi & ~bl, "colon_group"] = "B. infantis only"
    df.loc[~bi & bl, "colon_group"] = "B. longum only"

    df["delta_haz"] = df["haz6"] - df["haz1"]
    df["delta_waz"] = df["waz6"] - df["waz1"]

    group_order = ["Both B.inf + B.long", "B. infantis only", "B. longum only", "Neither"]
    group_colors = {
        "Both B.inf + B.long": "#264653",
        "B. infantis only": "#2A9D8F",
        "B. longum only": "#E9C46A",
        "Neither": "#E76F51",
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ΔHAZ (birth → 6mo)", "ΔWAZ (birth → 6mo)"],
    )

    for col_idx, (metric, label) in enumerate([("delta_haz", "ΔHAZ"), ("delta_waz", "ΔWAZ")], 1):
        for grp in group_order:
            vals = df.loc[df["colon_group"] == grp, metric].dropna()
            fig.add_trace(go.Box(
                y=vals,
                name=f"{grp} (n={len(vals)})",
                marker_color=group_colors[grp],
                boxmean=True,
                legendgroup=grp,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1, row=1, col=2)

    site_label = SITE_LABELS.get(site, site)
    fig.update_layout(
        title=dict(text=f"Growth Faltering by Colonization Group — {site_label} (AMANHI)"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Change in z-score", row=1, col=1)
    fig.update_yaxes(title_text="Change in z-score", row=1, col=2)
    return fig


def growth_faltering_cross_country(neonatal_df):
    """Grouped bar chart: mean ΔHAZ by B. infantis status across all 3 countries.

    Shows whether the B. infantis–growth relationship is consistent or site-specific.
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df.dropna(subset=["binfantis_positive", "haz1", "haz6"]).copy()
    df["bi"] = df["binfantis_positive"].astype(bool)
    df["delta_haz"] = df["haz6"] - df["haz1"]

    sites = ["PAK", "BGD", "TZA"]
    fig = go.Figure()

    for status, color, label in [
        (True, "#2A9D8F", "B. infantis +"),
        (False, "#E76F51", "B. infantis −"),
    ]:
        means, sems, labels_x, ns = [], [], [], []
        for site in sites:
            grp = df[(df["site"] == site) & (df["bi"] == status)]
            n = len(grp)
            if n == 0:
                continue
            d = grp["delta_haz"]
            means.append(d.mean())
            sems.append(d.std() / np.sqrt(n))
            labels_x.append(f"{SITE_LABELS[site]}<br>(n={n})")
            ns.append(n)

        fig.add_trace(go.Bar(
            x=labels_x, y=means,
            error_y=dict(type="data", array=sems, visible=True),
            marker_color=color,
            name=label,
            text=[f"{m:+.2f}" for m in means],
            textposition="outside",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1)

    fig.update_layout(
        title=dict(text="Growth Faltering (ΔHAZ birth→6mo) by B. infantis Status"),
        yaxis=dict(title="Mean ΔHAZ ± SE", range=[-1.3, 0.3]),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def binfantis_dose_response(neonatal_df, site="PAK"):
    """Scatter: B. infantis Ct value (bacterial load proxy) vs HAZ at 6 months.

    Among B.inf+ neonates only. Lower Ct = higher bacterial load.
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df[
        (neonatal_df["site"] == site) &
        (neonatal_df["binfantis_positive"] == True)  # noqa: E712
    ].dropna(subset=["binfantis_ct", "haz6"]).copy()

    if len(df) < 10:
        return _empty_figure(f"Insufficient B.inf+ data for {SITE_LABELS.get(site, site)}")

    # Compute correlation
    corr_text = f"n = {len(df)}"
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(df["binfantis_ct"], df["haz6"])
        corr_text = f"n = {len(df)}, Spearman r = {r:.3f}, p = {p:.3f}"
    except ImportError:
        pass

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["binfantis_ct"], y=df["haz6"],
        mode="markers",
        marker=dict(size=7, color="#2A9D8F", opacity=0.6),
        hovertemplate="Ct: %{x:.1f}<br>HAZ 6mo: %{y:.2f}<extra></extra>",
    ))

    # Add trend line
    z = np.polyfit(df["binfantis_ct"], df["haz6"], 1)
    x_range = np.linspace(df["binfantis_ct"].min(), df["binfantis_ct"].max(), 50)
    fig.add_trace(go.Scatter(
        x=x_range, y=np.polyval(z, x_range),
        mode="lines", line=dict(color="#E76F51", dash="dash", width=2),
        name="Trend", showlegend=False,
    ))

    site_label = SITE_LABELS.get(site, site)
    fig.update_layout(
        title=dict(text=f"B. infantis Load (Ct) vs HAZ at 6 Months — {site_label} (B.inf+ only)"),
        xaxis=dict(title="B. infantis Ct value (lower = more bacteria)"),
        yaxis=dict(title="HAZ at 6 months"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, font=FONT, showlegend=False,
        annotations=[dict(
            text=corr_text,
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False, font=dict(size=12, color="#666"),
            bgcolor="rgba(255,255,255,0.8)",
        )],
    )
    return fig


def growth_trajectory_waz(neonatal_df):
    """Slopegraph: mean WAZ from birth → 6 months by B. infantis status, all 3 countries.

    Parallel to HAZ trajectory but using weight-for-age instead of length-for-age.
    """
    if neonatal_df is None or neonatal_df.empty:
        return _empty_figure("No neonatal data")

    df = neonatal_df.dropna(subset=["binfantis_positive", "waz1", "waz6"]).copy()
    df["bi"] = df["binfantis_positive"].astype(bool)

    sites = ["PAK", "BGD", "TZA"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[SITE_LABELS.get(s, s) for s in sites],
        shared_yaxes=True,
    )

    for col_idx, site in enumerate(sites, 1):
        sub = df[df["site"] == site]
        for status, color, dash, label in [
            (True, "#2A9D8F", "solid", "B. infantis +"),
            (False, "#E76F51", "dash", "B. infantis −"),
        ]:
            grp = sub[sub["bi"] == status]
            n = len(grp)
            if n == 0:
                continue
            mean_birth = grp["waz1"].mean()
            mean_6mo = grp["waz6"].mean()
            delta = mean_6mo - mean_birth

            fig.add_trace(go.Scatter(
                x=["Birth", "6 months"],
                y=[mean_birth, mean_6mo],
                mode="lines+markers+text",
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=10, color=color),
                text=[f"{mean_birth:.2f}", f"{mean_6mo:.2f}"],
                textposition="top center",
                name=f"{label} (n={n}, Δ={delta:+.2f})",
                legendgroup=label,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        fig.add_hline(y=-2, line_dash="dot", line_color="#ccc", line_width=1,
                      row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="Weight Trajectory (WAZ): Birth → 6 Months by B. infantis Status"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="WAZ (z-score)", row=1, col=1)
    for i in range(1, 4):
        fig.update_yaxes(range=[-2.5, 0.5], row=1, col=i)
    return fig
