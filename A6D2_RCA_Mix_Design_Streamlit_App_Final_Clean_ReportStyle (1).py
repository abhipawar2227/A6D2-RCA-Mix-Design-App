
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="RCA Mix Design Charts Automation",
    page_icon="🏗️",
    layout="wide"
)
def add_bg_video(video_path):
    video_file = Path(video_path)

    if not video_file.exists():
        st.warning(f"Background video not found: {video_path}")
        return

    video_bytes = video_file.read_bytes()
    encoded_video = base64.b64encode(video_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent;
        }}

        #bg-video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -2;
            object-fit: cover;
        }}

        #bg-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0);
            z-index: -1;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        [data-testid="stSidebar"] {{
            background: rgba(20, 20, 30, 0.88);
        }}

        .block-container {{
            background: rgba(10, 10, 18, 0.72);
            padding: 2rem;
            border-radius: 18px;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: white;
        }}
        </style>

        <video autoplay muted loop id="bg-video">
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>

        <div id="bg-overlay"></div>
        """,
        unsafe_allow_html=True
    )
add_bg_video("background.mp4")
# ============================================================
# COMMON GRADE DATA
# ============================================================

GRADE_INFO = {
    "M20": {"fck": 20.0, "s": 4.0, "target": 26.60, "ref_wc": 0.55, "wc_min": 0.40, "wc_max": 0.65},
    "M25": {"fck": 25.0, "s": 4.0, "target": 31.60, "ref_wc": 0.50, "wc_min": 0.35, "wc_max": 0.60},
    "M30": {"fck": 30.0, "s": 5.0, "target": 38.25, "ref_wc": 0.45, "wc_min": 0.32, "wc_max": 0.55},
    "M35": {"fck": 35.0, "s": 5.0, "target": 43.25, "ref_wc": 0.42, "wc_min": 0.30, "wc_max": 0.52},
    "M40": {"fck": 40.0, "s": 5.0, "target": 48.25, "ref_wc": 0.40, "wc_min": 0.28, "wc_max": 0.50},
}

M20_TARGET = 26.60
M20_REFERENCE_WC = 0.55


# ============================================================
# COMMON CALCULATION FUNCTIONS
# ============================================================

def target_mean_strength(fck, s):
    return fck + 1.65 * s


def m20_control_strength(base_wc, control_a=26.72, control_b=1.50, control_c=10.00, wc_base=0.50):
    return control_a - control_b * (base_wc - 0.40) - control_c * (base_wc - wc_base) ** 2


def m20_rca_reduction(
    rca_percent,
    base_wc,
    base_factor=2.60,
    wc_factor=0.95,
    nonlinear_factor=0.75,
    nonlinear_power=1.40,
):
    r = rca_percent / 100.0
    base = base_factor * r
    curvature = wc_factor * r * (M20_REFERENCE_WC - base_wc)
    nonlinear = nonlinear_factor * (r ** nonlinear_power)
    return base + curvature + nonlinear


def m20_base_strength(base_wc, rca_percent, constants):
    fc0 = m20_control_strength(
        base_wc,
        constants["control_a"],
        constants["control_b"],
        constants["control_c"],
        constants["wc_base"],
    )
    if rca_percent == 0:
        return fc0
    return fc0 - m20_rca_reduction(
        rca_percent,
        base_wc,
        constants["base_factor"],
        constants["wc_factor"],
        constants["nonlinear_factor"],
        constants["nonlinear_power"],
    )


def equivalent_m20_wc(actual_wc, grade_ref_wc):
    return actual_wc + (M20_REFERENCE_WC - grade_ref_wc)


def predicted_strength_grade(actual_wc, rca_percent, grade_target, grade_ref_wc, constants):
    scale = grade_target / M20_TARGET
    base_wc = equivalent_m20_wc(actual_wc, grade_ref_wc)
    base_wc = float(np.clip(base_wc, 0.35, 0.75))
    return scale * m20_base_strength(base_wc, rca_percent, constants)


def slump_corrected_water(w50, slump):
    return w50 * (1 + 0.03 * ((slump - 50.0) / 25.0))


def compensated_cement(w50, water_eff, selected_wc, target, predicted_strength, min_cement=0.0, max_cement=9999.0):
    """
    Corrected cement compensation logic.

    Base cement is calculated using the 50 mm slump base water:
        Cbase = W50 / selected_wc

    Effective/slump-corrected water is still used for:
        wc_equiv = water_eff / ccomp
        batching water calculations
    """
    cbase = w50 / selected_wc

    if predicted_strength <= 0:
        ccomp = cbase
    elif predicted_strength >= target:
        ccomp = cbase
    else:
        ccomp = cbase * (target / predicted_strength)

    ccomp = min(max(ccomp, min_cement), max_cement)
    delta = ccomp - cbase
    wc_equiv = water_eff / ccomp if ccomp != 0 else np.nan
    return cbase, ccomp, delta, wc_equiv


def absolute_volume_mix(C, W_eff, air_percent, sg_cement, sg_fa, sg_nca, sg_rca, ca_fraction, rca_percent):
    Vc = C / (sg_cement * 1000.0)
    Vw = W_eff / 1000.0
    Vair = air_percent / 100.0

    Vagg = 1.0 - Vc - Vw - Vair
    if Vagg < 0:
        Vagg = np.nan

    Vca = ca_fraction * Vagg
    Vfa = (1.0 - ca_fraction) * Vagg

    Vrca = (rca_percent / 100.0) * Vca
    Vnca = Vca - Vrca

    Mfa = Vfa * sg_fa * 1000.0
    Mnca = Vnca * sg_nca * 1000.0
    Mrca = Vrca * sg_rca * 1000.0
    Mca = Mnca + Mrca
    Magg = Mfa + Mca

    return {
        "Vc": Vc, "Vw": Vw, "Vair": Vair, "Vagg": Vagg,
        "Vca": Vca, "Vfa": Vfa, "Vrca": Vrca, "Vnca": Vnca,
        "Mfa": Mfa, "Mnca": Mnca, "Mrca": Mrca, "Mca": Mca, "Magg": Magg,
    }


def interpolate_value(df, rca_percent, col):
    return float(np.interp(rca_percent, df["RCA replacement (%)"], df[col]))


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# DATA GENERATION FUNCTIONS
# ============================================================

def generate_chart1_data(grade, grade_target, grade_ref_wc, w50, water_eff, wc_min, wc_max, wc_step, rca_interval, constants):
    wc_values = np.round(np.arange(wc_min, wc_max + 0.0001, wc_step), 3)
    rca_levels = list(range(0, 101, rca_interval))
    rows = []

    for rca in rca_levels:
        for wc in wc_values:
            C = w50 / wc
            fc = predicted_strength_grade(wc, rca, grade_target, grade_ref_wc, constants)
            rows.append({
                "Grade": grade,
                "w/c ratio": wc,
                "Equivalent M20 w/c": equivalent_m20_wc(wc, grade_ref_wc),
                "RCA replacement (%)": rca,
                "Effective water (kg/m3)": water_eff,
                "Cement content C=W/(w/c) (kg/m3)": C,
                "Predicted strength (MPa)": fc,
            })

    return pd.DataFrame(rows)


def generate_chart2_data(grade, grade_target, grade_ref_wc, w50, water_eff, wc_min, wc_max, wc_step, rca_interval, constants):
    wc_values = np.round(np.arange(wc_min, wc_max + 0.0001, wc_step), 3)
    rca_levels = list(range(0, 101, rca_interval))
    rows = []

    for wc in wc_values:
        f_control = predicted_strength_grade(wc, 0, grade_target, grade_ref_wc, constants)

        for rca in rca_levels:
            f_rca = predicted_strength_grade(wc, rca, grade_target, grade_ref_wc, constants)
            srf = f_rca / f_control if f_control != 0 else np.nan
            cement = w50 / wc
            rows.append({
                "Grade": grade,
                "w/c ratio": wc,
                "Equivalent M20 w/c": equivalent_m20_wc(wc, grade_ref_wc),
                "RCA replacement (%)": rca,
                "Control strength 0% RCA (MPa)": f_control,
                "RCA strength (MPa)": f_rca,
                "SRF": srf,
                "Effective water (kg/m3)": water_eff,
                "Cement content C=W/(w/c) (kg/m3)": cement,
            })

    return pd.DataFrame(rows)


def generate_chart3_data(grade, grade_target, grade_ref_wc, w50, water_eff, wc_min, wc_max, wc_step, rca_interval, constants):
    wc_values = np.round(np.arange(wc_min, wc_max + 0.0001, wc_step), 3)
    rca_levels = list(range(0, 101, rca_interval))
    rows = []

    for rca in rca_levels:
        for wc in wc_values:
            f_pred = predicted_strength_grade(wc, rca, grade_target, grade_ref_wc, constants)
            Cbase = w50 / wc

            if f_pred >= grade_target:
                Ccomp = Cbase
                delta_c = 0.0
                wc_equivalent = wc
            else:
                Ccomp = Cbase * (grade_target / f_pred)
                delta_c = Ccomp - Cbase
                wc_equivalent = water_eff / Ccomp

            rows.append({
                "Grade": grade,
                "w/c ratio": wc,
                "Equivalent M20 w/c": equivalent_m20_wc(wc, grade_ref_wc),
                "RCA replacement (%)": rca,
                "Target mean strength (MPa)": grade_target,
                "Predicted strength (MPa)": f_pred,
                "Base cement Cbase=W/(w/c) (kg/m3)": Cbase,
                "Compensated cement Ccomp (kg/m3)": Ccomp,
                "Additional cement Delta C (kg/m3)": delta_c,
                "Equivalent w/c after compensation": wc_equivalent,
                "Effective water (kg/m3)": water_eff,
            })

    return pd.DataFrame(rows)


def generate_chart4_data(grade, fck, s, target, grade_ref_wc, selected_wc, w50, slump, air_percent,
                         sg_cement, sg_fa, sg_nca, sg_rca, rca_wa, ca_fraction, rca_interval,
                         constants, min_cement, max_cement):
    water_eff = slump_corrected_water(w50, slump)
    rows = []

    for rca in range(0, 101, rca_interval):
        f_pred = predicted_strength_grade(selected_wc, rca, target, grade_ref_wc, constants)
        cbase, ccomp, delta_c, wc_equiv = compensated_cement(
            w50, water_eff, selected_wc, target, f_pred, min_cement, max_cement
        )

        mix = absolute_volume_mix(ccomp, water_eff, air_percent, sg_cement, sg_fa, sg_nca, sg_rca, ca_fraction, rca)

        rows.append({
            "Grade": grade,
            "fck (MPa)": fck,
            "s (MPa)": s,
            "Target mean strength (MPa)": target,
            "Selected w/c": selected_wc,
            "RCA replacement (%)": rca,
            "Predicted strength (MPa)": f_pred,
            "Base cement Cbase (kg/m3)": cbase,
            "Compensated cement Ccomp (kg/m3)": ccomp,
            "Additional cement Delta C (kg/m3)": delta_c,
            "Equivalent w/c after compensation": wc_equiv,
            "Effective water W_eff (kg/m3)": water_eff,
            "Cement volume Vc": mix["Vc"],
            "Water volume Vw": mix["Vw"],
            "Air volume Vair": mix["Vair"],
            "Total aggregate volume Vagg": mix["Vagg"],
            "Coarse aggregate volume Vca": mix["Vca"],
            "Fine aggregate volume Vfa": mix["Vfa"],
            "RCA volume Vrca": mix["Vrca"],
            "NCA volume Vnca": mix["Vnca"],
            "Fine aggregate content (kg/m3)": mix["Mfa"],
            "NCA aggregate content (kg/m3)": mix["Mnca"],
            "RCA aggregate content (kg/m3)": mix["Mrca"],
            "Total coarse aggregate (kg/m3)": mix["Mca"],
            "Total aggregate content (kg/m3)": mix["Magg"],
            "RCA water absorption (%)": rca_wa,
            "RCA specific gravity": sg_rca,
        })

    return pd.DataFrame(rows)


def generate_chart6_data(grade, fck, s, target, grade_ref_wc, selected_wc, w50, slump, air_percent,
                         sg_cement, sg_fa, sg_nca, sg_rca, rca_wa, rca_mc, ca_fraction,
                         rca_interval, constants, min_cement, max_cement):
    df4 = generate_chart4_data(
        grade, fck, s, target, grade_ref_wc, selected_wc, w50, slump, air_percent,
        sg_cement, sg_fa, sg_nca, sg_rca, rca_wa, ca_fraction, rca_interval,
        constants, min_cement, max_cement
    )

    df = df4.copy()
    absorption_deficit = max(rca_wa - rca_mc, 0.0)
    free_surface_pct = max(rca_mc - rca_wa, 0.0)

    df["RCA moisture content MC (%)"] = rca_mc
    df["Absorption deficit WA-MC (%)"] = absorption_deficit
    df["Extra absorption water (kg/m3)"] = df["RCA aggregate content (kg/m3)"] * absorption_deficit / 100.0
    df["Free surface water if MC>WA (kg/m3)"] = df["RCA aggregate content (kg/m3)"] * free_surface_pct / 100.0
    df["Batching water to be added (kg/m3)"] = (
        df["Effective water W_eff (kg/m3)"]
        + df["Extra absorption water (kg/m3)"]
        - df["Free surface water if MC>WA (kg/m3)"]
    )

    return df


def classify_rca_quality(sg, wa):
    if sg >= 2.50 and wa <= 3.0:
        return "Good RCA", "Suitable RCA quality; low absorption and good specific gravity."
    elif sg >= 2.30 and wa <= 6.0:
        return "Moderate RCA", "Usable RCA; absorption correction and trial mix validation are required."
    elif sg >= 2.10 and wa <= 10.0:
        return "Poor / High absorption RCA", "Use with caution; higher water correction and trial mix validation are necessary."
    else:
        return "Not recommended", "Very low specific gravity or very high absorption; avoid unless separately validated."


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_chart1(df, grade, target, reference_wc, selected_rca, selected_wc, water_eff, slump_basis, aggregate_size, zoom_enabled=False, xlim_min=None, xlim_max=None, ylim_min=None, ylim_max=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["Predicted strength (MPa)"], marker="o", markersize=3.5,
                linewidth=lw, label=f"{int(rca)}% RCA")

    ax.axhline(target, linestyle="--", linewidth=1.5, label=f"{grade} target mean = {target:.2f} MPa")
    ax.axvline(reference_wc, linestyle=":", linewidth=1.5, label=f"{grade} reference design w/c = {reference_wc:.2f}")

    if selected_rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == selected_rca].sort_values("w/c ratio")
        selected_strength = float(np.interp(selected_wc, sub["w/c ratio"], sub["Predicted strength (MPa)"]))
        ax.scatter([selected_wc], [selected_strength], marker="*", s=160, zorder=6)
        ax.annotate(
            f"{grade}, {selected_rca}% RCA\nw/c = {selected_wc:.3f}\nf = {selected_strength:.2f} MPa",
            xy=(selected_wc, selected_strength),
            xytext=(selected_wc + 0.025, selected_strength + 0.30),
            arrowprops=dict(arrowstyle="->", linewidth=1),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
            fontsize=9,
        )

    ax.set_title(
        f"{grade} Predictive Strength vs w/c Ratio\n"
        f"Base chart: {slump_basis:.0f} mm slump, {aggregate_size:.0f} mm aggregate, "
        f"W_eff = {water_eff:.2f} kg/m³; cement table uses W50/(w/c)",
        fontsize=13,
    )
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Predicted 28-day compressive strength (MPa)")
    ax.grid(True, alpha=0.30)

    # Optional manual zoom for Chart 1
    if zoom_enabled:
        if xlim_min is not None and xlim_max is not None and xlim_min < xlim_max:
            ax.set_xlim(xlim_min, xlim_max)
        if ylim_min is not None and ylim_max is not None and ylim_min < ylim_max:
            ax.set_ylim(ylim_min, ylim_max)

    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def plot_chart2(df, grade, reference_wc, selected_rca, selected_wc):
    fig, ax = plt.subplots(figsize=(12, 8))
    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["SRF"], marker="o", markersize=3.5,
                linewidth=lw, label=f"{int(rca)}% RCA")

    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="0% RCA reference SRF = 1.0")
    ax.axvline(reference_wc, linestyle=":", linewidth=1.5, label=f"{grade} reference design w/c = {reference_wc:.2f}")

    sub = df[df["RCA replacement (%)"] == selected_rca].sort_values("w/c ratio")
    selected_srf = float(np.interp(selected_wc, sub["w/c ratio"], sub["SRF"]))
    ax.scatter([selected_wc], [selected_srf], marker="*", s=160, zorder=6)

    ax.annotate(
        f"{grade}, {selected_rca}% RCA\nw/c = {selected_wc:.3f}\nSRF = {selected_srf:.3f}",
        xy=(selected_wc, selected_srf),
        xytext=(selected_wc + 0.025, selected_srf + 0.015),
        arrowprops=dict(arrowstyle="->", linewidth=1),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
        fontsize=9,
    )

    ax.set_title(f"{grade} Chart 2: Strength Reduction Factor vs w/c Ratio\nSRF = f_RCA / f_0%RCA", fontsize=13)
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Strength Reduction Factor, SRF")
    ax.set_ylim(0.72, 1.03)
    ax.grid(True, alpha=0.30)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def plot_chart3_delta(df, grade, target, selected_rca, selected_wc):
    fig, ax = plt.subplots(figsize=(12, 8))

    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["Additional cement Delta C (kg/m3)"], marker="o",
                markersize=3.5, linewidth=lw, label=f"{int(rca)}% RCA")

    selected_delta = interpolate_value_wc(df, selected_rca, selected_wc, "Additional cement Delta C (kg/m3)")
    selected_ccomp = interpolate_value_wc(df, selected_rca, selected_wc, "Compensated cement Ccomp (kg/m3)")
    selected_fpred = interpolate_value_wc(df, selected_rca, selected_wc, "Predicted strength (MPa)")

    ax.axvline(selected_wc, linestyle=":", linewidth=1.5, label=f"Selected w/c = {selected_wc:.2f}")
    ax.scatter([selected_wc], [selected_delta], marker="*", s=170, zorder=6)

    ax.annotate(
        f"{grade}, {selected_rca}% RCA\nw/c = {selected_wc:.3f}\nf = {selected_fpred:.2f} MPa\n"
        f"ΔC = {selected_delta:.2f} kg/m³\nCcomp = {selected_ccomp:.2f} kg/m³",
        xy=(selected_wc, selected_delta),
        xytext=(selected_wc + 0.025, selected_delta + 8),
        arrowprops=dict(arrowstyle="->", linewidth=1),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
        fontsize=9,
    )

    ax.set_title(f"{grade} Chart 3: Cement Compensation vs w/c Ratio\nCcomp = Cbase × target / fpred, target = {target:.2f} MPa", fontsize=13)
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Additional cement required, ΔC (kg/m³)")
    ax.grid(True, alpha=0.30)
    ax.legend(ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_chart3_ccomp(df, grade, selected_rca, selected_wc):
    fig, ax = plt.subplots(figsize=(12, 8))

    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["Compensated cement Ccomp (kg/m3)"], marker="o",
                markersize=3.5, linewidth=lw, label=f"{int(rca)}% RCA")

    selected_ccomp = interpolate_value_wc(df, selected_rca, selected_wc, "Compensated cement Ccomp (kg/m3)")
    ax.axvline(selected_wc, linestyle=":", linewidth=1.5, label=f"Selected w/c = {selected_wc:.2f}")
    ax.scatter([selected_wc], [selected_ccomp], marker="*", s=170, zorder=6)

    ax.set_title(f"{grade}: Compensated Cement Content vs w/c Ratio", fontsize=13)
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Compensated cement content, Ccomp (kg/m³)")
    ax.grid(True, alpha=0.30)
    ax.legend(ncol=2, fontsize=9, loc="upper right")
    fig.tight_layout()
    return fig


def interpolate_value_wc(df, rca_percent, wc, col):
    sub = df[df["RCA replacement (%)"] == rca_percent].sort_values("w/c ratio")
    return float(np.interp(wc, sub["w/c ratio"], sub[col]))


def plot_chart4(df, grade, selected_rca, selected_wc, slump, aggregate_size, wa_axis_min, wa_axis_max):
    x = df["RCA replacement (%)"].values
    cement = df["Compensated cement Ccomp (kg/m3)"].values
    rca_agg = df["RCA aggregate content (kg/m3)"].values
    wa = df["RCA water absorption (%)"].values
    sg = df["RCA specific gravity"].values

    selected_c = interpolate_value(df, selected_rca, "Compensated cement Ccomp (kg/m3)")
    selected_rca_agg = interpolate_value(df, selected_rca, "RCA aggregate content (kg/m3)")
    selected_wa = interpolate_value(df, selected_rca, "RCA water absorption (%)")
    selected_sg = interpolate_value(df, selected_rca, "RCA specific gravity")

    fig, ax1 = plt.subplots(figsize=(13, 8))

    l1 = ax1.plot(x, cement, marker="o", linewidth=2.5, color="#0057c2", label="Compensated cement (kg/m³)")
    ax1.set_xlabel("RCA replacement (%)")
    ax1.set_ylabel("Compensated cement content (kg/m³)", color="#0057c2")
    ax1.tick_params(axis="y", labelcolor="#0057c2")
    ax1.set_xlim(-2, 102)
    ax1.set_xticks(x)
    ax1.set_ylim(min(cement) - 8, max(cement) + 8)
    ax1.grid(True, alpha=0.30)

    for xi, yi in zip(x, cement):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#0057c2")

    ax2 = ax1.twinx()
    l2 = ax2.plot(x, rca_agg, marker="^", linewidth=2.5, color="green", label="RCA aggregate (kg/m³)")
    ax2.set_ylabel("RCA aggregate content (kg/m³)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, max(100, np.nanmax(rca_agg) * 1.15))

    for xi, yi in zip(x, rca_agg):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color="green")

    ax3 = ax1.twinx()
    ax3.spines["left"].set_position(("axes", -0.13))
    ax3.yaxis.set_label_position("left")
    ax3.yaxis.set_ticks_position("left")
    ax3.spines["left"].set_visible(True)
    ax3.spines["right"].set_visible(False)

    l3 = ax3.plot(x, wa, marker="s", linestyle="--", linewidth=2.0, color="#ff7f0e",
                  label="RCA water absorption (%)")
    ax3.set_ylabel("RCA water absorption (%)", color="#ff7f0e")
    ax3.tick_params(axis="y", labelcolor="#ff7f0e")
    ax3.set_ylim(wa_axis_min, wa_axis_max)
    ax3.set_yticks(np.linspace(wa_axis_min, wa_axis_max, 6))

    for i, (xi, yi) in enumerate(zip(x, wa)):
        ax3.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 10 if i % 2 == 0 else 18), ha="center", fontsize=8, color="#ff7f0e")

    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("axes", 1.13))
    l4 = ax4.plot(x, sg, marker="D", linestyle="--", linewidth=2.0, color="red",
                  label="RCA specific gravity")
    ax4.set_ylabel("RCA specific gravity", color="red")
    ax4.tick_params(axis="y", labelcolor="red")
    ax4.set_ylim(min(sg) - 0.15, max(sg) + 0.15)

    for i, (xi, yi) in enumerate(zip(x, sg)):
        ax4.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, -18 if i % 2 == 0 else -26), ha="center", fontsize=8, color="red")

    ax1.axvline(selected_rca, linestyle=":", linewidth=1.6, color="dodgerblue")
    ax1.scatter([selected_rca], [selected_c], marker="*", s=200, color="#0057c2", zorder=8)
    ax2.scatter([selected_rca], [selected_rca_agg], marker="*", s=200, color="green", zorder=8)

    result_text = (
        f"Selected {grade}, {selected_rca}% RCA\n"
        f"C = {selected_c:.2f} kg/m³\n"
        f"RCA = {selected_rca_agg:.2f} kg/m³\n"
        f"WA = {selected_wa:.3f}%\n"
        f"SG = {selected_sg:.2f}"
    )
    ax1.text(0.985, 0.055, result_text, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"), zorder=10)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower center", fontsize=9, framealpha=0.92)

    ax1.set_title(
        f"{grade} Dynamic RCA Combined 4-Axis Chart\n"
        f"Basis: selected w/c = {selected_wc:.3f}, slump = {slump:.0f} mm, {aggregate_size:.0f} mm aggregate",
        fontsize=15, fontweight="bold"
    )

    fig.tight_layout(rect=[0.08, 0.04, 0.92, 0.96])
    return fig


def plot_chart5(selected_sg, selected_wa, xmin, xmax, ymin, ymax, show_reference_points=True):
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.set_facecolor("white")

    ax.add_patch(Rectangle((2.10, 0), xmax - 2.10, min(10, ymax), facecolor="#ffd6d6", edgecolor="none", alpha=0.35))
    ax.add_patch(Rectangle((2.30, 0), xmax - 2.30, min(6, ymax), facecolor="#ffe9b8", edgecolor="none", alpha=0.55))
    ax.add_patch(Rectangle((2.50, 0), xmax - 2.50, min(3, ymax), facecolor="#cfead1", edgecolor="none", alpha=0.75))

    ax.axvline(2.10, linestyle="--", linewidth=1.3, color="red")
    ax.axvline(2.30, linestyle="--", linewidth=1.3, color="orange")
    ax.axvline(2.50, linestyle="--", linewidth=1.3, color="green")
    ax.axhline(3.0, linestyle="--", linewidth=1.3, color="green")
    ax.axhline(6.0, linestyle="--", linewidth=1.3, color="orange")
    ax.axhline(10.0, linestyle="--", linewidth=1.3, color="red")

    ax.text(2.62, 1.5, "GOOD RCA\nSG ≥ 2.50\nWA ≤ 3%", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="green"), fontsize=10)
    ax.text(2.42, 4.5, "MODERATE RCA\nSG ≥ 2.30\nWA ≤ 6%", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="orange"), fontsize=10)
    ax.text(2.23, 8.2, "POOR / HIGH\nABSORPTION RCA\nUse with caution", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="red"), fontsize=10)

    if show_reference_points:
        ref_data = pd.DataFrame({
            "Name": ["Typical NCA", "Selected RCA basis", "High WA RCA example"],
            "SG": [2.70, 2.54, 2.20],
            "WA": [0.8, 4.515, 8.0],
        })
        ax.scatter(ref_data["SG"], ref_data["WA"], s=65, alpha=0.75, color="tab:blue", zorder=6)
        label_offsets = {"Typical NCA": (10, 10), "Selected RCA basis": (10, -18), "High WA RCA example": (10, 14)}
        for _, row in ref_data.iterrows():
            dx, dy = label_offsets[row["Name"]]
            ax.annotate(row["Name"], (row["SG"], row["WA"]), textcoords="offset points", xytext=(dx, dy), fontsize=9)

    quality, note = classify_rca_quality(selected_sg, selected_wa)
    ax.scatter([selected_sg], [selected_wa], marker="*", s=300, color="blue", edgecolor="black", linewidth=0.4, zorder=10)
    ax.axvline(selected_sg, linestyle=":", linewidth=1.1, color="blue", alpha=0.65)
    ax.axhline(selected_wa, linestyle=":", linewidth=1.1, color="blue", alpha=0.65)

    result_text = f"Selected RCA\nSG = {selected_sg:.2f}\nWA = {selected_wa:.3f}%\n{quality}"
    ax.text(0.985, 0.055, result_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"), zorder=12)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("RCA specific gravity")
    ax.set_ylabel("RCA water absorption (%)")
    ax.set_title("Chart 5: RCA Quality Check — Specific Gravity vs Water Absorption", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.28)

    legend_items = [
        Patch(facecolor="#cfead1", edgecolor="green", alpha=0.75, label="Good zone"),
        Patch(facecolor="#ffe9b8", edgecolor="orange", alpha=0.55, label="Moderate zone"),
        Patch(facecolor="#ffd6d6", edgecolor="red", alpha=0.35, label="Poor / high absorption zone"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return fig, quality, note


def plot_chart6(df, grade, selected_rca, selected_wc, rca_wa, rca_mc):
    x = df["RCA replacement (%)"].values
    extra_water = df["Extra absorption water (kg/m3)"].values
    rca_mass = df["RCA aggregate content (kg/m3)"].values
    selected_extra = interpolate_value(df, selected_rca, "Extra absorption water (kg/m3)")
    selected_rca_mass = interpolate_value(df, selected_rca, "RCA aggregate content (kg/m3)")
    selected_batching = interpolate_value(df, selected_rca, "Batching water to be added (kg/m3)")

    fig, ax1 = plt.subplots(figsize=(12.5, 8))
    l1 = ax1.plot(x, extra_water, marker="o", linewidth=2.5, color="#0057c2", label="Extra absorption water (kg/m³)")
    ax1.set_xlabel("RCA replacement (%)")
    ax1.set_ylabel("Extra absorption water (kg/m³)", color="#0057c2")
    ax1.tick_params(axis="y", labelcolor="#0057c2")
    ax1.set_xlim(-2, 102)
    ax1.set_xticks(x)
    ax1.set_ylim(0, max(5, np.nanmax(extra_water) * 1.25))
    ax1.grid(True, alpha=0.30)

    for xi, yi in zip(x, extra_water):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#0057c2")

    ax2 = ax1.twinx()
    l2 = ax2.plot(x, rca_mass, marker="^", linewidth=2.4, color="green", label="RCA aggregate content (kg/m³)")
    ax2.set_ylabel("RCA aggregate content (kg/m³)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, max(100, np.nanmax(rca_mass) * 1.15))

    ax1.axvline(selected_rca, linestyle=":", linewidth=1.5, color="dodgerblue")
    ax1.scatter([selected_rca], [selected_extra], marker="*", s=200, color="#0057c2", zorder=8)
    ax2.scatter([selected_rca], [selected_rca_mass], marker="*", s=200, color="green", zorder=8)

    result_text = (
        f"Selected {grade}, {selected_rca}% RCA\n"
        f"w/c = {selected_wc:.3f}\n"
        f"RCA = {selected_rca_mass:.2f} kg/m³\n"
        f"WA = {rca_wa:.3f}%\n"
        f"MC = {rca_mc:.3f}%\n"
        f"Extra water = {selected_extra:.2f} kg/m³\n"
        f"Batching water = {selected_batching:.2f} kg/m³"
    )
    ax1.text(0.985, 0.055, result_text, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"), zorder=10)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9, framealpha=0.95)
    ax1.set_title(
        f"{grade} Chart 6: Extra Absorption Water vs RCA Replacement\n"
        f"Wextra = RCA mass × max(WA - MC, 0) / 100; selected w/c = {selected_wc:.3f}",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    return fig




# ============================================================
# FINAL MIX TABLE FUNCTION
# ============================================================

def make_final_mix_table(
    grade,
    fck,
    target,
    selected_wc,
    selected_rca,
    w50,
    chart4_df,
    chart6_df,
    rca_wa,
    rca_mc,
    sg_cement,
    sg_fa,
    sg_nca,
    sg_rca
):
    ccomp = interpolate_value(chart4_df, selected_rca, "Compensated cement Ccomp (kg/m3)")
    cbase = interpolate_value(chart4_df, selected_rca, "Base cement Cbase (kg/m3)")
    delta_c = interpolate_value(chart4_df, selected_rca, "Additional cement Delta C (kg/m3)")
    water_eff = interpolate_value(chart4_df, selected_rca, "Effective water W_eff (kg/m3)")
    fa = interpolate_value(chart4_df, selected_rca, "Fine aggregate content (kg/m3)")
    nca = interpolate_value(chart4_df, selected_rca, "NCA aggregate content (kg/m3)")
    rca = interpolate_value(chart4_df, selected_rca, "RCA aggregate content (kg/m3)")
    total_ca = interpolate_value(chart4_df, selected_rca, "Total coarse aggregate (kg/m3)")
    total_agg = interpolate_value(chart4_df, selected_rca, "Total aggregate content (kg/m3)")
    extra_water = interpolate_value(chart6_df, selected_rca, "Extra absorption water (kg/m3)")
    free_water = interpolate_value(chart6_df, selected_rca, "Free surface water if MC>WA (kg/m3)")
    batching_water = interpolate_value(chart6_df, selected_rca, "Batching water to be added (kg/m3)")
    predicted_strength = interpolate_value(chart4_df, selected_rca, "Predicted strength (MPa)")
    equiv_wc = interpolate_value(chart4_df, selected_rca, "Equivalent w/c after compensation")

    # Ratios with respect to cement
    fa_ratio = fa / ccomp if ccomp else np.nan
    nca_ratio = nca / ccomp if ccomp else np.nan
    rca_ratio = rca / ccomp if ccomp else np.nan
    total_ca_ratio = total_ca / ccomp if ccomp else np.nan
    total_agg_ratio = total_agg / ccomp if ccomp else np.nan

    effective_wc = water_eff / ccomp if ccomp else np.nan
    batching_wc = batching_water / ccomp if ccomp else np.nan

    quantity_table = pd.DataFrame([
        {"Item": "Concrete grade", "Value": grade, "Unit": "-"},
        {"Item": "fck", "Value": f"{fck:.2f}", "Unit": "MPa"},
        {"Item": "Target mean strength", "Value": f"{target:.2f}", "Unit": "MPa"},
        {"Item": "Selected RCA replacement", "Value": f"{selected_rca:.0f}", "Unit": "%"},
        {"Item": "Selected design w/c", "Value": f"{selected_wc:.3f}", "Unit": "-"},
        {"Item": "Predicted strength at selected RCA", "Value": f"{predicted_strength:.2f}", "Unit": "MPa"},
        {"Item": "Base cement content", "Value": f"{cbase:.2f}", "Unit": "kg/m³"},
        {"Item": "Additional cement, ΔC", "Value": f"{delta_c:.2f}", "Unit": "kg/m³"},
        {"Item": "Final compensated cement", "Value": f"{ccomp:.2f}", "Unit": "kg/m³"},
        {"Item": "Base water W50 used for cement calculation", "Value": f"{w50:.2f}", "Unit": "kg/m³"},
        {"Item": "Effective water", "Value": f"{water_eff:.2f}", "Unit": "kg/m³"},
        {"Item": "Extra absorption water", "Value": f"{extra_water:.2f}", "Unit": "kg/m³"},
        {"Item": "Free surface water correction", "Value": f"{free_water:.2f}", "Unit": "kg/m³"},
        {"Item": "Batching water to be added", "Value": f"{batching_water:.2f}", "Unit": "kg/m³"},
        {"Item": "Fine aggregate", "Value": f"{fa:.2f}", "Unit": "kg/m³"},
        {"Item": "Natural coarse aggregate", "Value": f"{nca:.2f}", "Unit": "kg/m³"},
        {"Item": "Recycled coarse aggregate", "Value": f"{rca:.2f}", "Unit": "kg/m³"},
        {"Item": "Total coarse aggregate", "Value": f"{total_ca:.2f}", "Unit": "kg/m³"},
        {"Item": "Total aggregate", "Value": f"{total_agg:.2f}", "Unit": "kg/m³"},
        {"Item": "RCA water absorption", "Value": f"{rca_wa:.3f}", "Unit": "%"},
        {"Item": "RCA moisture content", "Value": f"{rca_mc:.3f}", "Unit": "%"},
        {"Item": "RCA specific gravity", "Value": f"{sg_rca:.2f}", "Unit": "-"},
        {"Item": "Effective w/c after compensation", "Value": f"{effective_wc:.3f}", "Unit": "-"},
        {"Item": "Batching water/cement ratio", "Value": f"{batching_wc:.3f}", "Unit": "-"},
    ])

    ratio_table = pd.DataFrame([
        {
            "Mix ratio type": "Conventional ratio",
            "Ratio": f"Cement : FA : Total CA = 1 : {fa_ratio:.2f} : {total_ca_ratio:.2f}",
            "Notes": "Total CA = NCA + RCA"
        },
        {
            "Mix ratio type": "RCA split ratio",
            "Ratio": f"Cement : FA : NCA : RCA = 1 : {fa_ratio:.2f} : {nca_ratio:.2f} : {rca_ratio:.2f}",
            "Notes": "Recommended for RCA mix reporting"
        },
        {
            "Mix ratio type": "Total aggregate ratio",
            "Ratio": f"Cement : Total Aggregate = 1 : {total_agg_ratio:.2f}",
            "Notes": "Total aggregate = FA + NCA + RCA"
        },
        {
            "Mix ratio type": "Water-cement basis",
            "Ratio": f"Effective w/c = {effective_wc:.3f}; Batching w/c = {batching_wc:.3f}",
            "Notes": "Batching water includes RCA absorption correction"
        },
    ])

    summary_dict = {
        "Cement kg/m3": ccomp,
        "Water effective kg/m3": water_eff,
        "Batching water kg/m3": batching_water,
        "FA kg/m3": fa,
        "NCA kg/m3": nca,
        "RCA kg/m3": rca,
        "Total CA kg/m3": total_ca,
        "Total aggregate kg/m3": total_agg,
        "Effective w/c": effective_wc,
        "Batching w/c": batching_wc,
        "Mix ratio C:FA:CA": f"1 : {fa_ratio:.2f} : {total_ca_ratio:.2f}",
        "Mix ratio C:FA:NCA:RCA": f"1 : {fa_ratio:.2f} : {nca_ratio:.2f} : {rca_ratio:.2f}",
    }

    return quantity_table, ratio_table, summary_dict


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("RCA Mix Design Charts Automation")
st.write(
    "Combined app for Charts 1–6 using one common input panel. "
    "Charts 1–4 and 6 are dynamic for M20–M40. Chart 5 checks RCA material quality. Cement base uses W50/selected w/c; batching water uses slump-corrected W_eff. Graph size can be kept constant, and Chart 1 has optional manual zoom."
)

with st.sidebar:
    st.header("Common grade inputs")

    grade = st.selectbox("Concrete grade", list(GRADE_INFO.keys()), index=0)
    g = GRADE_INFO[grade]

    fck = st.number_input("fck (MPa)", value=float(g["fck"]), step=1.0)
    s = st.number_input("Standard deviation s (MPa)", value=float(g["s"]), step=0.5)
    target = st.number_input("Target mean strength (MPa)", value=float(target_mean_strength(fck, s)), step=0.1)
    reference_wc = st.number_input("Reference/design w/c", value=float(g["ref_wc"]), step=0.01)
    selected_wc = st.number_input("Selected/check w/c", value=float(g["ref_wc"]), min_value=0.20, max_value=1.00, step=0.01)

    st.header("Chart range")
    wc_min = st.number_input("Chart w/c minimum", value=float(g["wc_min"]), step=0.01)
    wc_max = st.number_input("Chart w/c maximum", value=float(g["wc_max"]), step=0.01)
    wc_step = st.number_input("Chart w/c interval", value=0.01, min_value=0.005, max_value=0.05, step=0.005)
    rca_interval = st.selectbox("RCA replacement interval (%)", [5, 10, 20, 25], index=1)

    available_rca = list(range(0, 101, rca_interval))
    default_rca = 40 if 40 in available_rca else 50
    selected_rca = st.select_slider("Highlight RCA replacement (%)", options=available_rca, value=default_rca)

    st.header("Page display zoom")
    page_zoom = st.select_slider(
        "Page zoom / UI scale (%)",
        options=[75, 80, 90, 100, 110, 125, 150],
        value=100
    )
    compact_mode = st.checkbox("Compact page spacing", value=False)

    st.header("Graph display / zoom")
    constant_graph_size = st.checkbox("Keep graph size constant", value=True)
    enable_chart1_zoom = st.checkbox("Enable Chart 1 manual zoom", value=False)

    if enable_chart1_zoom:
        chart1_xmin = st.number_input("Chart 1 zoom x-min", value=float(wc_min), step=0.01, format="%.3f")
        chart1_xmax = st.number_input("Chart 1 zoom x-max", value=float(wc_max), step=0.01, format="%.3f")
        chart1_ymin = st.number_input("Chart 1 zoom y-min", value=20.0, step=0.5)
        chart1_ymax = st.number_input("Chart 1 zoom y-max", value=max(float(target) + 2.0, 30.0), step=0.5)
    else:
        chart1_xmin = None
        chart1_xmax = None
        chart1_ymin = None
        chart1_ymax = None

    st.header("Water and aggregate inputs")
    w50 = st.number_input("Base water for 50 mm slump, W50 (kg/m³)", value=186.0, step=1.0)
    slump = st.number_input("Required slump (mm)", value=100.0, step=5.0)
    aggregate_size = st.number_input("Nominal aggregate size (mm)", value=20.0, step=1.0)
    water_eff_chart = slump_corrected_water(w50, slump)
    st.caption(f"Calculated W_eff = {water_eff_chart:.2f} kg/m³")

    air_percent = st.number_input("Air content (%)", value=0.5, step=0.1)
    ca_fraction = st.number_input("CA volume fraction", value=0.620, step=0.001, format="%.3f")

    st.header("Material properties")
    sg_cement = st.number_input("Cement specific gravity", value=3.15, step=0.01)
    sg_fa = st.number_input("FA specific gravity", value=2.65, step=0.01)
    sg_nca = st.number_input("NCA specific gravity", value=2.70, step=0.01)
    sg_rca = st.number_input("RCA specific gravity", value=2.54, step=0.01)
    rca_wa = st.number_input("RCA water absorption, WA (%)", value=4.515, min_value=0.0, max_value=20.0, step=0.001)
    rca_mc = st.number_input("RCA moisture content, MC (%)", value=0.000, min_value=0.0, max_value=20.0, step=0.001)

    st.header("Axis/settings")
    wa_axis_min = st.number_input("Chart 4 WA axis minimum (%)", value=0.0, step=0.5)
    wa_axis_max = st.number_input("Chart 4 WA axis maximum (%)", value=10.0, step=0.5)

    chart5_xmin = st.number_input("Chart 5 SG axis minimum", value=2.00, step=0.05)
    chart5_xmax = st.number_input("Chart 5 SG axis maximum", value=2.80, step=0.05)
    chart5_ymin = st.number_input("Chart 5 WA axis minimum (%)", value=0.00, step=0.5)
    chart5_ymax = st.number_input("Chart 5 WA axis maximum (%)", value=10.00, step=0.5)
    show_reference_points = st.checkbox("Show Chart 5 reference points", value=True)

    st.header("Cement limits")
    min_cement = st.number_input("Minimum cement limit (kg/m³)", value=0.0, step=5.0)
    max_cement = st.number_input("Maximum cement limit (kg/m³)", value=9999.0, step=5.0)

    st.header("Curve calibration")
    st.caption("Keep defaults to match the earlier accepted M20 chart style.")
    control_a = st.number_input("Control curve constant a", value=26.72, step=0.01)
    control_b = st.number_input("Control curve slope b", value=1.50, step=0.05)
    control_c = st.number_input("Control curve curvature c", value=10.00, step=0.50)
    wc_base = st.number_input("Control curve curvature base w/c", value=0.50, step=0.01)

    base_factor = st.number_input("RCA base reduction factor", value=2.60, step=0.05)
    wc_factor = st.number_input("RCA w/c curvature factor", value=0.95, step=0.05)
    nonlinear_factor = st.number_input("RCA nonlinear factor", value=0.75, step=0.05)
    nonlinear_power = st.number_input("RCA nonlinear power", value=1.40, step=0.05)

# ============================================================
# PAGE DISPLAY ZOOM CSS
# ============================================================

zoom_value = page_zoom / 100.0
container_padding = "1.0rem 1.0rem 1.0rem 1.0rem" if compact_mode else "2rem 2.2rem 2.2rem 2.2rem"
metric_font = 0.90 if compact_mode else 1.00

st.markdown(
    f"""
    <style>
    html {{
        zoom: {zoom_value};
    }}

    .block-container {{
        padding: {container_padding};
    }}

    [data-testid="stMetricValue"] {{
        font-size: {metric_font}rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


constants = {
    "control_a": control_a,
    "control_b": control_b,
    "control_c": control_c,
    "wc_base": wc_base,
    "base_factor": base_factor,
    "wc_factor": wc_factor,
    "nonlinear_factor": nonlinear_factor,
    "nonlinear_power": nonlinear_power,
}

# Generate all data once
chart1_df = generate_chart1_data(grade, target, reference_wc, w50, water_eff_chart, wc_min, wc_max, wc_step, rca_interval, constants)
chart2_df = generate_chart2_data(grade, target, reference_wc, w50, water_eff_chart, wc_min, wc_max, wc_step, rca_interval, constants)
chart3_df = generate_chart3_data(grade, target, reference_wc, w50, water_eff_chart, wc_min, wc_max, wc_step, rca_interval, constants)
chart4_df = generate_chart4_data(
    grade, fck, s, target, reference_wc, selected_wc, w50, slump, air_percent, sg_cement, sg_fa,
    sg_nca, sg_rca, rca_wa, ca_fraction, rca_interval, constants, min_cement, max_cement
)
chart6_df = generate_chart6_data(
    grade, fck, s, target, reference_wc, selected_wc, w50, slump, air_percent, sg_cement, sg_fa,
    sg_nca, sg_rca, rca_wa, rca_mc, ca_fraction, rca_interval, constants, min_cement, max_cement
)

tab1, tab2, tab3, tab4, tab5, tab6, tab_data = st.tabs([
    "Chart 1 Strength",
    "Chart 2 SRF",
    "Chart 3 Cement",
    "Chart 4 4-Axis",
    "Chart 5 RCA Quality",
    "Chart 6 Water",
    "Data + Download"
])

figures = {}

with tab1:
    st.subheader("Chart 1: Predictive Strength vs w/c Ratio")
    fig1 = plot_chart1(
        chart1_df,
        grade,
        target,
        reference_wc,
        selected_rca,
        selected_wc,
        water_eff_chart,
        slump,
        aggregate_size,
        zoom_enabled=enable_chart1_zoom,
        xlim_min=chart1_xmin,
        xlim_max=chart1_xmax,
        ylim_min=chart1_ymin,
        ylim_max=chart1_ymax,
    )
    figures["Chart1_Strength_vs_wc.png"] = fig1
    st.pyplot(fig1, use_container_width=not constant_graph_size)

with tab2:
    st.subheader("Chart 2: SRF vs w/c Ratio")
    fig2 = plot_chart2(chart2_df, grade, reference_wc, selected_rca, selected_wc)
    figures["Chart2_SRF_vs_wc.png"] = fig2
    st.pyplot(fig2, use_container_width=not constant_graph_size)

with tab3:
    st.subheader("Chart 3: Cement Compensation")
    ctab1, ctab2 = st.tabs(["Additional cement ΔC", "Compensated cement Ccomp"])
    with ctab1:
        fig3a = plot_chart3_delta(chart3_df, grade, target, selected_rca, selected_wc)
        figures["Chart3A_Delta_Cement_vs_wc.png"] = fig3a
        st.pyplot(fig3a, use_container_width=not constant_graph_size)
    with ctab2:
        fig3b = plot_chart3_ccomp(chart3_df, grade, selected_rca, selected_wc)
        figures["Chart3B_Compensated_Cement_vs_wc.png"] = fig3b
        st.pyplot(fig3b, use_container_width=not constant_graph_size)

with tab4:
    st.subheader("Chart 4: Dynamic Grade-Wise 4-Axis Chart")
    fig4 = plot_chart4(chart4_df, grade, selected_rca, selected_wc, slump, aggregate_size, wa_axis_min, wa_axis_max)
    figures["Chart4_Dynamic_4Axis.png"] = fig4
    st.pyplot(fig4, use_container_width=not constant_graph_size)

with tab5:
    st.subheader("Chart 5: RCA Quality Check")
    fig5, quality, note = plot_chart5(sg_rca, rca_wa, chart5_xmin, chart5_xmax, chart5_ymin, chart5_ymax, show_reference_points)
    figures["Chart5_RCA_Quality_Check.png"] = fig5
    st.pyplot(fig5, use_container_width=not constant_graph_size)
    c1, c2, c3 = st.columns(3)
    c1.metric("RCA SG", f"{sg_rca:.2f}")
    c2.metric("RCA WA", f"{rca_wa:.3f}%")
    c3.metric("Quality class", quality)
    st.info(note)

with tab6:
    st.subheader("Chart 6: Extra Absorption Water vs RCA Replacement")
    fig6 = plot_chart6(chart6_df, grade, selected_rca, selected_wc, rca_wa, rca_mc)
    figures["Chart6_Extra_Absorption_Water.png"] = fig6
    st.pyplot(fig6, use_container_width=not constant_graph_size)

with tab_data:
    st.subheader("Selected design summary")

    final_quantity_table, final_ratio_table, final_mix_summary = make_final_mix_table(
        grade=grade,
        fck=fck,
        target=target,
        selected_wc=selected_wc,
        selected_rca=selected_rca,
        w50=w50,
        chart4_df=chart4_df,
        chart6_df=chart6_df,
        rca_wa=rca_wa,
        rca_mc=rca_mc,
        sg_cement=sg_cement,
        sg_fa=sg_fa,
        sg_nca=sg_nca,
        sg_rca=sg_rca
    )

    st.markdown("### Final selected mix quantities")
    st.dataframe(final_quantity_table, use_container_width=True)

    st.markdown("### Final mix ratio")
    st.dataframe(final_ratio_table, use_container_width=True)

    st.success(f"Recommended RCA split mix ratio: **Cement : FA : NCA : RCA = {final_mix_summary['Mix ratio C:FA:NCA:RCA']}**")

    selected_c = interpolate_value(chart4_df, selected_rca, "Compensated cement Ccomp (kg/m3)")
    selected_rca_mass = interpolate_value(chart4_df, selected_rca, "RCA aggregate content (kg/m3)")
    selected_nca = interpolate_value(chart4_df, selected_rca, "NCA aggregate content (kg/m3)")
    selected_fa = interpolate_value(chart4_df, selected_rca, "Fine aggregate content (kg/m3)")
    selected_extra = interpolate_value(chart6_df, selected_rca, "Extra absorption water (kg/m3)")
    selected_batching = interpolate_value(chart6_df, selected_rca, "Batching water to be added (kg/m3)")

    cols = st.columns(6)
    cols[0].metric("Ccomp", f"{selected_c:.2f} kg/m³")
    cols[1].metric("RCA", f"{selected_rca_mass:.2f} kg/m³")
    cols[2].metric("NCA", f"{selected_nca:.2f} kg/m³")
    cols[3].metric("FA", f"{selected_fa:.2f} kg/m³")
    cols[4].metric("Extra water", f"{selected_extra:.2f} kg/m³")
    cols[5].metric("Batching water", f"{selected_batching:.2f} kg/m³")

    with st.expander("Chart 1 data"):
        st.dataframe(chart1_df, use_container_width=True)
    with st.expander("Chart 2 data"):
        st.dataframe(chart2_df, use_container_width=True)
    with st.expander("Chart 3 data"):
        st.dataframe(chart3_df, use_container_width=True)
    with st.expander("Chart 4 data"):
        st.dataframe(chart4_df, use_container_width=True)
    with st.expander("Chart 6 data"):
        st.dataframe(chart6_df, use_container_width=True)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for filename, fig in figures.items():
            z.writestr(f"{grade}_{filename}", fig_to_png(fig))

        z.writestr(f"{grade}_Chart1_Strength_data.csv", chart1_df.to_csv(index=False))
        z.writestr(f"{grade}_Chart2_SRF_data.csv", chart2_df.to_csv(index=False))
        z.writestr(f"{grade}_Chart3_Cement_Compensation_data.csv", chart3_df.to_csv(index=False))
        z.writestr(f"{grade}_Chart4_Dynamic_4Axis_data.csv", chart4_df.to_csv(index=False))
        z.writestr(f"{grade}_Chart6_Extra_Water_data.csv", chart6_df.to_csv(index=False))

        summary_df = pd.DataFrame([{
            "Grade": grade,
            "fck (MPa)": fck,
            "Target mean strength (MPa)": target,
            "Selected w/c": selected_wc,
            "Selected RCA (%)": selected_rca,
            "Slump (mm)": slump,
            "W50 base water (kg/m3)": w50,
            "W_eff slump-corrected water (kg/m3)": water_eff_chart,
            "Ccomp (kg/m3)": selected_c,
            "RCA (kg/m3)": selected_rca_mass,
            "NCA (kg/m3)": selected_nca,
            "FA (kg/m3)": selected_fa,
            "RCA WA (%)": rca_wa,
            "RCA SG": sg_rca,
            "Extra absorption water (kg/m3)": selected_extra,
            "Batching water (kg/m3)": selected_batching,
            "RCA quality": quality,
        }])
        z.writestr(f"{grade}_Selected_Design_Summary.csv", summary_df.to_csv(index=False))
        z.writestr(f"{grade}_Final_Selected_Mix_Quantities.csv", final_quantity_table.to_csv(index=False))
        z.writestr(f"{grade}_Final_Mix_Ratio.csv", final_ratio_table.to_csv(index=False))

    zip_buffer.seek(0)

    st.download_button(
        "Download all charts + CSV data",
        data=zip_buffer,
        file_name=f"{grade}_RCA_Mix_Design_Charts_All.zip",
        mime="application/zip"
    )
