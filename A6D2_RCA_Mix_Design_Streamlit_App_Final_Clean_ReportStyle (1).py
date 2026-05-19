
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import streamlit as st


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="A6-D2 RCA Mix Design App",
    page_icon="🏗️",
    layout="wide"
)

# ============================================================
# OPTIONAL BACKGROUND VIDEO
# ============================================================

def add_local_background_video(video_file="background.mp4", overlay_opacity=0.55):
    """
    Place a file named background.mp4 in the same GitHub folder as this app.
    The app will run normally even if the video file is missing.

    Recommended video:
    - MP4 format
    - 8 to 15 seconds loop
    - compressed below 20 to 30 MB for Streamlit Cloud
    """
    import base64
    from pathlib import Path

    video_path = Path(video_file)

    if not video_path.exists():
        return

    video_bytes = video_path.read_bytes()
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
            z-index: -3;
            object-fit: cover;
        }}

        #bg-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, {overlay_opacity});
            z-index: -2;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        [data-testid="stSidebar"] {{
            background: rgba(14, 17, 23, 0.90);
            backdrop-filter: blur(6px);
        }}

        .block-container {{
            background: rgba(0, 0, 0, 0.66);
            border-radius: 18px;
            padding: 2rem 2.2rem 2.2rem 2.2rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: #ffffff;
        }}

        .stDataFrame, .stTable {{
            background: rgba(255, 255, 255, 0.04);
            border-radius: 10px;
        }}
        </style>

        <video autoplay muted loop playsinline id="bg-video">
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
        <div id="bg-overlay"></div>
        """,
        unsafe_allow_html=True
    )


add_local_background_video("background.mp4", overlay_opacity=0)

# ============================================================
# GRADE DATA
# ============================================================

GRADE_INFO = {
    "M20": {"fck": 20.0, "s": 4.0, "target": 26.60, "ref_wc": 0.50, "wc_min": 0.40, "wc_max": 0.65},
    "M25": {"fck": 25.0, "s": 4.0, "target": 31.60, "ref_wc": 0.45, "wc_min": 0.35, "wc_max": 0.60},
    "M30": {"fck": 30.0, "s": 5.0, "target": 38.25, "ref_wc": 0.42, "wc_min": 0.32, "wc_max": 0.55},
    "M35": {"fck": 35.0, "s": 5.0, "target": 43.25, "ref_wc": 0.40, "wc_min": 0.30, "wc_max": 0.52},
    "M40": {"fck": 40.0, "s": 5.0, "target": 48.25, "ref_wc": 0.38, "wc_min": 0.28, "wc_max": 0.50},
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def target_mean_strength(fck, s):
    return fck + 1.65 * s


def slump_corrected_water(w50, slump):
    """
    IS 10262-style slump correction:
    3% increase in water content for every 25 mm increase in slump above 50 mm.
    """
    return w50 * (1.0 + 0.03 * ((slump - 50.0) / 25.0))


def control_curve(wc, target, wc_anchor, slope, curvature):
    """
    A6-D2 control curve.
    0% RCA = control.
    At wc_anchor, control strength = target mean strength.
    """
    t = wc - wc_anchor
    return target - slope * t - curvature * (t ** 2)


def a6d2_srf(wc, rca_percent, severity_base, severity_wc, severity_wc2, nonlinear_factor):
    """
    Constrained A6-D2 strength reduction factor.
    SRF = 1 at 0% RCA and reduces with RCA replacement.
    """
    r = rca_percent / 100.0
    x = wc - 0.50
    severity = severity_base + severity_wc * x + severity_wc2 * (x ** 2)
    nonlinear = nonlinear_factor * (r ** 2)
    return float(np.clip(1.0 - severity * r - nonlinear, 0.50, 1.00))


def predicted_strength(wc, rca_percent, target, wc_anchor, slope, curvature,
                       severity_base, severity_wc, severity_wc2, nonlinear_factor):
    f0 = control_curve(wc, target, wc_anchor, slope, curvature)
    srf = a6d2_srf(wc, rca_percent, severity_base, severity_wc, severity_wc2, nonlinear_factor)
    return f0 * srf


def cement_compensation(water_eff, wc, target, f_pred, min_cement, max_cement):
    cbase = water_eff / wc
    if f_pred <= 0 or f_pred >= target:
        ccomp = cbase
    else:
        ccomp = cbase * target / f_pred

    ccomp = float(np.clip(ccomp, min_cement, max_cement))
    delta_c = ccomp - cbase
    final_wc = water_eff / ccomp if ccomp else np.nan
    return cbase, ccomp, delta_c, final_wc


def aggregate_water_correction(mass, wa, mc):
    """
    Positive value means water to be added.
    Negative value means water to be deducted due to free surface moisture.
    """
    return mass * (wa - mc) / 100.0


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def interp_rca(df, rca, col):
    return float(np.interp(rca, df["RCA replacement (%)"], df[col]))


def interp_wc(df, rca, wc, col):
    sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
    return float(np.interp(wc, sub["w/c ratio"], sub[col]))


# ============================================================
# DATA GENERATION
# ============================================================

def generate_strength_df(grade, target, wc_anchor, wc_min, wc_max, wc_step, rca_step,
                         water_eff, slope, curvature, severity_base, severity_wc,
                         severity_wc2, nonlinear_factor):
    rows = []
    wc_values = np.round(np.arange(wc_min, wc_max + 0.0001, wc_step), 3)

    for rca in range(0, 101, rca_step):
        for wc in wc_values:
            f0 = control_curve(wc, target, wc_anchor, slope, curvature)
            srf = a6d2_srf(wc, rca, severity_base, severity_wc, severity_wc2, nonlinear_factor)
            fpred = f0 * srf
            rows.append({
                "Grade": grade,
                "w/c ratio": wc,
                "RCA replacement (%)": rca,
                "Control strength 0% RCA (MPa)": f0,
                "A6-D2 SRF": srf,
                "Predicted strength (MPa)": fpred,
                "Effective water W_eff (kg/m3)": water_eff,
                "Cement content W_eff/(w/c) (kg/m3)": water_eff / wc,
            })

    return pd.DataFrame(rows)


def generate_cement_df(strength_df, target, water_eff, min_cement, max_cement):
    rows = []

    for _, row in strength_df.iterrows():
        cbase, ccomp, delta_c, final_wc = cement_compensation(
            water_eff=water_eff,
            wc=row["w/c ratio"],
            target=target,
            f_pred=row["Predicted strength (MPa)"],
            min_cement=min_cement,
            max_cement=max_cement,
        )

        d = row.to_dict()
        d.update({
            "Target mean strength (MPa)": target,
            "Base cement Cbase=W_eff/(w/c) (kg/m3)": cbase,
            "Compensated cement Ccomp (kg/m3)": ccomp,
            "Additional cement Delta C (kg/m3)": delta_c,
            "Effective w/c after compensation": final_wc,
        })
        rows.append(d)

    return pd.DataFrame(rows)


def generate_mix_df(
    grade, fck, s, target, selected_wc, water_eff, air_percent,
    sg_cement, sg_fa, sg_nca, sg_rca,
    wa_fa, mc_fa, wa_nca, mc_nca, wa_rca, mc_rca,
    ca_fraction, rca_step, min_cement, max_cement,
    wc_anchor, slope, curvature, severity_base, severity_wc, severity_wc2, nonlinear_factor
):
    """
    This is the corrected IS 10262 + A6-D2 mix calculation.

    Important:
    VCA,total is NOT entered directly.
    It is calculated as:
    Vagg = 1 - (Vc + Vw + Vair)
    VCA,total = CA_fraction × Vagg
    VFA = (1 - CA_fraction) × Vagg
    """
    rows = []

    for rca in range(0, 101, rca_step):
        f0 = control_curve(selected_wc, target, wc_anchor, slope, curvature)
        srf = a6d2_srf(selected_wc, rca, severity_base, severity_wc, severity_wc2, nonlinear_factor)
        f_pred = f0 * srf

        cbase, ccomp, delta_c, final_wc = cement_compensation(
            water_eff=water_eff,
            wc=selected_wc,
            target=target,
            f_pred=f_pred,
            min_cement=min_cement,
            max_cement=max_cement,
        )

        # Absolute volume calculation
        Vc = ccomp / (sg_cement * 1000.0)
        Vw = water_eff / 1000.0
        Vair = air_percent / 100.0
        Vagg = 1.0 - (Vc + Vw + Vair)

        Vca_total = ca_fraction * Vagg
        Vfa = (1.0 - ca_fraction) * Vagg

        Vrca = (rca / 100.0) * Vca_total
        Vnca = Vca_total - Vrca

        Mfa = Vfa * sg_fa * 1000.0
        Mnca = Vnca * sg_nca * 1000.0
        Mrca = Vrca * sg_rca * 1000.0
        Mca = Mnca + Mrca
        Magg = Mfa + Mca

        # Final water corrections for aggregate absorption/moisture
        Wcorr_fa = aggregate_water_correction(Mfa, wa_fa, mc_fa)
        Wcorr_nca = aggregate_water_correction(Mnca, wa_nca, mc_nca)
        Wcorr_rca = aggregate_water_correction(Mrca, wa_rca, mc_rca)
        Wcorr_total = Wcorr_fa + Wcorr_nca + Wcorr_rca
        Wbatch = water_eff + Wcorr_total

        rows.append({
            "Grade": grade,
            "fck (MPa)": fck,
            "s (MPa)": s,
            "Target mean strength (MPa)": target,
            "Selected w/c": selected_wc,
            "RCA replacement (%)": rca,
            "Control strength 0% RCA (MPa)": f0,
            "A6-D2 SRF": srf,
            "Predicted strength (MPa)": f_pred,
            "Base cement Cbase (kg/m3)": cbase,
            "Compensated cement Ccomp (kg/m3)": ccomp,
            "Additional cement Delta C (kg/m3)": delta_c,
            "Effective w/c after compensation": final_wc,
            "Effective water W_eff (kg/m3)": water_eff,
            "Air content (%)": air_percent,
            "Air volume Vair": Vair,
            "Cement volume Vc": Vc,
            "Water volume Vw": Vw,
            "Total aggregate volume Vagg": Vagg,
            "CA volume fraction": ca_fraction,
            "Total coarse aggregate volume Vca_total": Vca_total,
            "Fine aggregate volume Vfa": Vfa,
            "NCA volume Vnca": Vnca,
            "RCA volume Vrca": Vrca,
            "Fine aggregate content (kg/m3)": Mfa,
            "NCA aggregate content (kg/m3)": Mnca,
            "RCA aggregate content (kg/m3)": Mrca,
            "Total coarse aggregate (kg/m3)": Mca,
            "Total aggregate content (kg/m3)": Magg,
            "FA water absorption (%)": wa_fa,
            "FA moisture content (%)": mc_fa,
            "NCA water absorption (%)": wa_nca,
            "NCA moisture content (%)": mc_nca,
            "RCA water absorption (%)": wa_rca,
            "RCA moisture content (%)": mc_rca,
            "RCA specific gravity": sg_rca,
            "FA water correction (kg/m3)": Wcorr_fa,
            "NCA water correction (kg/m3)": Wcorr_nca,
            "RCA water correction (kg/m3)": Wcorr_rca,
            "Total aggregate water correction (kg/m3)": Wcorr_total,
            "Batching water to be added (kg/m3)": Wbatch,
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def plot_strength(df, grade, target, selected_rca, selected_wc, water_eff, slump, agg_size):
    fig, ax = plt.subplots(figsize=(12.8, 7.9))

    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["Predicted strength (MPa)"],
                marker="o", markersize=3.5, linewidth=lw, label=f"{rca}% RCA")

    y = interp_wc(df, selected_rca, selected_wc, "Predicted strength (MPa)")

    ax.axhline(target, linestyle="--", linewidth=1.6, label=f"{grade} target mean = {target:.2f} MPa")
    ax.axvline(selected_wc, linestyle=":", linewidth=1.6, label=f"{grade} selected w/c = {selected_wc:.2f}")
    ax.scatter([selected_wc], [y], marker="*", s=220, zorder=6)

    ax.annotate(
        f"{grade}, {selected_rca}% RCA\nw/c = {selected_wc:.3f}\nf = {y:.2f} MPa",
        xy=(selected_wc, y),
        xytext=(selected_wc + 0.025, y + 0.25),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
        fontsize=9
    )

    ax.set_title(
        f"{grade} Predictive Strength vs w/c Ratio\n"
        f"A6-D2: 0% RCA = control; {slump:.0f} mm slump, "
        f"{agg_size:.0f} mm aggregate, W_eff = {water_eff:.2f} kg/m³",
        fontsize=13
    )
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Predicted 28-day compressive strength (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def plot_srf(df, grade, selected_rca, selected_wc):
    fig, ax = plt.subplots(figsize=(12.8, 7.9))

    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["A6-D2 SRF"],
                marker="o", markersize=3.5, linewidth=lw, label=f"{rca}% RCA")

    srf = interp_wc(df, selected_rca, selected_wc, "A6-D2 SRF")

    ax.axhline(1.0, linestyle="--", linewidth=1.6, label="0% RCA reference SRF = 1.0")
    ax.axvline(selected_wc, linestyle=":", linewidth=1.6, label=f"{grade} selected w/c = {selected_wc:.2f}")
    ax.scatter([selected_wc], [srf], marker="*", s=220, zorder=6)

    ax.annotate(
        f"{grade}, {selected_rca}% RCA\nw/c = {selected_wc:.3f}\nSRF = {srf:.3f}",
        xy=(selected_wc, srf),
        xytext=(selected_wc + 0.025, srf + 0.010),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
        fontsize=9
    )

    ax.set_title(f"{grade} Chart 2: Strength Reduction Factor vs w/c Ratio\nSRF = f_RCA / f_0%RCA")
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Strength Reduction Factor, SRF")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def plot_cement(df, grade, target, selected_rca, selected_wc):
    fig, ax = plt.subplots(figsize=(12.8, 7.9))

    for rca in sorted(df["RCA replacement (%)"].unique()):
        sub = df[df["RCA replacement (%)"] == rca].sort_values("w/c ratio")
        lw = 2.8 if rca == selected_rca else 1.8
        ax.plot(sub["w/c ratio"], sub["Additional cement Delta C (kg/m3)"],
                marker="o", markersize=3.5, linewidth=lw, label=f"{rca}% RCA")

    y = interp_wc(df, selected_rca, selected_wc, "Additional cement Delta C (kg/m3)")
    c = interp_wc(df, selected_rca, selected_wc, "Compensated cement Ccomp (kg/m3)")
    f = interp_wc(df, selected_rca, selected_wc, "Predicted strength (MPa)")

    ax.axvline(selected_wc, linestyle=":", linewidth=1.6, label=f"Selected w/c = {selected_wc:.2f}")
    ax.scatter([selected_wc], [y], marker="*", s=220)

    ax.annotate(
        f"{grade}, {selected_rca}% RCA\nw/c = {selected_wc:.3f}\nf = {f:.2f} MPa\n"
        f"ΔC = {y:.2f} kg/m³\nCcomp = {c:.2f} kg/m³",
        xy=(selected_wc, y),
        xytext=(selected_wc + 0.025, y + 8),
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
        fontsize=9
    )

    ax.set_title(
        f"{grade} Chart 3: Cement Compensation vs w/c Ratio\n"
        f"Ccomp = Cbase × target / fpred, target = {target:.2f} MPa"
    )
    ax.set_xlabel("Water-cement ratio")
    ax.set_ylabel("Additional cement required, ΔC (kg/m³)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_water(df, grade, selected_rca, selected_wc, wa_rca, mc_rca):
    x = df["RCA replacement (%)"].values
    extra = df["RCA water correction (kg/m3)"].values
    total_corr = df["Total aggregate water correction (kg/m3)"].values
    rca_mass = df["RCA aggregate content (kg/m3)"].values

    selected_extra = interp_rca(df, selected_rca, "RCA water correction (kg/m3)")
    selected_total_corr = interp_rca(df, selected_rca, "Total aggregate water correction (kg/m3)")
    selected_rca_mass = interp_rca(df, selected_rca, "RCA aggregate content (kg/m3)")
    selected_batch = interp_rca(df, selected_rca, "Batching water to be added (kg/m3)")

    fig, ax1 = plt.subplots(figsize=(12.5, 8))

    l1 = ax1.plot(x, extra, marker="o", linewidth=2.5, color="#0057c2",
                  label="RCA water correction (kg/m³)")
    l2 = ax1.plot(x, total_corr, marker="s", linestyle="--", linewidth=2.0, color="#ff7f0e",
                  label="Total aggregate water correction (kg/m³)")

    ax1.set_xlabel("RCA replacement (%)")
    ax1.set_ylabel("Water correction (kg/m³)")
    ax1.set_xlim(-2, 102)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l3 = ax2.plot(x, rca_mass, marker="^", linewidth=2.4, color="green",
                  label="RCA aggregate content (kg/m³)")
    ax2.set_ylabel("RCA aggregate content (kg/m³)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, max(100, np.nanmax(rca_mass) * 1.15))

    ax1.axvline(selected_rca, linestyle=":", color="dodgerblue")
    ax1.scatter([selected_rca], [selected_extra], marker="*", s=200, color="#0057c2")
    ax2.scatter([selected_rca], [selected_rca_mass], marker="*", s=200, color="green")

    ax1.text(
        0.985, 0.055,
        f"Selected {grade}, {selected_rca}% RCA\n"
        f"w/c = {selected_wc:.3f}\n"
        f"RCA = {selected_rca_mass:.2f} kg/m³\n"
        f"RCA WA = {wa_rca:.3f}%\n"
        f"RCA MC = {mc_rca:.3f}%\n"
        f"RCA correction = {selected_extra:.2f} kg/m³\n"
        f"Total correction = {selected_total_corr:.2f} kg/m³\n"
        f"Batching water = {selected_batch:.2f} kg/m³",
        transform=ax1.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black")
    )

    lines = l1 + l2 + l3
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")

    ax1.set_title(
        f"{grade} Chart 6: Aggregate Water Correction vs RCA Replacement\n"
        f"Wbatch = W_eff + FA correction + NCA correction + RCA correction"
    )
    fig.tight_layout()
    return fig


def plot_4axis(df, grade, selected_rca, selected_wc, slump, agg_size):
    x = df["RCA replacement (%)"].values
    cement = df["Compensated cement Ccomp (kg/m3)"].values
    rca = df["RCA aggregate content (kg/m3)"].values
    wa = df["RCA water absorption (%)"].values
    sg = df["RCA specific gravity"].values

    sc = interp_rca(df, selected_rca, "Compensated cement Ccomp (kg/m3)")
    sr = interp_rca(df, selected_rca, "RCA aggregate content (kg/m3)")
    swa = interp_rca(df, selected_rca, "RCA water absorption (%)")
    ssg = interp_rca(df, selected_rca, "RCA specific gravity")

    fig, ax1 = plt.subplots(figsize=(13, 8))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()

    ax3.spines["left"].set_position(("axes", -0.13))
    ax3.yaxis.set_label_position("left")
    ax3.yaxis.set_ticks_position("left")
    ax3.spines["left"].set_visible(True)
    ax3.spines["right"].set_visible(False)

    ax4.spines["right"].set_position(("axes", 1.13))

    l1 = ax1.plot(x, cement, marker="o", linewidth=2.5, color="#0057c2",
                  label="Compensated cement (kg/m³)")
    l2 = ax2.plot(x, rca, marker="^", linewidth=2.5, color="green",
                  label="RCA aggregate content (kg/m³)")
    l3 = ax3.plot(x, wa, marker="s", linestyle="--", linewidth=2.0, color="#ff7f0e",
                  label="RCA water absorption (%)")
    l4 = ax4.plot(x, sg, marker="D", linestyle="--", linewidth=2.0, color="red",
                  label="RCA specific gravity")

    ax1.set_xlabel("RCA replacement (%)")
    ax1.set_ylabel("Compensated cement content (kg/m³)", color="#0057c2")
    ax2.set_ylabel("RCA aggregate content (kg/m³)", color="green")
    ax3.set_ylabel("RCA water absorption (%)", color="#ff7f0e")
    ax4.set_ylabel("RCA specific gravity", color="red")

    ax1.tick_params(axis="y", labelcolor="#0057c2")
    ax2.tick_params(axis="y", labelcolor="green")
    ax3.tick_params(axis="y", labelcolor="#ff7f0e")
    ax4.tick_params(axis="y", labelcolor="red")

    ax1.set_xlim(-2, 102)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    ax1.set_ylim(min(cement) - 8, max(cement) + 8)
    ax2.set_ylim(0, max(100, np.nanmax(rca) * 1.15))
    ax3.set_ylim(max(0, min(wa) - 2), max(10, max(wa) + 2))
    ax4.set_ylim(min(sg) - 0.15, max(sg) + 0.15)

    for xi, yi in zip(x, cement):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#0057c2")

    for xi, yi in zip(x, rca):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color="green")

    ax3.text(100, swa, f" WA = {swa:.3f}%", color="#ff7f0e", ha="right", va="bottom")
    ax4.text(100, ssg, f" SG = {ssg:.3f}", color="red", ha="right", va="bottom")

    ax1.axvline(selected_rca, linestyle=":", color="dodgerblue")
    ax1.scatter([selected_rca], [sc], marker="*", s=200, color="#0057c2")
    ax2.scatter([selected_rca], [sr], marker="*", s=200, color="green")

    ax1.text(
        0.985, 0.055,
        f"Selected {grade}, {selected_rca}% RCA\n"
        f"C = {sc:.2f} kg/m³\n"
        f"RCA = {sr:.2f} kg/m³\n"
        f"WA = {swa:.3f}%\n"
        f"SG = {ssg:.3f}",
        transform=ax1.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black")
    )

    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=9)

    ax1.set_title(
        f"{grade} Dynamic RCA Combined 4-Axis Chart\n"
        f"Basis: selected w/c = {selected_wc:.3f}, slump = {slump:.0f} mm, {agg_size:.0f} mm aggregate",
        fontsize=15, fontweight="bold"
    )
    fig.tight_layout(rect=[0.08, 0.04, 0.92, 0.96])
    return fig


def classify_rca_quality(sg, wa):
    if sg >= 2.50 and wa <= 3.0:
        return "Good RCA", "Suitable RCA quality; low absorption and good specific gravity."
    if sg >= 2.30 and wa <= 6.0:
        return "Moderate RCA", "Usable RCA; absorption correction and trial mix validation are required."
    if sg >= 2.10 and wa <= 10.0:
        return "Poor / High absorption RCA", "Use with caution; higher water correction and trial mix validation are necessary."
    return "Not recommended", "Very low SG or very high absorption; avoid unless separately validated."


def plot_quality(sg, wa, xmin, xmax, ymin, ymax, show_ref):
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.set_facecolor("white")

    ax.add_patch(Rectangle((2.10, 0), xmax - 2.10, min(10, ymax), facecolor="#ffd6d6", edgecolor="none", alpha=0.35))
    ax.add_patch(Rectangle((2.30, 0), xmax - 2.30, min(6, ymax), facecolor="#ffe9b8", edgecolor="none", alpha=0.55))
    ax.add_patch(Rectangle((2.50, 0), xmax - 2.50, min(3, ymax), facecolor="#cfead1", edgecolor="none", alpha=0.75))

    for xv, col in [(2.10, "red"), (2.30, "orange"), (2.50, "green")]:
        ax.axvline(xv, linestyle="--", color=col)

    for yv, col in [(3, "green"), (6, "orange"), (10, "red")]:
        ax.axhline(yv, linestyle="--", color=col)

    if show_ref:
        ref_data = pd.DataFrame({
            "Name": ["Typical NCA", "Selected RCA", "High WA RCA"],
            "SG": [2.70, sg, 2.20],
            "WA": [0.8, wa, 8.0],
        })
        ax.scatter(ref_data["SG"], ref_data["WA"], s=65, alpha=0.75, color="tab:blue")
        for _, row in ref_data.iterrows():
            ax.annotate(row["Name"], (row["SG"], row["WA"]), textcoords="offset points", xytext=(10, 10), fontsize=9)

    quality, note = classify_rca_quality(sg, wa)
    ax.scatter([sg], [wa], marker="*", s=300, color="blue", edgecolor="black")
    ax.axvline(sg, linestyle=":", color="blue", alpha=0.65)
    ax.axhline(wa, linestyle=":", color="blue", alpha=0.65)

    ax.text(
        0.985, 0.055,
        f"Selected RCA\nSG = {sg:.3f}\nWA = {wa:.3f}%\n{quality}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black")
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("RCA specific gravity")
    ax.set_ylabel("RCA water absorption (%)")
    ax.set_title("Chart 5: RCA Quality Check — Specific Gravity vs Water Absorption", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.28)

    ax.legend(handles=[
        Patch(facecolor="#cfead1", edgecolor="green", alpha=0.75, label="Good zone"),
        Patch(facecolor="#ffe9b8", edgecolor="orange", alpha=0.55, label="Moderate zone"),
        Patch(facecolor="#ffd6d6", edgecolor="red", alpha=0.35, label="Poor / high absorption zone"),
    ], loc="upper right")

    fig.tight_layout()
    return fig, quality, note


# ============================================================
# MANUAL DESIGN TABLES
# ============================================================

def manual_design_tables(
    grade, fck, s, target, selected_wc, selected_rca,
    w50, slump, water_eff, air_percent,
    sg_cement, sg_fa, sg_nca, sg_rca,
    wa_fa, mc_fa, wa_nca, mc_nca, wa_rca, mc_rca,
    ca_fraction, mix_df
):
    row = mix_df[mix_df["RCA replacement (%)"] == selected_rca].iloc[0]

    f0 = row["Control strength 0% RCA (MPa)"]
    srf = row["A6-D2 SRF"]
    fpred = row["Predicted strength (MPa)"]
    cbase = row["Base cement Cbase (kg/m3)"]
    ccomp = row["Compensated cement Ccomp (kg/m3)"]
    delta_c = row["Additional cement Delta C (kg/m3)"]
    final_wc = row["Effective w/c after compensation"]

    Vc = row["Cement volume Vc"]
    Vw = row["Water volume Vw"]
    Vair = row["Air volume Vair"]
    Vagg = row["Total aggregate volume Vagg"]
    Vca_total = row["Total coarse aggregate volume Vca_total"]
    Vfa = row["Fine aggregate volume Vfa"]
    Vnca = row["NCA volume Vnca"]
    Vrca = row["RCA volume Vrca"]

    Mfa = row["Fine aggregate content (kg/m3)"]
    Mnca = row["NCA aggregate content (kg/m3)"]
    Mrca = row["RCA aggregate content (kg/m3)"]
    Mca = row["Total coarse aggregate (kg/m3)"]

    Wcorr_fa = row["FA water correction (kg/m3)"]
    Wcorr_nca = row["NCA water correction (kg/m3)"]
    Wcorr_rca = row["RCA water correction (kg/m3)"]
    Wcorr_total = row["Total aggregate water correction (kg/m3)"]
    Wbatch = row["Batching water to be added (kg/m3)"]

    steps = pd.DataFrame([
        {
            "Step": "1. Target mean strength",
            "Formula": "fck' = fck + 1.65s",
            "Substitution": f"{fck:.2f} + 1.65 × {s:.2f}",
            "Result": f"{target:.2f} MPa",
            "Remark": "IS 10262 target strength basis",
        },
        {
            "Step": "2. Air content",
            "Formula": "Vair = air% / 100",
            "Substitution": f"{air_percent:.2f} / 100",
            "Result": f"{Vair:.4f} m³",
            "Remark": "Entrapped air for absolute volume method",
        },
        {
            "Step": "3. Assumed water-cement ratio",
            "Formula": "Assumed/check w/c",
            "Substitution": f"w/c = {selected_wc:.3f}",
            "Result": f"{selected_wc:.3f}",
            "Remark": "Must satisfy durability limit separately",
        },
        {
            "Step": "4. A6-D2 predicted compressive strength",
            "Formula": "fRAC = f0 × SRF",
            "Substitution": f"{f0:.2f} × {srf:.3f}",
            "Result": f"{fpred:.2f} MPa",
            "Remark": "Graph value proof for selected w/c and RCA%",
        },
        {
            "Step": "5. Slump correction",
            "Formula": "W_eff = W50 × [1 + 0.03 × (slump - 50)/25]",
            "Substitution": f"{w50:.2f} × [1 + 0.03 × ({slump:.0f} - 50)/25]",
            "Result": f"{water_eff:.2f} kg/m³",
            "Remark": "3% water increase per 25 mm slump increase",
        },
        {
            "Step": "6. Base cement content",
            "Formula": "Cbase = W_eff / (w/c)",
            "Substitution": f"{water_eff:.2f} / {selected_wc:.3f}",
            "Result": f"{cbase:.2f} kg/m³",
            "Remark": "Initial cement before A6-D2 compensation",
        },
        {
            "Step": "7. Cement compensation",
            "Formula": "Ccomp = Cbase × target / fpred",
            "Substitution": f"{cbase:.2f} × {target:.2f} / {fpred:.2f}",
            "Result": f"{ccomp:.2f} kg/m³",
            "Remark": "Applied when predicted strength is below target",
        },
        {
            "Step": "8. Final cement content and final w/c",
            "Formula": "Cfinal = Ccomp; (w/c)final = W_eff / Ccomp",
            "Substitution": f"{water_eff:.2f} / {ccomp:.2f}",
            "Result": f"C = {ccomp:.2f} kg/m³; final w/c = {final_wc:.3f}",
            "Remark": "Strength-governing effective w/c",
        },
        {
            "Step": "9. 4-axis graph values",
            "Formula": "Ccomp, RCA mass, RCA WA, RCA SG",
            "Substitution": f"RCA = {selected_rca:.0f}%",
            "Result": f"C = {ccomp:.2f} kg/m³; RCA = {Mrca:.2f} kg/m³; WA = {wa_rca:.3f}%; SG = {sg_rca:.3f}",
            "Remark": "Values shown in corrected 4-axis chart",
        },
        {
            "Step": "10. Total aggregate volume",
            "Formula": "Vagg = 1 - (Vc + Vw + Vair)",
            "Substitution": f"1 - ({Vc:.4f} + {Vw:.4f} + {Vair:.4f})",
            "Result": f"{Vagg:.4f} m³",
            "Remark": "Solved automatically, not manually entered",
        },
        {
            "Step": "11. Total coarse aggregate volume",
            "Formula": "VCA,total = CA fraction × Vagg",
            "Substitution": f"{ca_fraction:.3f} × {Vagg:.4f}",
            "Result": f"{Vca_total:.4f} m³",
            "Remark": "This replaces fixed input VCA,total",
        },
        {
            "Step": "12. Volumetric RCA replacement",
            "Formula": "VRCA = R × VCA,total; VNCA = (1 - R) × VCA,total",
            "Substitution": f"{selected_rca/100:.2f} × {Vca_total:.4f}; {1-selected_rca/100:.2f} × {Vca_total:.4f}",
            "Result": f"VRCA = {Vrca:.4f} m³; VNCA = {Vnca:.4f} m³",
            "Remark": "Replacement is by volume, then converted to mass",
        },
        {
            "Step": "13. Split CA into NCA and RCA masses",
            "Formula": "M = V × SG × 1000",
            "Substitution": f"RCA: {Vrca:.4f} × {sg_rca:.3f} × 1000; NCA: {Vnca:.4f} × {sg_nca:.3f} × 1000",
            "Result": f"RCA = {Mrca:.2f} kg/m³; NCA = {Mnca:.2f} kg/m³",
            "Remark": "Volume batching converted to kg/m³",
        },
        {
            "Step": "14. Fine aggregate quantity",
            "Formula": "VFA = (1 - CA fraction) × Vagg; MFA = VFA × SGFA × 1000",
            "Substitution": f"{1-ca_fraction:.3f} × {Vagg:.4f}; {Vfa:.4f} × {sg_fa:.3f} × 1000",
            "Result": f"VFA = {Vfa:.4f} m³; FA = {Mfa:.2f} kg/m³",
            "Remark": "Absolute volume balance",
        },
        {
            "Step": "15. Aggregate water corrections",
            "Formula": "Wcorr = M × (WA - MC) / 100",
            "Substitution": (
                f"FA: {Mfa:.2f}×({wa_fa:.3f}-{mc_fa:.3f})/100; "
                f"NCA: {Mnca:.2f}×({wa_nca:.3f}-{mc_nca:.3f})/100; "
                f"RCA: {Mrca:.2f}×({wa_rca:.3f}-{mc_rca:.3f})/100"
            ),
            "Result": f"FA = {Wcorr_fa:.2f}; NCA = {Wcorr_nca:.2f}; RCA = {Wcorr_rca:.2f} kg/m³",
            "Remark": "Positive = add water; negative = deduct water",
        },
        {
            "Step": "16. Final batching water",
            "Formula": "Wbatch = W_eff + Wcorr_FA + Wcorr_NCA + Wcorr_RCA",
            "Substitution": f"{water_eff:.2f} + {Wcorr_fa:.2f} + {Wcorr_nca:.2f} + {Wcorr_rca:.2f}",
            "Result": f"{Wbatch:.2f} kg/m³",
            "Remark": "Final water to be added during batching",
        },
    ])

    final = pd.DataFrame([
        {"Material / parameter": "Cement", "Value": ccomp, "Unit": "kg/m³"},
        {"Material / parameter": "Effective water", "Value": water_eff, "Unit": "kg/m³"},
        {"Material / parameter": "Total aggregate water correction", "Value": Wcorr_total, "Unit": "kg/m³"},
        {"Material / parameter": "Batching water", "Value": Wbatch, "Unit": "kg/m³"},
        {"Material / parameter": "Fine aggregate", "Value": Mfa, "Unit": "kg/m³"},
        {"Material / parameter": "Natural coarse aggregate", "Value": Mnca, "Unit": "kg/m³"},
        {"Material / parameter": "Recycled coarse aggregate", "Value": Mrca, "Unit": "kg/m³"},
        {"Material / parameter": "Total coarse aggregate", "Value": Mca, "Unit": "kg/m³"},
        {"Material / parameter": "Predicted strength before compensation", "Value": fpred, "Unit": "MPa"},
        {"Material / parameter": "Target mean strength", "Value": target, "Unit": "MPa"},
        {"Material / parameter": "Effective w/c after compensation", "Value": final_wc, "Unit": "-"},
        {"Material / parameter": "Batching water/cement ratio", "Value": Wbatch / ccomp, "Unit": "-"},
        {"Material / parameter": "Solved VCA,total", "Value": Vca_total, "Unit": "m³"},
    ])

    ratio = pd.DataFrame([
        {
            "Ratio type": "RCA split ratio",
            "Ratio": f"Cement : FA : NCA : RCA = 1 : {Mfa/ccomp:.2f} : {Mnca/ccomp:.2f} : {Mrca/ccomp:.2f}",
            "Use": "Recommended final reporting format",
        },
        {
            "Ratio type": "Conventional ratio",
            "Ratio": f"Cement : FA : Total CA = 1 : {Mfa/ccomp:.2f} : {Mca/ccomp:.2f}",
            "Use": "Conventional concrete mix reporting",
        },
        {
            "Ratio type": "Water-cement basis",
            "Ratio": f"Effective w/c = {final_wc:.3f}; Batching w/c = {Wbatch/ccomp:.3f}",
            "Use": "Batching water includes aggregate moisture correction",
        },
    ])

    return steps, final, ratio


def final_tables(grade, fck, target, selected_wc, selected_rca, mix_df):
    row = mix_df[mix_df["RCA replacement (%)"] == selected_rca].iloc[0]

    c = row["Compensated cement Ccomp (kg/m3)"]
    w = row["Effective water W_eff (kg/m3)"]
    bw = row["Batching water to be added (kg/m3)"]
    fa = row["Fine aggregate content (kg/m3)"]
    nca = row["NCA aggregate content (kg/m3)"]
    rca = row["RCA aggregate content (kg/m3)"]
    tca = row["Total coarse aggregate (kg/m3)"]
    f = row["Predicted strength (MPa)"]
    ewcr = row["Effective w/c after compensation"]
    vca = row["Total coarse aggregate volume Vca_total"]

    qty = pd.DataFrame([
        ["Concrete grade", grade, "-"],
        ["fck", f"{fck:.2f}", "MPa"],
        ["Target mean strength", f"{target:.2f}", "MPa"],
        ["Selected RCA replacement", f"{selected_rca:.0f}", "%"],
        ["Selected design w/c", f"{selected_wc:.3f}", "-"],
        ["Predicted strength", f"{f:.2f}", "MPa"],
        ["Compensated cement", f"{c:.2f}", "kg/m³"],
        ["Effective water", f"{w:.2f}", "kg/m³"],
        ["Batching water", f"{bw:.2f}", "kg/m³"],
        ["Fine aggregate", f"{fa:.2f}", "kg/m³"],
        ["NCA", f"{nca:.2f}", "kg/m³"],
        ["RCA", f"{rca:.2f}", "kg/m³"],
        ["Total coarse aggregate", f"{tca:.2f}", "kg/m³"],
        ["Solved VCA,total", f"{vca:.4f}", "m³"],
        ["Effective w/c after compensation", f"{ewcr:.3f}", "-"],
        ["Batching water/cement ratio", f"{bw/c:.3f}", "-"],
    ], columns=["Item", "Value", "Unit"])

    ratio = pd.DataFrame([
        ["Conventional ratio", f"Cement : FA : Total CA = 1 : {fa/c:.2f} : {tca/c:.2f}", "Total CA = NCA + RCA"],
        ["RCA split ratio", f"Cement : FA : NCA : RCA = 1 : {fa/c:.2f} : {nca/c:.2f} : {rca/c:.2f}", "Recommended for RCA mix reporting"],
        ["Water-cement basis", f"Effective w/c = {ewcr:.3f}; Batching w/c = {bw/c:.3f}", "Batching water includes aggregate moisture correction"],
    ], columns=["Mix ratio type", "Ratio", "Notes"])

    summary = {
        "Cement": c,
        "Water": w,
        "Batching water": bw,
        "FA": fa,
        "NCA": nca,
        "RCA": rca,
        "Solved VCA,total": vca,
        "Ratio": f"1 : {fa/c:.2f} : {nca/c:.2f} : {rca/c:.2f}",
    }

    return qty, ratio, summary



def display_manual_solution_markdown(manual_steps, manual_final, manual_ratio, mix_df, selected_rca):
    """Display manual mix design in report-style format instead of only table format."""
    row = mix_df[mix_df["RCA replacement (%)"] == selected_rca].iloc[0]

    def val(label):
        return row[label]

    # Extract common values
    fck = float(row["fck (MPa)"])
    s = float(row["s (MPa)"])
    target = float(row["Target mean strength (MPa)"])
    wc = float(row["Selected w/c"])
    rca_pct = float(row["RCA replacement (%)"])
    f0 = float(row["Control strength 0% RCA (MPa)"])
    srf = float(row["A6-D2 SRF"])
    fpred = float(row["Predicted strength (MPa)"])
    cbase = float(row["Base cement Cbase (kg/m3)"])
    ccomp = float(row["Compensated cement Ccomp (kg/m3)"])
    delta_c = float(row["Additional cement Delta C (kg/m3)"])
    final_wc = float(row["Effective w/c after compensation"])
    water_eff = float(row["Effective water W_eff (kg/m3)"])
    Vair = float(row["Air volume Vair"])
    Vc = float(row["Cement volume Vc"])
    Vw = float(row["Water volume Vw"])
    Vagg = float(row["Total aggregate volume Vagg"])
    ca_fraction = float(row["CA volume fraction"])
    Vca_total = float(row["Total coarse aggregate volume Vca_total"])
    Vfa = float(row["Fine aggregate volume Vfa"])
    Vnca = float(row["NCA volume Vnca"])
    Vrca = float(row["RCA volume Vrca"])
    Mfa = float(row["Fine aggregate content (kg/m3)"])
    Mnca = float(row["NCA aggregate content (kg/m3)"])
    Mrca = float(row["RCA aggregate content (kg/m3)"])
    Mca = float(row["Total coarse aggregate (kg/m3)"])
    Wcorr_fa = float(row["FA water correction (kg/m3)"])
    Wcorr_nca = float(row["NCA water correction (kg/m3)"])
    Wcorr_rca = float(row["RCA water correction (kg/m3)"])
    Wbatch = float(row["Batching water to be added (kg/m3)"])
    sg_fa = Mfa / (Vfa * 1000) if Vfa else np.nan
    sg_nca = Mnca / (Vnca * 1000) if Vnca else np.nan
    sg_rca = float(row["RCA specific gravity"])
    wa_fa = float(row["FA water absorption (%)"])
    mc_fa = float(row["FA moisture content (%)"])
    wa_nca = float(row["NCA water absorption (%)"])
    mc_nca = float(row["NCA moisture content (%)"])
    wa_rca = float(row["RCA water absorption (%)"])
    mc_rca = float(row["RCA moisture content (%)"])

    st.markdown("## Manual Mix Design Calculation")
    st.markdown("---")

    st.markdown("### Step 1: Target strength")
    st.latex(r"f'_{ck}=f_{ck}+1.65s")
    st.markdown(f"For the selected grade:")
    st.latex(rf"f'_{{ck}}={fck:.0f}+1.65({s:.0f})={target:.2f}\;MPa")
    st.markdown("---")

    st.markdown("### Step 2: Air content")
    st.latex(r"V_{air}=\frac{\text{Air content}}{100}")
    st.latex(rf"V_{{air}}={Vair:.4f}\;m^3")
    st.markdown("---")

    st.markdown("### Step 3: Assumed water-cement ratio")
    st.markdown("For the trial/design chart check:")
    st.latex(rf"w/c={wc:.3f}, \qquad RCA={rca_pct:.0f}\%")
    st.markdown("---")

    st.markdown("### Step 4: Predicted compressive strength from A6-D2")
    st.markdown("A6-D2 uses:")
    st.latex(r"f_{RAC}=f_{0\%RCA}\times SRF_{A6-D2}")
    st.latex(rf"f_{{0\%RCA}}={f0:.2f}\;MPa")
    st.latex(rf"SRF_{{A6-D2}}={srf:.3f}")
    st.latex(rf"f_{{pred}}={f0:.2f}\times{srf:.3f}={fpred:.2f}\;MPa")
    st.markdown("---")

    st.markdown("### Step 5: Slump-corrected water content")
    st.markdown("Using IS 10262-style correction: **3% increase in water for every 25 mm increase in slump above 50 mm**.")
    # Get W50 and slump from manual_steps substitution where possible
    slump_row = manual_steps[manual_steps["Step"].str.contains("Slump correction", na=False)].iloc[0]
    st.latex(r"W_{eff}=W_{50}\left[1+0.03\left(\frac{S-50}{25}\right)\right]")
    st.markdown(f"Substitution: `{slump_row['Substitution']}`")
    st.latex(rf"W_{{eff}}={water_eff:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Step 6: Base cement content")
    st.latex(r"C_{base}=\frac{W_{eff}}{w/c}")
    st.latex(rf"C_{{base}}=\frac{{{water_eff:.2f}}}{{{wc:.3f}}}={cbase:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Step 7: Cement compensation")
    st.markdown("Since the predicted strength is compared with the target mean strength:")
    st.latex(r"C_{comp}=C_{base}\times\frac{f'_{ck}}{f_{pred}}")
    st.latex(rf"C_{{comp}}={cbase:.2f}\times\frac{{{target:.2f}}}{{{fpred:.2f}}}={ccomp:.2f}\;kg/m^3")
    st.latex(rf"\Delta C=C_{{comp}}-C_{{base}}={ccomp:.2f}-{cbase:.2f}={delta_c:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Step 8: Final cement content and effective w/c")
    st.latex(rf"C_{{final}}=C_{{comp}}={ccomp:.2f}\;kg/m^3")
    st.latex(r"(w/c)_{final}=\frac{W_{eff}}{C_{final}}")
    st.latex(rf"(w/c)_{{final}}=\frac{{{water_eff:.2f}}}{{{ccomp:.2f}}}={final_wc:.3f}")
    st.markdown("---")

    st.markdown("### Step 9: Values from the 4-axis graph")
    st.markdown("At the selected RCA replacement level:")
    st.latex(rf"C={ccomp:.2f}\;kg/m^3,\qquad RCA={Mrca:.2f}\;kg/m^3")
    st.latex(rf"WA_{{RCA}}={wa_rca:.3f}\%,\qquad SG_{{RCA}}={sg_rca:.3f}")
    st.markdown("---")

    st.markdown("### Step 10: Total aggregate volume")
    st.markdown("Now the total aggregate volume is solved. It is **not manually entered**.")
    st.latex(r"V_{agg}=1-(V_c+V_w+V_{air})")
    st.latex(rf"V_{{agg}}=1-({Vc:.4f}+{Vw:.4f}+{Vair:.4f})={Vagg:.4f}\;m^3")
    st.markdown("---")

    st.markdown("### Step 11: Total coarse aggregate volume")
    st.latex(r"V_{CA,total}=CA_{fraction}\times V_{agg}")
    st.latex(rf"V_{{CA,total}}={ca_fraction:.3f}\times{Vagg:.4f}={Vca_total:.4f}\;m^3")
    st.markdown("---")

    st.markdown("### Step 12: Volumetric batching / volume-based RCA replacement")
    st.markdown("RCA replacement is first done by **volume**, then converted into kg/m³.")
    st.latex(r"V_{RCA}=R\times V_{CA,total}")
    st.latex(r"V_{NCA}=(1-R)\times V_{CA,total}")
    st.latex(rf"V_{{RCA}}={rca_pct/100:.2f}\times{Vca_total:.4f}={Vrca:.4f}\;m^3")
    st.latex(rf"V_{{NCA}}={1-rca_pct/100:.2f}\times{Vca_total:.4f}={Vnca:.4f}\;m^3")
    st.markdown("---")

    st.markdown("### Step 13: Split CA into NCA and RCA")
    st.latex(r"M_{RCA}=V_{RCA}\times SG_{RCA}\times1000")
    st.latex(r"M_{NCA}=V_{NCA}\times SG_{NCA}\times1000")
    st.latex(rf"M_{{RCA}}={Vrca:.4f}\times{sg_rca:.3f}\times1000={Mrca:.2f}\;kg/m^3")
    st.latex(rf"M_{{NCA}}={Vnca:.4f}\times{sg_nca:.3f}\times1000={Mnca:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Step 14: Fine aggregate quantity")
    st.latex(r"V_{FA}=(1-CA_{fraction})\times V_{agg}")
    st.latex(rf"V_{{FA}}={(1-ca_fraction):.3f}\times{Vagg:.4f}={Vfa:.4f}\;m^3")
    st.latex(r"M_{FA}=V_{FA}\times SG_{FA}\times1000")
    st.latex(rf"M_{{FA}}={Vfa:.4f}\times{sg_fa:.3f}\times1000={Mfa:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Step 15: Aggregate water corrections")
    st.markdown("Positive value means water should be added; negative value means water should be deducted.")
    st.latex(r"W_{corr}=M_{agg}\times\frac{WA-MC}{100}")
    st.latex(rf"W_{{FA,corr}}={Mfa:.2f}\times\frac{{{wa_fa:.3f}-{mc_fa:.3f}}}{{100}}={Wcorr_fa:.2f}\;kg/m^3")
    st.latex(rf"W_{{NCA,corr}}={Mnca:.2f}\times\frac{{{wa_nca:.3f}-{mc_nca:.3f}}}{{100}}={Wcorr_nca:.2f}\;kg/m^3")
    st.latex(rf"W_{{RCA,corr}}={Mrca:.2f}\times\frac{{{wa_rca:.3f}-{mc_rca:.3f}}}{{100}}={Wcorr_rca:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Step 16: Final batching water")
    st.latex(r"W_{batch}=W_{eff}+W_{FA,corr}+W_{NCA,corr}+W_{RCA,corr}")
    st.latex(rf"W_{{batch}}={water_eff:.2f}+{Wcorr_fa:.2f}+{Wcorr_nca:.2f}+{Wcorr_rca:.2f}={Wbatch:.2f}\;kg/m^3")
    st.markdown("---")

    st.markdown("### Final mix proportion")
    st.latex(rf"C:FA:NCA:RCA=1:{Mfa/ccomp:.2f}:{Mnca/ccomp:.2f}:{Mrca/ccomp:.2f}")
    st.latex(rf"C:FA:Total\;CA=1:{Mfa/ccomp:.2f}:{Mca/ccomp:.2f}")

    st.markdown("## Final Mix Design")
    final_mix_design_display = pd.DataFrame([
        {"Material": "Cement", "Quantity": f"{ccomp:.2f} kg/m³"},
        {"Material": "Effective water", "Quantity": f"{water_eff:.2f} kg/m³"},
        {"Material": "FA water correction", "Quantity": f"{Wcorr_fa:.2f} kg/m³"},
        {"Material": "NCA water correction", "Quantity": f"{Wcorr_nca:.2f} kg/m³"},
        {"Material": "RCA water correction", "Quantity": f"{Wcorr_rca:.2f} kg/m³"},
        {"Material": "Total aggregate water correction", "Quantity": f"{(Wcorr_fa + Wcorr_nca + Wcorr_rca):.2f} kg/m³"},
        {"Material": "Batching water", "Quantity": f"{Wbatch:.2f} kg/m³"},
        {"Material": "Fine aggregate", "Quantity": f"{Mfa:.2f} kg/m³"},
        {"Material": "Natural coarse aggregate", "Quantity": f"{Mnca:.2f} kg/m³"},
        {"Material": "Recycled coarse aggregate", "Quantity": f"{Mrca:.2f} kg/m³"},
    ])
    st.table(final_mix_design_display)



# ============================================================
# STREAMLIT UI
# ============================================================

st.title("A6-D2 RCA Mix Design Charts Automation")
st.write(
    "Corrected version: total coarse aggregate volume is solved automatically from IS 10262 absolute volume logic. "
    "Aggregate water corrections are applied at the end for FA, NCA and RCA."
)

with st.sidebar:
    st.header("1. Grade and strength inputs")
    grade = st.selectbox("Concrete grade", list(GRADE_INFO.keys()), index=0)
    g = GRADE_INFO[grade]

    fck = st.number_input("fck (MPa)", value=float(g["fck"]), step=1.0)
    s = st.number_input("Standard deviation s (MPa)", value=float(g["s"]), step=0.5)
    target = st.number_input("Target mean strength fck' (MPa)", value=float(target_mean_strength(fck, s)), step=0.1)
    selected_wc = st.number_input("Assumed/check w/c ratio", value=float(g["ref_wc"]), min_value=0.20, max_value=1.00, step=0.01)

    st.header("2. Chart range")
    wc_min = st.number_input("Chart w/c minimum", value=float(g["wc_min"]), step=0.01)
    wc_max = st.number_input("Chart w/c maximum", value=float(g["wc_max"]), step=0.01)
    wc_step = st.number_input("Chart w/c interval", value=0.01, min_value=0.005, max_value=0.05, step=0.005)
    rca_interval = st.selectbox("RCA replacement interval (%)", [5, 10, 20, 25], index=1)

    choices = list(range(0, 101, rca_interval))
    selected_rca = st.select_slider("Highlight RCA replacement (%)", options=choices, value=40 if 40 in choices else 50)

    st.header("3. IS 10262 water and air inputs")
    w50 = st.number_input("Base water for 50 mm slump, W50 (kg/m³)", value=186.0, step=1.0)
    slump = st.number_input("Required slump (mm)", value=100.0, step=5.0)
    water_eff = slump_corrected_water(w50, slump)
    st.caption(f"Calculated W_eff = {water_eff:.2f} kg/m³ using 3% per 25 mm slump correction")

    agg_size = st.number_input("Nominal aggregate size (mm)", value=20.0, step=1.0)
    air_percent = st.number_input("Entrapped air (%)", value=2.0, step=0.1)

    st.header("4. Aggregate proportion")
    ca_fraction = st.number_input(
        "CA volume fraction from IS 10262 table",
        value=0.620,
        min_value=0.10,
        max_value=0.90,
        step=0.001,
        format="%.3f"
    )
    st.caption("VCA,total will be solved as CA fraction × [1 - (Vc + Vw + Vair)].")

    st.header("5. Specific gravities")
    sg_cement = st.number_input("Cement specific gravity", value=3.15, step=0.01)
    sg_fa = st.number_input("FA specific gravity", value=2.65, step=0.01)
    sg_nca = st.number_input("NCA specific gravity", value=2.65, step=0.01)
    sg_rca = st.number_input("RCA specific gravity", value=2.463, step=0.001, format="%.3f")

    st.header("6. Absorption and moisture corrections")
    wa_fa = st.number_input("FA water absorption WA (%)", value=1.000, min_value=0.0, max_value=20.0, step=0.001, format="%.3f")
    mc_fa = st.number_input("FA moisture content MC (%)", value=0.000, min_value=0.0, max_value=20.0, step=0.001, format="%.3f")

    wa_nca = st.number_input("NCA water absorption WA (%)", value=0.500, min_value=0.0, max_value=20.0, step=0.001, format="%.3f")
    mc_nca = st.number_input("NCA moisture content MC (%)", value=0.000, min_value=0.0, max_value=20.0, step=0.001, format="%.3f")

    wa_rca = st.number_input("RCA water absorption WA (%)", value=4.130, min_value=0.0, max_value=20.0, step=0.001, format="%.3f")
    mc_rca = st.number_input("RCA moisture content MC (%)", value=0.000, min_value=0.0, max_value=20.0, step=0.001, format="%.3f")

    st.header("7. Cement limits")
    min_cement = st.number_input("Minimum cement limit (kg/m³)", value=0.0, step=5.0)
    max_cement = st.number_input("Maximum cement limit (kg/m³)", value=9999.0, step=5.0)

    # --------------------------------------------------------
    # Fixed A6-D2 model constants
    # These are hidden from the final user interface because
    # they are model calibration parameters, not mix-design inputs.
    # --------------------------------------------------------
    wc_anchor = float(g["wc_min"])
    slope = 0.55
    curvature = 9.50
    severity_base = 0.120
    severity_wc = 0.010
    severity_wc2 = 0.250
    nonlinear_factor = 0.005

    st.header("8. RCA quality chart settings")
    qxmin = st.number_input("Chart 5 SG min", value=2.00, step=0.05)
    qxmax = st.number_input("Chart 5 SG max", value=2.80, step=0.05)
    qymin = st.number_input("Chart 5 WA min (%)", value=0.0, step=0.5)
    qymax = st.number_input("Chart 5 WA max (%)", value=10.0, step=0.5)
    show_ref = st.checkbox("Show reference points", value=True)


strength_df = generate_strength_df(
    grade, target, wc_anchor, wc_min, wc_max, wc_step, rca_interval,
    water_eff, slope, curvature, severity_base, severity_wc, severity_wc2, nonlinear_factor
)

cement_df = generate_cement_df(strength_df, target, water_eff, min_cement, max_cement)

mix_df = generate_mix_df(
    grade, fck, s, target, selected_wc, water_eff, air_percent,
    sg_cement, sg_fa, sg_nca, sg_rca,
    wa_fa, mc_fa, wa_nca, mc_nca, wa_rca, mc_rca,
    ca_fraction, rca_interval, min_cement, max_cement,
    wc_anchor, slope, curvature, severity_base, severity_wc, severity_wc2, nonlinear_factor
)

selected_row = mix_df[mix_df["RCA replacement (%)"] == selected_rca].iloc[0]

st.markdown("### Selected A6-D2 design result")
cols = st.columns(7)
cols[0].metric("Target mean", f"{target:.2f} MPa")
cols[1].metric("Predicted strength", f"{selected_row['Predicted strength (MPa)']:.2f} MPa")
cols[2].metric("A6-D2 SRF", f"{selected_row['A6-D2 SRF']:.3f}")
cols[3].metric("Ccomp", f"{selected_row['Compensated cement Ccomp (kg/m3)']:.2f} kg/m³")
cols[4].metric("Solved VCA,total", f"{selected_row['Total coarse aggregate volume Vca_total']:.4f} m³")
cols[5].metric("RCA", f"{selected_row['RCA aggregate content (kg/m3)']:.2f} kg/m³")
cols[6].metric("Batching water", f"{selected_row['Batching water to be added (kg/m3)']:.2f} kg/m³")

tabs = st.tabs([
    "Chart 1 Strength",
    "Chart 2 SRF",
    "Chart 3 Cement",
    "Chart 4 4-Axis",
    "Chart 5 RCA Quality",
    "Chart 6 Water",
    "Manual Mix Design",
    "Data + Download"
])

figures = {}

with tabs[0]:
    fig = plot_strength(strength_df, grade, target, selected_rca, selected_wc, water_eff, slump, agg_size)
    figures["Chart1_Strength_vs_wc.png"] = fig
    st.pyplot(fig, use_container_width=True)

with tabs[1]:
    fig = plot_srf(strength_df, grade, selected_rca, selected_wc)
    figures["Chart2_SRF_vs_wc.png"] = fig
    st.pyplot(fig, use_container_width=True)

with tabs[2]:
    fig = plot_cement(cement_df, grade, target, selected_rca, selected_wc)
    figures["Chart3_Cement_Compensation_vs_wc.png"] = fig
    st.pyplot(fig, use_container_width=True)

with tabs[3]:
    fig = plot_4axis(mix_df, grade, selected_rca, selected_wc, slump, agg_size)
    figures["Chart4_Corrected_4Axis.png"] = fig
    st.pyplot(fig, use_container_width=True)

with tabs[4]:
    fig, quality, note = plot_quality(sg_rca, wa_rca, qxmin, qxmax, qymin, qymax, show_ref)
    figures["Chart5_RCA_Quality_Check.png"] = fig
    st.pyplot(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("RCA SG", f"{sg_rca:.3f}")
    c2.metric("RCA WA", f"{wa_rca:.3f}%")
    c3.metric("Quality class", quality)
    st.info(note)

with tabs[5]:
    fig = plot_water(mix_df, grade, selected_rca, selected_wc, wa_rca, mc_rca)
    figures["Chart6_Aggregate_Water_Correction.png"] = fig
    st.pyplot(fig, use_container_width=True)

with tabs[6]:
    st.subheader("Full Manual Mix Design: IS 10262 + A6-D2")
    st.write(
        "This manual tab solves the mix in the correct order. "
        "Total coarse aggregate volume is calculated automatically after final cement content is known."
    )

    manual_steps, manual_final, manual_ratio = manual_design_tables(
        grade, fck, s, target, selected_wc, selected_rca,
        w50, slump, water_eff, air_percent,
        sg_cement, sg_fa, sg_nca, sg_rca,
        wa_fa, mc_fa, wa_nca, mc_nca, wa_rca, mc_rca,
        ca_fraction, mix_df
    )

    display_manual_solution_markdown(manual_steps, manual_final, manual_ratio, mix_df, selected_rca)

    with st.expander("Show calculation table"):
        st.markdown("### A. Step-by-step calculation table")
        st.dataframe(manual_steps, use_container_width=True)

        st.markdown("### B. Final quantities")
        st.dataframe(manual_final.style.format({"Value": "{:.3f}"}), use_container_width=True)

        st.markdown("### C. Final mix ratio")
        st.dataframe(manual_ratio, use_container_width=True)

    st.info(
        "Key correction: VCA,total is not entered manually. It is solved as "
        "VCA,total = CA fraction × [1 - (Vc + Vw + Vair)]. "
        "Then RCA and NCA are split volumetrically."
    )

    st.download_button(
        "Download manual calculation as CSV",
        data=manual_steps.to_csv(index=False),
        file_name=f"{grade}_A6D2_Manual_Mix_Design_Steps.csv",
        mime="text/csv"
    )

with tabs[7]:
    qty, ratio, summary = final_tables(grade, fck, target, selected_wc, selected_rca, mix_df)

    st.markdown("### Final selected mix quantities")
    st.dataframe(qty, use_container_width=True)

    st.markdown("### Final mix ratio")
    st.dataframe(ratio, use_container_width=True)

    st.success(f"Recommended RCA split mix ratio: **Cement : FA : NCA : RCA = {summary['Ratio']}**")

    with st.expander("Strength chart data"):
        st.dataframe(strength_df, use_container_width=True)
    with st.expander("Cement compensation data"):
        st.dataframe(cement_df, use_container_width=True)
    with st.expander("Mix design data"):
        st.dataframe(mix_df, use_container_width=True)

    manual_steps, manual_final, manual_ratio = manual_design_tables(
        grade, fck, s, target, selected_wc, selected_rca,
        w50, slump, water_eff, air_percent,
        sg_cement, sg_fa, sg_nca, sg_rca,
        wa_fa, mc_fa, wa_nca, mc_nca, wa_rca, mc_rca,
        ca_fraction, mix_df
    )

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in figures.items():
            z.writestr(f"{grade}_{name}", fig_to_png(fig))

        z.writestr(f"{grade}_A6D2_Strength_Data.csv", strength_df.to_csv(index=False))
        z.writestr(f"{grade}_A6D2_Cement_Data.csv", cement_df.to_csv(index=False))
        z.writestr(f"{grade}_A6D2_Mix_Data.csv", mix_df.to_csv(index=False))
        z.writestr(f"{grade}_A6D2_Manual_Mix_Design_Steps.csv", manual_steps.to_csv(index=False))
        z.writestr(f"{grade}_A6D2_Manual_Final_Quantities.csv", manual_final.to_csv(index=False))
        z.writestr(f"{grade}_A6D2_Manual_Mix_Ratio.csv", manual_ratio.to_csv(index=False))
        z.writestr(f"{grade}_Final_Selected_Mix_Quantities.csv", qty.to_csv(index=False))
        z.writestr(f"{grade}_Final_Mix_Ratio.csv", ratio.to_csv(index=False))

    zbuf.seek(0)

    st.download_button(
        "Download all A6-D2 charts + CSV data",
        data=zbuf,
        file_name=f"{grade}_A6D2_RCA_Mix_Design_Charts_All.zip",
        mime="application/zip"
    )
