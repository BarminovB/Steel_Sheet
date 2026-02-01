"""
Profiled Steel Sheet Calculator v3.3
EN 1993-1-3 compliant with detailed calculation output
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve
import copy

# ============================================================================
# PROFILE DATABASE
# ============================================================================

def get_profile_database():
    """Full Ruukki profile database. Units: mm, cm4/m, cm3/m, mm2/m"""
    db = {}
    
    # T45-30L-905
    t45_base = {"type": "T45-30L-905", "h": 45, "pitch": 151, "b_eff": 905,
        "h_w": 40, "phi": 75, "r": 3, "num_webs": 6, "b_top": 50, "b_bottom": 44}
    t45_data = {
        0.6: {"I_pos": 26.1, "I_neg": 23.9, "W_pos": 11.6, "W_neg": 10.6, "A_eff": 773},
        0.7: {"I_pos": 30.0, "I_neg": 27.5, "W_pos": 13.3, "W_neg": 12.2, "A_eff": 902},
        0.8: {"I_pos": 33.7, "I_neg": 30.9, "W_pos": 15.0, "W_neg": 13.7, "A_eff": 1031},
        0.9: {"I_pos": 37.3, "I_neg": 34.2, "W_pos": 16.6, "W_neg": 15.2, "A_eff": 1160},
        1.0: {"I_pos": 40.7, "I_neg": 37.3, "W_pos": 18.1, "W_neg": 16.6, "A_eff": 1289},
    }
    for t, props in t45_data.items():
        db[f"T45-30L-905-{t}"] = {**t45_base, "t": t, "E": 210000, **props}
    
    # T55-53L-976
    t55_base = {"type": "T55-53L-976", "h": 55, "pitch": 163, "b_eff": 976,
        "h_w": 50, "phi": 72, "r": 4, "num_webs": 6, "b_top": 53, "b_bottom": 48}
    t55_data = {
        0.7: {"I_pos": 50.0, "I_neg": 45.0, "W_pos": 18.2, "W_neg": 16.4, "A_eff": 950},
        0.8: {"I_pos": 56.5, "I_neg": 50.9, "W_pos": 20.5, "W_neg": 18.5, "A_eff": 1086},
        0.9: {"I_pos": 62.8, "I_neg": 56.5, "W_pos": 22.8, "W_neg": 20.5, "A_eff": 1221},
        1.0: {"I_pos": 68.8, "I_neg": 61.9, "W_pos": 25.0, "W_neg": 22.5, "A_eff": 1357},
    }
    for t, props in t55_data.items():
        db[f"T55-53L-976-{t}"] = {**t55_base, "t": t, "E": 210000, **props}
    
    # T60-53L-915
    t60_base = {"type": "T60-53L-915", "h": 60, "pitch": 183, "b_eff": 915,
        "h_w": 55, "phi": 70, "r": 4, "num_webs": 5, "b_top": 53, "b_bottom": 50}
    t60_data = {
        0.7: {"I_pos": 65.0, "I_neg": 58.0, "W_pos": 21.7, "W_neg": 19.3, "A_eff": 1050},
        0.8: {"I_pos": 73.5, "I_neg": 65.6, "W_pos": 24.5, "W_neg": 21.9, "A_eff": 1200},
        0.9: {"I_pos": 81.8, "I_neg": 73.0, "W_pos": 27.3, "W_neg": 24.3, "A_eff": 1350},
        1.0: {"I_pos": 89.8, "I_neg": 80.2, "W_pos": 29.9, "W_neg": 26.7, "A_eff": 1500},
    }
    for t, props in t60_data.items():
        db[f"T60-53L-915-{t}"] = {**t60_base, "t": t, "E": 210000, **props}
    
    # T70-57L-1058
    t70_base = {"type": "T70-57L-1058", "h": 70, "pitch": 212, "b_eff": 1058,
        "h_w": 63, "phi": 70, "r": 4, "num_webs": 5, "b_top": 57, "b_bottom": 55}
    t70_data = {
        0.7: {"I_pos": 95.0, "I_neg": 85.0, "W_pos": 27.1, "W_neg": 24.3, "A_eff": 1100},
        0.8: {"I_pos": 108.0, "I_neg": 97.0, "W_pos": 30.9, "W_neg": 27.7, "A_eff": 1257},
        0.9: {"I_pos": 120.5, "I_neg": 108.0, "W_pos": 34.4, "W_neg": 30.9, "A_eff": 1414},
        1.0: {"I_pos": 132.5, "I_neg": 119.0, "W_pos": 37.9, "W_neg": 34.0, "A_eff": 1571},
        1.1: {"I_pos": 144.0, "I_neg": 129.0, "W_pos": 41.1, "W_neg": 36.9, "A_eff": 1729},
        1.25: {"I_pos": 161.0, "I_neg": 145.0, "W_pos": 46.0, "W_neg": 41.4, "A_eff": 1964},
    }
    for t, props in t70_data.items():
        db[f"T70-57L-1058-{t}"] = {**t70_base, "t": t, "E": 210000, **props}
    
    # T85-40L-1120
    t85_base = {"type": "T85-40L-1120", "h": 85, "pitch": 224, "b_eff": 1120,
        "h_w": 78, "phi": 72, "r": 5, "num_webs": 5, "b_top": 40, "b_bottom": 80}
    t85_data = {
        0.7: {"I_pos": 140.0, "I_neg": 125.0, "W_pos": 32.9, "W_neg": 29.4, "A_eff": 1050},
        0.8: {"I_pos": 158.0, "I_neg": 141.0, "W_pos": 37.2, "W_neg": 33.2, "A_eff": 1200},
        0.9: {"I_pos": 175.0, "I_neg": 156.0, "W_pos": 41.2, "W_neg": 36.7, "A_eff": 1350},
        1.0: {"I_pos": 192.0, "I_neg": 171.0, "W_pos": 45.2, "W_neg": 40.2, "A_eff": 1500},
        1.1: {"I_pos": 208.0, "I_neg": 185.0, "W_pos": 48.9, "W_neg": 43.5, "A_eff": 1650},
        1.25: {"I_pos": 232.0, "I_neg": 207.0, "W_pos": 54.6, "W_neg": 48.7, "A_eff": 1875},
    }
    for t, props in t85_data.items():
        db[f"T85-40L-1120-{t}"] = {**t85_base, "t": t, "E": 210000, **props}
    
    # T130M-75L-930
    t130m_base = {"type": "T130M-75L-930", "h": 130, "pitch": 233, "b_eff": 930,
        "h_w": 120, "phi": 68, "r": 5, "num_webs": 4, "b_top": 75, "b_bottom": 65}
    t130m_data = {
        0.8: {"I_pos": 350.0, "I_neg": 310.0, "W_pos": 53.8, "W_neg": 47.7, "A_eff": 1350},
        0.9: {"I_pos": 390.0, "I_neg": 345.0, "W_pos": 60.0, "W_neg": 53.1, "A_eff": 1519},
        1.0: {"I_pos": 430.0, "I_neg": 380.0, "W_pos": 66.2, "W_neg": 58.5, "A_eff": 1688},
        1.1: {"I_pos": 468.0, "I_neg": 414.0, "W_pos": 72.0, "W_neg": 63.7, "A_eff": 1856},
        1.25: {"I_pos": 524.0, "I_neg": 463.0, "W_pos": 80.6, "W_neg": 71.2, "A_eff": 2109},
        1.5: {"I_pos": 616.0, "I_neg": 545.0, "W_pos": 94.8, "W_neg": 83.8, "A_eff": 2531},
    }
    for t, props in t130m_data.items():
        db[f"T130M-75L-930-{t}"] = {**t130m_base, "t": t, "E": 210000, **props}
    
    # T153-40L-840
    t153_base = {"type": "T153-40L-840", "h": 153, "pitch": 280, "b_eff": 840,
        "h_w": 143, "phi": 65, "r": 5, "num_webs": 3, "b_top": 40, "b_bottom": 100}
    t153_data = {
        0.9: {"I_pos": 560.0, "I_neg": 495.0, "W_pos": 73.2, "W_neg": 64.7, "A_eff": 1575},
        1.0: {"I_pos": 620.0, "I_neg": 548.0, "W_pos": 81.0, "W_neg": 71.6, "A_eff": 1750},
        1.1: {"I_pos": 678.0, "I_neg": 600.0, "W_pos": 88.6, "W_neg": 78.4, "A_eff": 1925},
        1.2: {"I_pos": 735.0, "I_neg": 650.0, "W_pos": 96.1, "W_neg": 84.9, "A_eff": 2100},
        1.25: {"I_pos": 762.0, "I_neg": 674.0, "W_pos": 99.6, "W_neg": 88.1, "A_eff": 2188},
        1.5: {"I_pos": 897.0, "I_neg": 793.0, "W_pos": 117.2, "W_neg": 103.7, "A_eff": 2625},
    }
    for t, props in t153_data.items():
        db[f"T153-40L-840-{t}"] = {**t153_base, "t": t, "E": 210000, **props}
    
    return db

def get_profile_types():
    db = get_profile_database()
    types = set()
    for name in db.keys():
        parts = name.rsplit("-", 1)
        if len(parts) == 2:
            types.add(parts[0])
    return sorted(list(types))

def get_thicknesses_for_type(profile_type):
    db = get_profile_database()
    thicknesses = []
    for name, props in db.items():
        if name.startswith(profile_type + "-"):
            thicknesses.append(props["t"])
    return sorted(thicknesses)

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_profile_cross_section(props, profile_name=""):
    fig, ax = plt.subplots(figsize=(10, 4))
    h = props["h"]
    t = props["t"]
    pitch = props.get("pitch", 150)
    phi = props.get("phi", 75)
    b_top = props.get("b_top", pitch * 0.35)
    b_bottom = props.get("b_bottom", pitch * 0.4)
    phi_rad = np.radians(phi)
    web_offset = h / np.tan(phi_rad) if phi < 90 else 0
    num_ribs = 2
    profile_x = []
    profile_y = []
    x_current = 0
    for rib in range(num_ribs + 1):
        if rib == 0:
            profile_x.append(x_current)
            profile_y.append(0)
        profile_x.append(x_current + b_bottom/2)
        profile_y.append(0)
        profile_x.append(x_current + b_bottom/2 + web_offset)
        profile_y.append(h)
        profile_x.append(x_current + b_bottom/2 + web_offset + b_top)
        profile_y.append(h)
        profile_x.append(x_current + b_bottom/2 + 2*web_offset + b_top)
        profile_y.append(0)
        if rib < num_ribs:
            profile_x.append(x_current + pitch)
            profile_y.append(0)
            x_current += pitch
    ax.plot(profile_x, profile_y, "b-", linewidth=2.5)
    inner_y = [max(0, y - t) if y > 0 else y for y in profile_y]
    ax.fill_between(profile_x, inner_y, profile_y, alpha=0.4, color="steelblue")
    total_width = max(profile_x)
    ax.annotate("", xy=(-15, h), xytext=(-15, 0), arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(-25, h/2, f"h={h}", fontsize=10, ha="right", va="center", color="red", fontweight="bold")
    ax.annotate("", xy=(0, -12), xytext=(pitch, -12), arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(pitch/2, -22, f"pitch={pitch}", fontsize=10, ha="center", color="green", fontweight="bold")
    ax.text(total_width * 0.8, h + 15, f"t = {t} mm", fontsize=11, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8, edgecolor="orange"), fontweight="bold")
    ax.set_xlim(-50, total_width + 30)
    ax.set_ylim(-35, h + 35)
    ax.set_aspect("equal")
    ax.set_xlabel("Width [mm]")
    ax.set_ylabel("Height [mm]")
    ax.set_title(f"Profile: {profile_name}", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def draw_beam_diagram(spans, udl_uls, udl_sls, point_loads):
    total_length = sum(spans)
    fig, ax = plt.subplots(figsize=(12, 4))
    beam_y = 1.0
    ax.plot([0, total_length], [beam_y, beam_y], "b-", linewidth=5)
    support_positions = [0]
    cumulative = 0
    for span in spans:
        cumulative += span
        support_positions.append(cumulative)
    for i, x_sup in enumerate(support_positions):
        triangle = plt.Polygon([[x_sup - 0.15, beam_y - 0.05], [x_sup + 0.15, beam_y - 0.05],
                                [x_sup, beam_y - 0.30]], closed=True, facecolor="dimgray", edgecolor="black")
        ax.add_patch(triangle)
        ax.plot([x_sup - 0.2, x_sup + 0.2], [beam_y - 0.32, beam_y - 0.32], "k-", linewidth=2)
        ax.text(x_sup, beam_y - 0.45, f"{i+1}", ha="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="lightgray", edgecolor="black"))
    if udl_uls > 0:
        n_arrows = max(int(total_length / 0.25), 10)
        for x_arr in np.linspace(0.05, total_length - 0.05, n_arrows):
            ax.annotate("", xy=(x_arr, beam_y + 0.03), xytext=(x_arr, beam_y + 0.35),
                       arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        ax.plot([0, total_length], [beam_y + 0.35, beam_y + 0.35], "r-", linewidth=2)
        ax.text(total_length / 2, beam_y + 0.50, f"ULS: {udl_uls:.2f} kN/m2 | SLS: {udl_sls:.2f} kN/m2",
                ha="center", fontsize=11, color="darkred", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.95))
    for i, pl in enumerate(point_loads):
        x_pl = pl["position"]
        p_uls = pl.get("magnitude_uls", pl.get("magnitude", 0))
        p_sls = pl.get("magnitude_sls", p_uls)
        ax.annotate("", xy=(x_pl, beam_y + 0.03), xytext=(x_pl, beam_y + 0.65),
                   arrowprops=dict(arrowstyle="->", color="blue", lw=2.5))
        ax.text(x_pl, beam_y + 0.75, f"P{i+1}\n{p_uls:.1f}/{p_sls:.1f}", ha="center", fontsize=9,
                color="darkblue", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
    cumulative = 0
    for i, span in enumerate(spans):
        mid = cumulative + span / 2
        ax.annotate("", xy=(cumulative + 0.05, beam_y - 0.60), xytext=(cumulative + span - 0.05, beam_y - 0.60),
                   arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
        ax.text(mid, beam_y - 0.72, f"L{i+1} = {span:.2f} m", ha="center", fontsize=10,
                color="darkgreen", fontweight="bold")
        cumulative += span
    ax.set_xlim(-0.4, total_length + 0.4)
    ax.set_ylim(-0.95, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Loading Diagram", fontweight="bold")
    plt.tight_layout()
    return fig

# ============================================================================
# STRUCTURAL CALCULATIONS
# ============================================================================

def calculate_single_span(span, udl, point_loads, E, I):
    n = 1001
    x = np.linspace(0, span, n)
    Ra = udl * span / 2
    for pl in point_loads:
        if 0 <= pl["position"] <= span:
            Ra += pl["magnitude"] * (span - pl["position"]) / span
    V = np.full_like(x, Ra, dtype=float) - udl * x
    for pl in point_loads:
        if 0 <= pl["position"] <= span:
            V = np.where(x > pl["position"], V - pl["magnitude"], V)
    M = cumulative_trapezoid(V, x, initial=0)
    I_m4 = I * 1e-8
    E_Pa = E * 1e6
    EI = E_Pa * I_m4
    M_Nm = M * 1000
    theta = cumulative_trapezoid(M_Nm / EI, x, initial=0)
    y_raw = cumulative_trapezoid(theta, x, initial=0)
    y_corrected = y_raw - (y_raw[-1] / span) * x
    defl = -y_corrected * 1000
    return x, M, V, defl

def calculate_multi_span(spans, udl, point_loads, E, I):
    num_spans = len(spans)
    total_length = sum(spans)
    n = 1001
    if num_spans == 1:
        x, M, V, defl = calculate_single_span(spans[0], udl, point_loads, E, I)
        Ra = udl * spans[0] / 2
        for pl in point_loads:
            Ra += pl["magnitude"] * (spans[0] - pl["position"]) / spans[0]
        Rb = udl * spans[0] + sum(pl["magnitude"] for pl in point_loads) - Ra
        return x, M, V, defl, [Ra, Rb]
    n_unknowns = num_spans - 1
    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)
    for i in range(n_unknowns):
        L_i = spans[i]
        L_ip1 = spans[i + 1]
        A[i, i] = 2 * (L_i + L_ip1)
        if i > 0:
            A[i, i-1] = L_i
        if i < n_unknowns - 1:
            A[i, i+1] = L_ip1
        b[i] = -udl / 4 * (L_i**3 + L_ip1**3)
    try:
        M_supports = solve(A, b)
    except:
        M_supports = np.zeros(n_unknowns)
    M_sup_full = np.concatenate([[0], M_supports, [0]])
    x = np.linspace(0, total_length, n)
    M = np.zeros(n)
    V = np.zeros(n)
    cumulative = 0
    reactions = []
    for span_idx, span_len in enumerate(spans):
        M_left = M_sup_full[span_idx]
        M_right = M_sup_full[span_idx + 1]
        mask = (x >= cumulative - 1e-9) & (x <= cumulative + span_len + 1e-9)
        x_local = x[mask] - cumulative
        M_udl = udl * x_local * (span_len - x_local) / 2
        M_support = M_left + (M_right - M_left) * x_local / span_len
        M[mask] = M_udl + M_support
        R_left = udl * span_len / 2 - (M_right - M_left) / span_len
        if span_idx == 0:
            reactions.append(R_left)
        V_local = R_left - udl * x_local
        V[mask] = V_local
        if span_idx < num_spans - 1:
            reactions.append(-V_local[-1])
        cumulative += span_len
    reactions.append(-V[-1] if len(V) > 0 else 0)
    I_m4 = I * 1e-8
    E_Pa = E * 1e6
    EI = E_Pa * I_m4
    defl = np.zeros(n)
    cumulative = 0
    for span_idx, span_len in enumerate(spans):
        mask = (x >= cumulative - 1e-9) & (x <= cumulative + span_len + 1e-9)
        indices = np.where(mask)[0]
        if len(indices) > 1:
            x_span = x[indices]
            M_span = M[indices]
            M_Nm = M_span * 1000
            theta = cumulative_trapezoid(M_Nm / EI, x_span, initial=0)
            y_raw = cumulative_trapezoid(theta, x_span, initial=0)
            y_corrected = y_raw - (y_raw[-1] / span_len) * (x_span - cumulative)
            defl[indices] = -y_corrected * 1000
        cumulative += span_len
    return x, M, V, defl, reactions

# ============================================================================
# EN 1993-1-3 RESISTANCE CALCULATIONS
# ============================================================================

def calculate_resistances(props, fy, gamma_m):
    """
    EN 1993-1-3, 6.1.4 and 6.1.5
    M_c,Rd = W_eff * f_yb / gamma_M0
    V_Rd = A_eff * f_yb / (sqrt(3) * gamma_M0)
    """
    W_pos = props.get("W_pos", 20) * 1e-6  # cm3/m -> m3/m
    W_neg = props.get("W_neg", W_pos * 1e6 * 0.9) * 1e-6
    f_Pa = fy * 1e6  # MPa -> Pa
    
    M_Rd_pos = W_pos * f_Pa / gamma_m / 1000  # kNm/m
    M_Rd_neg = W_neg * f_Pa / gamma_m / 1000
    
    A_eff = props["A_eff"] * 1e-6  # mm2/m -> m2/m
    V_Rd = A_eff * f_Pa / (np.sqrt(3) * gamma_m) / 1000  # kN/m
    
    return {"M_Rd_pos": M_Rd_pos, "M_Rd_neg": M_Rd_neg, "V_Rd": V_Rd,
            "W_pos": props.get("W_pos", 20), "W_neg": props.get("W_neg", 18)}

def check_web_crippling(F_Ed, props, fy, gamma_m, s_s, category, num_webs_eff=None):
    """
    EN 1993-1-3, 6.1.7.3 - Web crippling for sheeting
    R_w,Rd = alpha * t^2 * sqrt(f_yb * E) * k1 * k2 * k3 / gamma_M1
    
    where:
    k1 = (1 - 0.1 * sqrt(r/t))
    k2 = (0.5 + sqrt(0.02 * l_a / t))
    k3 = (2.4 + (phi/90)^2)
    
    Category 1: End support (c <= 1.5*h_w), alpha=0.075, l_a=10mm
    Category 2: Internal support, alpha=0.15, l_a=s_s
    
    num_webs_eff: Number of webs effectively resisting the load
        - For support reactions (UDL): all webs
        - For true point loads: 1-2 webs only
    """
    t = props["t"]
    r = props["r"]
    phi = props["phi"]
    E = props["E"]
    num_webs = props.get("num_webs", 2)
    
    # Use effective webs if specified, otherwise all webs
    if num_webs_eff is None:
        num_webs_eff = num_webs
    
    if category == 1:
        alpha = 0.075
        l_a = 10.0
    else:
        alpha = 0.15
        l_a = s_s
    
    k1 = 1 - 0.1 * np.sqrt(r / t)
    k2 = 0.5 + np.sqrt(0.02 * l_a / t)
    k3 = 2.4 + (phi / 90)**2
    sqrt_fy_E = np.sqrt(fy * E)
    
    R_w_Rd_single = alpha * t**2 * sqrt_fy_E * k1 * k2 * k3 / gamma_m  # N per web
    R_w_Rd = R_w_Rd_single * num_webs_eff / 1000  # kN total
    
    utilization = F_Ed / R_w_Rd if R_w_Rd > 0 else float("inf")
    
    return {
        "R_w_Rd": R_w_Rd,
        "R_w_Rd_single": R_w_Rd_single / 1000,
        "utilization": utilization,
        "status": "OK" if utilization <= 1.0 else "NOT OK",
        "alpha": alpha,
        "l_a": l_a,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "sqrt_fy_E": sqrt_fy_E,
        "t": t,
        "r": r,
        "phi": phi,
        "num_webs": num_webs,
        "num_webs_eff": num_webs_eff,
        "category": category
    }

def check_combined_bending_shear(M_Ed, V_Ed, M_Rd, V_Rd):
    """EN 1993-1-3, 6.1.10"""
    if V_Ed <= 0.5 * V_Rd:
        M_Rd_reduced = M_Rd
        rho = 0
    else:
        rho = (2 * V_Ed / V_Rd - 1)**2
        M_Rd_reduced = M_Rd * (1 - rho)
    utilization = M_Ed / M_Rd_reduced if M_Rd_reduced > 0 else float("inf")
    return {"M_Rd_reduced": M_Rd_reduced, "utilization": utilization, 
            "status": "OK" if utilization <= 1.0 else "NOT OK", "rho": rho}

def check_combined_bending_web_crippling(M_Ed, F_Ed, M_Rd, R_w_Rd):
    """EN 1993-1-3, 6.1.11"""
    if M_Rd <= 0 or R_w_Rd <= 0:
        return {"interaction": float("inf"), "status": "NOT OK", 
                "M_ratio": 0, "F_ratio": 0}
    M_ratio = M_Ed / M_Rd
    F_ratio = F_Ed / R_w_Rd
    interaction = M_ratio + F_ratio
    return {"interaction": interaction, "utilization": interaction / 1.25, 
            "status": "OK" if interaction <= 1.25 else "NOT OK",
            "M_ratio": M_ratio, "F_ratio": F_ratio}

# ============================================================================
# PLOTTING
# ============================================================================

def plot_internal_forces(x, M, V, spans, point_loads):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.fill_between(x, 0, V, where=(V >= 0), alpha=0.3, color="blue")
    ax1.fill_between(x, 0, V, where=(V < 0), alpha=0.3, color="red")
    ax1.plot(x, V, "k-", linewidth=1.5)
    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.set_ylabel("Shear V [kN/m]")
    ax1.set_title("Shear Force Diagram (ULS)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    cumulative = 0
    for span in spans:
        ax1.axvline(x=cumulative, color="gray", linestyle="--", alpha=0.5)
        cumulative += span
    ax1.axvline(x=cumulative, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(x, 0, M, where=(M >= 0), alpha=0.3, color="red", label="M+ (sagging)")
    ax2.fill_between(x, 0, M, where=(M < 0), alpha=0.3, color="blue", label="M- (hogging)")
    ax2.plot(x, M, "k-", linewidth=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.invert_yaxis()
    ax2.set_ylabel("Moment M [kNm/m]")
    ax2.set_xlabel("Position x [m]")
    ax2.set_title("Bending Moment Diagram (ULS) - Positive DOWN", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    cumulative = 0
    for span in spans:
        ax2.axvline(x=cumulative, color="gray", linestyle="--", alpha=0.5)
        cumulative += span
    ax2.axvline(x=cumulative, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_deflection(x, defl, spans, defl_limit, point_loads):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, 0, defl, alpha=0.3, color="orange")
    ax.plot(x, defl, "b-", linewidth=2, label="Deflection")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axhline(y=defl_limit, color="red", linestyle="--", linewidth=2, 
               label=f"Limit L/200 = {defl_limit:.1f} mm")
    cumulative = 0
    for span in spans:
        ax.axvline(x=cumulative, color="gray", linestyle="--", alpha=0.5)
        ax.plot(cumulative, 0, marker="^", markersize=12, color="dimgray")
        cumulative += span
    ax.axvline(x=cumulative, color="gray", linestyle="--", alpha=0.5)
    ax.plot(cumulative, 0, marker="^", markersize=12, color="dimgray")
    idx_max = np.argmax(np.abs(defl))
    d_max = defl[idx_max]
    ax.plot(x[idx_max], d_max, "ro", markersize=10, zorder=5)
    ax.annotate(f"d_max = {d_max:.2f} mm\nx = {x[idx_max]:.2f} m", xy=(x[idx_max], d_max),
                xytext=(x[idx_max] + sum(spans)*0.05, d_max + defl_limit*0.15),
                fontsize=11, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="red"))
    ax.invert_yaxis()
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Deflection d [mm] (down)")
    ax.set_title("Deflection Diagram (SLS) - Positive DOWN", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Steel Sheet Calculator v3.3", page_icon="S", layout="wide")
    st.title("Steel Sheet Calculator v3.3")
    st.markdown("**EN 1993-1-3 | Full Ruukki Catalog | Detailed Calculation Output**")
    
    if "copied_profile" not in st.session_state:
        st.session_state.copied_profile = None
    
    # === SIDEBAR ===
    st.sidebar.header("Input Parameters")
    
    # 1. Profile Selection
    st.sidebar.subheader("1. Profile Selection")
    profile_mode = st.sidebar.radio("Mode", ["From Catalog", "Copy & Modify", "Full Manual"])
    
    if profile_mode == "From Catalog":
        profile_types = get_profile_types()
        selected_type = st.sidebar.selectbox("Profile Type", profile_types)
        available_t = get_thicknesses_for_type(selected_type)
        selected_t = st.sidebar.selectbox("Thickness [mm]", available_t, format_func=lambda x: f"{x} mm")
        profile_name = f"{selected_type}-{selected_t}"
        props = copy.deepcopy(get_profile_database().get(profile_name, {}))
        if not props:
            st.sidebar.error(f"Profile {profile_name} not found!")
            return
        if st.sidebar.button("Copy for modification"):
            st.session_state.copied_profile = copy.deepcopy(props)
            st.session_state.copied_profile["source_name"] = profile_name
            st.sidebar.success(f"Copied!")
            
    elif profile_mode == "Copy & Modify":
        if st.session_state.copied_profile is None:
            st.sidebar.warning("No profile copied! Select from Catalog first.")
            profile_types = get_profile_types()
            selected_type = st.sidebar.selectbox("Select profile", profile_types)
            available_t = get_thicknesses_for_type(selected_type)
            selected_t = st.sidebar.selectbox("Thickness", available_t, format_func=lambda x: f"{x} mm")
            profile_name = f"{selected_type}-{selected_t}"
            base_props = get_profile_database().get(profile_name, {})
            if st.sidebar.button("Copy"):
                st.session_state.copied_profile = copy.deepcopy(base_props)
                st.session_state.copied_profile["source_name"] = profile_name
                st.rerun()
            props = base_props
        else:
            base = st.session_state.copied_profile
            st.sidebar.success(f"Base: {base.get('source_name', 'Custom')}")
            st.sidebar.markdown("**Modify:**")
            
            new_t = st.sidebar.number_input("Thickness t [mm]", 0.4, 2.0, float(base["t"]), 0.05)
            t_ratio = new_t / base["t"]
            
            # Scale properties approximately
            props = copy.deepcopy(base)
            props["t"] = new_t
            props["I_pos"] = base["I_pos"] * t_ratio
            props["I_neg"] = base["I_neg"] * t_ratio
            props["W_pos"] = base["W_pos"] * t_ratio
            props["W_neg"] = base["W_neg"] * t_ratio
            props["A_eff"] = base["A_eff"] * t_ratio
            
            with st.sidebar.expander("Fine-tune scaled properties"):
                props["I_pos"] = st.number_input("I_pos [cm4/m]", 1.0, 2000.0, float(props["I_pos"]), 1.0)
                props["I_neg"] = st.number_input("I_neg [cm4/m]", 1.0, 2000.0, float(props["I_neg"]), 1.0)
                props["W_pos"] = st.number_input("W_pos [cm3/m]", 1.0, 500.0, float(props["W_pos"]), 0.5)
                props["W_neg"] = st.number_input("W_neg [cm3/m]", 1.0, 500.0, float(props["W_neg"]), 0.5)
                props["A_eff"] = st.number_input("A_eff [mm2/m]", 100.0, 5000.0, float(props["A_eff"]), 10.0)
            
            profile_name = f"{base.get('type', 'Custom')}-{new_t} (modified)"
            
            if st.sidebar.button("Clear"):
                st.session_state.copied_profile = None
                st.rerun()
    else:
        st.sidebar.markdown("**Manual input:**")
        c1, c2 = st.sidebar.columns(2)
        props = {
            "h": c1.number_input("h [mm]", 20.0, 250.0, 70.0),
            "t": c2.number_input("t [mm]", 0.4, 2.0, 0.9),
            "I_pos": c1.number_input("I_pos [cm4/m]", 10.0, 1500.0, 100.0),
            "I_neg": c2.number_input("I_neg [cm4/m]", 10.0, 1500.0, 90.0),
            "W_pos": c1.number_input("W_pos [cm3/m]", 5.0, 200.0, 30.0),
            "W_neg": c2.number_input("W_neg [cm3/m]", 5.0, 200.0, 27.0),
            "A_eff": c1.number_input("A_eff [mm2/m]", 500.0, 4000.0, 1200.0),
            "E": 210000,
            "h_w": c2.number_input("h_w [mm]", 15.0, 240.0, 63.0),
            "phi": c1.number_input("phi [deg]", 50.0, 90.0, 70.0),
            "r": c2.number_input("r [mm]", 1.0, 15.0, 4.0),
            "num_webs": int(c1.number_input("Webs", 2, 10, 5)),
            "pitch": c2.number_input("Pitch [mm]", 100.0, 350.0, 200.0),
            "b_top": 55, "b_bottom": 50
        }
        profile_name = "Manual"
    
    # 2. Geometry
    st.sidebar.subheader("2. Geometry")
    num_spans = st.sidebar.number_input("Number of spans", 1, 5, 1)
    spans = [st.sidebar.number_input(f"L{i+1} [m]", 0.5, 20.0, 6.0, key=f"L{i}") for i in range(int(num_spans))]
    total_length = sum(spans)
    
    # 3. Material
    st.sidebar.subheader("3. Material")
    steel_grade = st.sidebar.selectbox("Steel Grade", ["S350GD", "S320GD", "S280GD", "S250GD", "S420GD"])
    fy = {"S350GD": 350, "S320GD": 320, "S280GD": 280, "S250GD": 250, "S420GD": 420}[steel_grade]
    
    # 4. Loads
    st.sidebar.subheader("4. Loads")
    c1, c2 = st.sidebar.columns(2)
    g_k = c1.number_input("G_k [kN/m2]", 0.0, 10.0, 0.3)
    q_k = c2.number_input("Q_k [kN/m2]", 0.0, 20.0, 1.5)
    c1, c2, c3 = st.sidebar.columns(3)
    gamma_g = c1.number_input("gamma_G", 1.0, 1.5, 1.35)
    gamma_q = c2.number_input("gamma_Q", 1.0, 2.0, 1.5)
    gamma_m = c3.number_input("gamma_M0", 1.0, 1.2, 1.0)
    udl_uls = gamma_g * g_k + gamma_q * q_k
    udl_sls = g_k + q_k
    st.sidebar.info(f"q_ULS = {udl_uls:.2f} | q_SLS = {udl_sls:.2f} kN/m2")
    
    # 5. Point Loads
    st.sidebar.subheader("5. Point Loads")
    num_pl = st.sidebar.number_input("Number of point loads", 0, 10, 0)
    point_loads = []
    for i in range(int(num_pl)):
        c1, c2 = st.sidebar.columns(2)
        p_g = c1.number_input(f"P{i+1}_G [kN]", 0.0, 50.0, 0.0, key=f"Pg{i}")
        p_q = c2.number_input(f"P{i+1}_Q [kN]", 0.0, 50.0, 2.0, key=f"Pq{i}")
        pos = st.sidebar.number_input(f"x{i+1} [m]", 0.0, total_length, min(3.0, total_length/2), key=f"x{i}")
        p_uls = gamma_g * p_g + gamma_q * p_q
        p_sls = p_g + p_q
        point_loads.append({"magnitude": p_uls, "magnitude_uls": p_uls, "magnitude_sls": p_sls, "position": pos})
    
    # 6. Local Effects
    st.sidebar.subheader("6. Local Effects (6.1.7.3)")
    s_s = st.sidebar.number_input("Bearing length s_s [mm]", 10.0, 200.0, 50.0)
    
    st.sidebar.markdown("**Support Category:**")
    st.sidebar.markdown("""
    **Category 1 (End):** Support at or near free end of sheet (c <= 1.5*h_w). 
    Uses alpha=0.075, l_a=10mm.
    
    **Category 2 (Internal):** Internal support or far from free end (c > 1.5*h_w).
    Uses alpha=0.15, l_a=s_s.
    """)
    wc_category = st.sidebar.radio("Category", [1, 2], index=1,
        format_func=lambda x: f"Cat.{x}: {'End (alpha=0.075)' if x==1 else 'Internal (alpha=0.15)'}")
    
    st.sidebar.markdown("**Load distribution:**")
    st.sidebar.markdown("""
    **All webs:** Support reaction from UDL - load distributed across all webs within bearing width.
    
    **1 web only:** True point load (equipment leg, hanger) - acts on single web directly beneath.
    
    **2 webs:** Point load with small spread - affects 2 adjacent webs.
    """)
    num_webs_effective = st.sidebar.radio("Effective webs for point loads", 
        ["All webs", "1 web", "2 webs"], index=0,
        help="How many webs resist the point load")
    
    calc_btn = st.sidebar.button("CALCULATE", type="primary", use_container_width=True)
    
    # === MAIN CONTENT ===
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile")
        fig_profile = draw_profile_cross_section(props, profile_name)
        st.pyplot(fig_profile)
        plt.close()
        
        with st.expander("Section Properties", expanded=True):
            st.markdown(f"""
            | Property | Value | Unit |
            |----------|-------|------|
            | Type | {props.get('type', 'Custom')} | - |
            | h | {props['h']} | mm |
            | t | {props['t']} | mm |
            | I_pos | {props.get('I_pos', 0):.1f} | cm4/m |
            | I_neg | {props.get('I_neg', 0):.1f} | cm4/m |
            | W_pos | {props.get('W_pos', 0):.1f} | cm3/m |
            | W_neg | {props.get('W_neg', 0):.1f} | cm3/m |
            | A_eff | {props.get('A_eff', 0):.0f} | mm2/m |
            | phi | {props.get('phi', 70)} | deg |
            | r | {props.get('r', 4)} | mm |
            | Webs | {props.get('num_webs', 2)} | - |
            """)
    
    with col2:
        st.subheader("Loading")
        fig_beam = draw_beam_diagram(spans, udl_uls, udl_sls, point_loads)
        st.pyplot(fig_beam)
        plt.close()
    
    if calc_btn:
        st.markdown("---")
        st.header("Calculation Results")
        
        I_calc = props.get("I_pos", 50)
        
        # Calculate internal forces
        if num_spans == 1:
            x_uls, M_uls, V_uls, _ = calculate_single_span(spans[0], udl_uls, point_loads, props["E"], I_calc)
            pl_sls = [{"magnitude": pl["magnitude_sls"], "position": pl["position"]} for pl in point_loads]
            x_sls, _, _, defl_sls = calculate_single_span(spans[0], udl_sls, pl_sls, props["E"], I_calc)
            reactions = None
        else:
            x_uls, M_uls, V_uls, _, reactions = calculate_multi_span(spans, udl_uls, point_loads, props["E"], I_calc)
            pl_sls = [{"magnitude": pl["magnitude_sls"], "position": pl["position"]} for pl in point_loads]
            x_sls, _, _, defl_sls, _ = calculate_multi_span(spans, udl_sls, pl_sls, props["E"], I_calc)
        
        # Calculate resistances
        res = calculate_resistances(props, fy, gamma_m)
        M_Rd_pos = res["M_Rd_pos"]
        M_Rd_neg = res["M_Rd_neg"]
        V_Rd = res["V_Rd"]
        
        M_Ed_pos = np.max(M_uls)
        M_Ed_neg = abs(np.min(M_uls)) if np.min(M_uls) < 0 else 0
        V_Ed = np.max(np.abs(V_uls))
        defl_max = np.max(np.abs(defl_sls))
        defl_limit = max(spans) * 1000 / 200
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("EN 1993-1-3 Verification")
            
            # Summary table
            checks = []
            util_m_pos = M_Ed_pos / M_Rd_pos if M_Rd_pos > 0 else 0
            checks.append(["6.1.4", "Bending (sagging)", f"{M_Ed_pos:.2f}/{M_Rd_pos:.2f}", 
                          f"{util_m_pos*100:.1f}%", "OK" if util_m_pos <= 1 else "NOT OK"])
            
            if num_spans > 1 and M_Ed_neg > 0:
                util_m_neg = M_Ed_neg / M_Rd_neg if M_Rd_neg > 0 else 0
                checks.append(["6.1.4", "Bending (hogging)", f"{M_Ed_neg:.2f}/{M_Rd_neg:.2f}",
                              f"{util_m_neg*100:.1f}%", "OK" if util_m_neg <= 1 else "NOT OK"])
            
            util_v = V_Ed / V_Rd if V_Rd > 0 else 0
            checks.append(["6.1.5", "Shear", f"{V_Ed:.2f}/{V_Rd:.2f}",
                          f"{util_v*100:.1f}%", "OK" if util_v <= 1 else "NOT OK"])
            
            comb_mv = check_combined_bending_shear(M_Ed_pos, V_Ed, M_Rd_pos, V_Rd)
            checks.append(["6.1.10", "M+V Combined", f"M_Rd,red={comb_mv['M_Rd_reduced']:.2f}",
                          f"{comb_mv['utilization']*100:.1f}%", comb_mv["status"]])
            
            util_defl = defl_max / defl_limit if defl_limit > 0 else 0
            checks.append(["7.3", "Deflection (SLS)", f"{defl_max:.2f}/{defl_limit:.1f} mm",
                          f"{util_defl*100:.1f}%", "OK" if util_defl <= 1 else "NOT OK"])
            
            df = pd.DataFrame(checks, columns=["Clause", "Check", "Ed/Rd", "Util.", "Status"])
            
            def color_status(val):
                if val == "OK":
                    return "background-color: #90EE90"
                elif "NOT OK" in str(val):
                    return "background-color: #FFB6C1"
                return ""
            
            st.dataframe(df.style.applymap(color_status, subset=["Status"]), hide_index=True, use_container_width=True)
            
            # Detailed resistance calculations
            with st.expander("Detailed Resistance Calculation (6.1.4, 6.1.5)"):
                st.markdown(f"""
                **Bending Resistance (EN 1993-1-3, 6.1.4):**
                
                M_c,Rd = W_eff * f_yb / gamma_M0
                
                - W_eff,pos = {res['W_pos']:.1f} cm3/m = {res['W_pos']*1e-6:.2e} m3/m
                - f_yb = {fy} MPa = {fy*1e6:.0f} N/m2
                - gamma_M0 = {gamma_m}
                
                M_c,Rd,pos = {res['W_pos']*1e-6:.2e} * {fy*1e6:.0f} / {gamma_m} = {M_Rd_pos*1000:.0f} Nm/m = **{M_Rd_pos:.2f} kNm/m**
                
                ---
                
                **Shear Resistance (EN 1993-1-3, 6.1.5):**
                
                V_Rd = A_eff * f_yb / (sqrt(3) * gamma_M0)
                
                - A_eff = {props['A_eff']:.0f} mm2/m = {props['A_eff']*1e-6:.2e} m2/m
                
                V_Rd = {props['A_eff']*1e-6:.2e} * {fy*1e6:.0f} / ({np.sqrt(3):.4f} * {gamma_m}) = **{V_Rd:.2f} kN/m**
                """)
            
            # Local effects detailed
            if point_loads:
                st.subheader("Local Effects - Web Crippling (6.1.7.3)")
                
                # Determine effective number of webs
                if num_webs_effective == "All webs":
                    n_webs_eff = props.get("num_webs", 2)
                elif num_webs_effective == "1 web":
                    n_webs_eff = 1
                else:
                    n_webs_eff = 2
                
                for i, pl in enumerate(point_loads):
                    wc = check_web_crippling(pl["magnitude_uls"], props, fy, gamma_m, s_s, wc_category, n_webs_eff)
                    comb = check_combined_bending_web_crippling(M_Ed_pos, pl["magnitude_uls"], M_Rd_pos, wc["R_w_Rd"])
                    
                    status_color = "green" if wc["status"]=="OK" and comb["status"]=="OK" else "red"
                    
                    st.markdown(f"### Point Load P{i+1} = {pl['magnitude_uls']:.2f} kN")
                    
                    # Warning about load distribution
                    if n_webs_eff < props.get("num_webs", 2):
                        st.warning(f"Using {n_webs_eff} effective web(s) - true point load assumption")
                    else:
                        st.info(f"Using all {n_webs_eff} webs - load distributed (support reaction assumption)")
                    
                    st.markdown(f"""
                    **Web Crippling Resistance (EN 1993-1-3, 6.1.7.3, Formula 6.18):**
                    
                    Category {wc['category']}: {'End support (c <= 1.5*h_w)' if wc['category']==1 else 'Internal support'}
                    
                    R_w,Rd = alpha * t^2 * sqrt(f_yb * E) * k1 * k2 * k3 / gamma_M1
                    
                    **Input parameters:**
                    | Parameter | Value | Description |
                    |-----------|-------|-------------|
                    | t | {wc['t']} mm | Sheet thickness |
                    | r | {wc['r']} mm | Corner radius |
                    | phi | {wc['phi']} deg | Web angle |
                    | f_yb | {fy} MPa | Yield strength |
                    | E | {props['E']} MPa | Elastic modulus |
                    | alpha | {wc['alpha']} | Category coefficient |
                    | l_a | {wc['l_a']} mm | Effective bearing length |
                    | n_webs (total) | {wc['num_webs']} | Total webs in profile |
                    | **n_webs (effective)** | **{wc['num_webs_eff']}** | **Webs resisting load** |
                    | gamma_M1 | {gamma_m} | Partial factor |
                    
                    **Intermediate calculations:**
                    - k1 = 1 - 0.1 * sqrt(r/t) = 1 - 0.1 * sqrt({wc['r']}/{wc['t']}) = **{wc['k1']:.4f}**
                    - k2 = 0.5 + sqrt(0.02 * l_a / t) = 0.5 + sqrt(0.02 * {wc['l_a']} / {wc['t']}) = **{wc['k2']:.4f}**
                    - k3 = 2.4 + (phi/90)^2 = 2.4 + ({wc['phi']}/90)^2 = **{wc['k3']:.4f}**
                    - sqrt(f_yb * E) = sqrt({fy} * {props['E']}) = **{wc['sqrt_fy_E']:.2f}**
                    
                    **Resistance per web:**
                    R_w,Rd,single = {wc['alpha']} * {wc['t']}^2 * {wc['sqrt_fy_E']:.2f} * {wc['k1']:.4f} * {wc['k2']:.4f} * {wc['k3']:.4f} / {gamma_m}
                    R_w,Rd,single = **{wc['R_w_Rd_single']*1000:.1f} N = {wc['R_w_Rd_single']:.3f} kN**
                    
                    **Total resistance ({wc['num_webs_eff']} effective webs):**
                    R_w,Rd = {wc['R_w_Rd_single']:.3f} * {wc['num_webs_eff']} = **{wc['R_w_Rd']:.2f} kN**
                    
                    **Utilization:**
                    F_Ed / R_w,Rd = {pl['magnitude_uls']:.2f} / {wc['R_w_Rd']:.2f} = **{wc['utilization']*100:.1f}%** --> **{wc['status']}**
                    
                    ---
                    
                    **Combined Bending + Web Crippling (EN 1993-1-3, 6.1.11):**
                    
                    M_Ed/M_c,Rd + F_Ed/R_w,Rd <= 1.25
                    
                    - M_Ed/M_c,Rd = {M_Ed_pos:.2f} / {M_Rd_pos:.2f} = {comb['M_ratio']:.4f}
                    - F_Ed/R_w,Rd = {pl['magnitude_uls']:.2f} / {wc['R_w_Rd']:.2f} = {comb['F_ratio']:.4f}
                    - Interaction = {comb['M_ratio']:.4f} + {comb['F_ratio']:.4f} = **{comb['interaction']:.4f}**
                    - Limit = 1.25
                    - Utilization = {comb['interaction']:.4f} / 1.25 = **{comb['utilization']*100:.1f}%** --> **{comb['status']}**
                    """)
                    
                    st.markdown(f"**Result: :{status_color}[{wc['status']}]**")
                    st.markdown("---")
            
            if reactions:
                st.subheader("Support Reactions (ULS)")
                cols = st.columns(len(reactions))
                for i, (col, R) in enumerate(zip(cols, reactions)):
                    col.metric(f"R{i+1}", f"{R:.2f} kN/m")
        
        with col2:
            st.subheader("Summary")
            all_ok = util_m_pos <= 1 and util_v <= 1 and util_defl <= 1 and comb_mv["utilization"] <= 1
            
            if all_ok:
                st.success("ALL CHECKS PASSED")
            else:
                st.error("SOME CHECKS FAILED")
            
            st.metric("M_Ed,max", f"{max(M_Ed_pos, M_Ed_neg):.2f} kNm/m")
            st.metric("V_Ed,max", f"{V_Ed:.2f} kN/m")
            st.metric("d_max", f"{defl_max:.2f} mm")
            st.metric("M_Rd,pos", f"{M_Rd_pos:.2f} kNm/m")
            st.metric("V_Rd", f"{V_Rd:.2f} kN/m")
        
        st.markdown("---")
        st.subheader("Diagrams")
        
        tab1, tab2 = st.tabs(["Internal Forces (ULS)", "Deflection (SLS)"])
        
        with tab1:
            fig = plot_internal_forces(x_uls, M_uls, V_uls, spans, point_loads)
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            fig = plot_deflection(x_sls, defl_sls, spans, defl_limit, point_loads)
            st.pyplot(fig)
            plt.close()
    
    else:
        st.info("Configure parameters and click CALCULATE")

if __name__ == "__main__":
    main()
