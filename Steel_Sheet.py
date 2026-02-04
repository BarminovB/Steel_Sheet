"""
Profiled Steel Sheet Calculator v3.4
EN 1993-1-3 compliant with full verification
Includes: 6.1.4, 6.1.5, 6.1.7, 6.1.10, 6.1.11, 7.2, 7.3
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
    """Full Ruukki profile database with geometric data for all checks"""
    db = {}
    
    # T45-30L-905: h=45, pitch=151, 6 webs
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
    profile_x, profile_y = [], []
    x_current = 0
    for rib in range(num_ribs + 1):
        if rib == 0:
            profile_x.append(x_current)
            profile_y.append(0)
        profile_x.extend([x_current + b_bottom/2, x_current + b_bottom/2 + web_offset,
                         x_current + b_bottom/2 + web_offset + b_top,
                         x_current + b_bottom/2 + 2*web_offset + b_top])
        profile_y.extend([0, h, h, 0])
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
    ax.text(total_width * 0.8, h + 15, f"t = {t} mm", fontsize=11, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8), fontweight="bold")
    ax.set_xlim(-50, total_width + 30)
    ax.set_ylim(-20, h + 35)
    ax.set_aspect("equal")
    ax.set_xlabel("Width [mm]")
    ax.set_ylabel("Height [mm]")
    ax.set_title(f"Profile: {profile_name}", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
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
                                [x_sup, beam_y - 0.30]], closed=True, facecolor="dimgray")
        ax.add_patch(triangle)
        ax.text(x_sup, beam_y - 0.45, f"{i+1}", ha="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="lightgray"))
    if udl_uls > 0:
        n_arrows = max(int(total_length / 0.25), 10)
        for x_arr in np.linspace(0.05, total_length - 0.05, n_arrows):
            ax.annotate("", xy=(x_arr, beam_y + 0.03), xytext=(x_arr, beam_y + 0.35),
                       arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        ax.text(total_length / 2, beam_y + 0.50, f"ULS: {udl_uls:.2f} | SLS: {udl_sls:.2f} kN/m2",
                ha="center", fontsize=11, color="darkred", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.95))
    for i, pl in enumerate(point_loads):
        x_pl = pl["position"]
        p_uls = pl.get("magnitude_uls", pl.get("magnitude", 0))
        ax.annotate("", xy=(x_pl, beam_y + 0.03), xytext=(x_pl, beam_y + 0.65),
                   arrowprops=dict(arrowstyle="->", color="blue", lw=2.5))
        ax.text(x_pl, beam_y + 0.75, f"P{i+1}={p_uls:.1f}kN", ha="center", fontsize=9,
                color="darkblue", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
    cumulative = 0
    for i, span in enumerate(spans):
        mid = cumulative + span / 2
        ax.text(mid, beam_y - 0.72, f"L{i+1}={span:.2f}m", ha="center", fontsize=10, color="darkgreen", fontweight="bold")
        cumulative += span
    ax.set_xlim(-0.4, total_length + 0.4)
    ax.set_ylim(-0.95, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
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
    EI = E * 1e6 * I * 1e-8
    M_Nm = M * 1000
    theta = cumulative_trapezoid(M_Nm / EI, x, initial=0)
    y_raw = cumulative_trapezoid(theta, x, initial=0)
    y_corrected = y_raw - (y_raw[-1] / span) * x
    defl = -y_corrected * 1000
    return x, M, V, defl

def calculate_multi_span(spans, udl, point_loads, E, I):
    num_spans = len(spans)
    if num_spans == 1:
        x, M, V, defl = calculate_single_span(spans[0], udl, point_loads, E, I)
        Ra = udl * spans[0] / 2
        Rb = udl * spans[0] - Ra
        return x, M, V, defl, [Ra, Rb]
    total_length = sum(spans)
    n = 1001
    n_unknowns = num_spans - 1
    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)
    for i in range(n_unknowns):
        L_i, L_ip1 = spans[i], spans[i + 1]
        A[i, i] = 2 * (L_i + L_ip1)
        if i > 0: A[i, i-1] = L_i
        if i < n_unknowns - 1: A[i, i+1] = L_ip1
        b[i] = -udl / 4 * (L_i**3 + L_ip1**3)
    M_supports = solve(A, b)
    M_sup_full = np.concatenate([[0], M_supports, [0]])
    x = np.linspace(0, total_length, n)
    M, V = np.zeros(n), np.zeros(n)
    cumulative = 0
    reactions = []
    for span_idx, span_len in enumerate(spans):
        M_left, M_right = M_sup_full[span_idx], M_sup_full[span_idx + 1]
        mask = (x >= cumulative - 1e-9) & (x <= cumulative + span_len + 1e-9)
        x_local = x[mask] - cumulative
        M[mask] = udl * x_local * (span_len - x_local) / 2 + M_left + (M_right - M_left) * x_local / span_len
        R_left = udl * span_len / 2 - (M_right - M_left) / span_len
        if span_idx == 0: reactions.append(R_left)
        V[mask] = R_left - udl * x_local
        if span_idx < num_spans - 1: reactions.append(-V[mask][-1])
        cumulative += span_len
    reactions.append(-V[-1])
    EI = E * 1e6 * I * 1e-8
    defl = np.zeros(n)
    cumulative = 0
    for span_idx, span_len in enumerate(spans):
        mask = (x >= cumulative - 1e-9) & (x <= cumulative + span_len + 1e-9)
        indices = np.where(mask)[0]
        if len(indices) > 1:
            x_span, M_span = x[indices], M[indices]
            theta = cumulative_trapezoid(M_span * 1000 / EI, x_span, initial=0)
            y_raw = cumulative_trapezoid(theta, x_span, initial=0)
            defl[indices] = -(y_raw - (y_raw[-1] / span_len) * (x_span - cumulative)) * 1000
        cumulative += span_len
    return x, M, V, defl, reactions

# ============================================================================
# EN 1993-1-3 CHECKS
# ============================================================================

def calculate_resistances(props, fy, gamma_m):
    """6.1.4 Bending, 6.1.5 Shear"""
    W_pos = props.get("W_pos", 20) * 1e-6
    W_neg = props.get("W_neg", W_pos * 1e6 * 0.9) * 1e-6
    f_Pa = fy * 1e6
    M_Rd_pos = W_pos * f_Pa / gamma_m / 1000
    M_Rd_neg = W_neg * f_Pa / gamma_m / 1000
    A_eff = props["A_eff"] * 1e-6
    V_Rd = A_eff * f_Pa / (np.sqrt(3) * gamma_m) / 1000
    return {"M_Rd_pos": M_Rd_pos, "M_Rd_neg": M_Rd_neg, "V_Rd": V_Rd,
            "W_pos": props.get("W_pos", 20), "W_neg": props.get("W_neg", 18)}

def check_web_shear_buckling(props, fy):
    """6.1.7.1(2) - Shear buckling of unstiffened web"""
    t = props["t"]
    h_w = props.get("h_w", props["h"] - 10)
    phi = props["phi"]
    E = props["E"]
    
    s_w = h_w / np.sin(np.radians(phi))
    ratio = s_w / t
    limit = 0.346 * np.sqrt(E / fy) * (np.sin(np.radians(phi)))**0.7
    
    return {
        "s_w": s_w,
        "ratio": ratio,
        "limit": limit,
        "status": "OK" if ratio <= limit else "Check 6.1.5(1)",
        "needs_full_check": ratio > limit
    }

def check_web_crippling(F_Ed, props, fy, gamma_m, s_s, load_type, edge_dist, num_webs_eff=None):
    """
    EN 1993-1-3, 6.1.7.3 - Web crippling for sheeting
    
    load_type: Type of local load
        - "IOF": Interior One Flange (single load from one side)
        - "EOF": End One Flange (single load at end)
        - "ITF": Interior Two Flanges (opposite loads)
        - "ETF": End Two Flanges (opposite loads at end)
    
    edge_dist: Distance c from sheet edge to support edge [mm]
        If c <= 1.5*h_w -> Category 1
        If c > 1.5*h_w -> Category 2
    """
    t = props["t"]
    r = props["r"]
    phi = props["phi"]
    E = props["E"]
    h_w = props.get("h_w", props["h"] - 10)
    num_webs = props.get("num_webs", 2)
    
    if num_webs_eff is None:
        num_webs_eff = num_webs
    
    # Determine category based on edge distance
    cat_limit = 1.5 * h_w
    if edge_dist <= cat_limit:
        category = 1
        alpha = 0.075
        l_a = 10.0
        cat_desc = f"Category 1: c={edge_dist:.0f}mm <= 1.5*h_w={cat_limit:.0f}mm (End)"
    else:
        category = 2
        alpha = 0.15
        l_a = s_s
        cat_desc = f"Category 2: c={edge_dist:.0f}mm > 1.5*h_w={cat_limit:.0f}mm (Internal)"
    
    # Check validity limits (6.1.7.3(2))
    r_t_check = r / t <= 10
    phi_check = phi >= 45
    
    k1 = 1 - 0.1 * np.sqrt(r / t)
    k2 = 0.5 + np.sqrt(0.02 * l_a / t)
    k3 = 2.4 + (phi / 90)**2
    sqrt_fy_E = np.sqrt(fy * E)
    
    R_w_Rd_single = alpha * t**2 * sqrt_fy_E * k1 * k2 * k3 / gamma_m
    R_w_Rd = R_w_Rd_single * num_webs_eff / 1000
    
    utilization = F_Ed / R_w_Rd if R_w_Rd > 0 else float("inf")
    
    return {
        "R_w_Rd": R_w_Rd,
        "R_w_Rd_single": R_w_Rd_single / 1000,
        "utilization": utilization,
        "status": "OK" if utilization <= 1.0 else "NOT OK",
        "category": category,
        "cat_desc": cat_desc,
        "alpha": alpha,
        "l_a": l_a,
        "k1": k1, "k2": k2, "k3": k3,
        "sqrt_fy_E": sqrt_fy_E,
        "t": t, "r": r, "phi": phi,
        "h_w": h_w,
        "num_webs": num_webs,
        "num_webs_eff": num_webs_eff,
        "load_type": load_type,
        "edge_dist": edge_dist,
        "r_t_check": r_t_check,
        "phi_check": phi_check,
        "validity": "OK" if (r_t_check and phi_check) else "Check limits"
    }

def check_flange_buckling(props, fy):
    """5.5.3.2 - Effective width of compression flange"""
    t = props["t"]
    b_p = props.get("b_top", 50)
    E = props["E"]
    
    epsilon = np.sqrt(235 / fy)
    k_sigma = 4.0  # uniform compression
    
    lambda_p = (b_p / t) / (28.4 * epsilon * np.sqrt(k_sigma))
    
    if lambda_p <= 0.673:
        rho = 1.0
    else:
        psi = 1.0
        rho = (1 - 0.055 * (3 + psi) / lambda_p) / lambda_p
        rho = min(rho, 1.0)
    
    b_eff = rho * b_p
    
    return {
        "b_p": b_p,
        "lambda_p": lambda_p,
        "rho": rho,
        "b_eff": b_eff,
        "reduction": (1 - rho) * 100,
        "status": "OK" if rho >= 0.8 else "Significant reduction"
    }

def check_local_flange_deflection(props, q_sls):
    """7.2 - Serviceability of profiled sheeting (local deflection)"""
    t = props["t"]
    b_p = props.get("b_top", 50)
    E = props["E"]
    
    # Plate deflection coefficient for fixed edges
    k = 0.0138  # approximate for long plate with fixed edges
    
    # q in N/mm2, b in mm, t in mm, E in MPa
    q_Nmm2 = q_sls / 1000  # kN/m2 -> N/mm2
    
    delta_local = k * q_Nmm2 * b_p**4 / (E * t**3)
    limit = b_p / 100
    
    return {
        "delta_local": delta_local,
        "limit": limit,
        "utilization": delta_local / limit if limit > 0 else 0,
        "status": "OK" if delta_local <= limit else "NOT OK",
        "b_p": b_p
    }

def check_combined_bending_shear(M_Ed, V_Ed, M_Rd, V_Rd):
    """6.1.10 - Combined bending and shear"""
    if V_Ed <= 0.5 * V_Rd:
        M_Rd_red = M_Rd
        rho = 0
    else:
        rho = (2 * V_Ed / V_Rd - 1)**2
        M_Rd_red = M_Rd * (1 - rho)
    util = M_Ed / M_Rd_red if M_Rd_red > 0 else float("inf")
    return {"M_Rd_reduced": M_Rd_red, "utilization": util, "status": "OK" if util <= 1.0 else "NOT OK", "rho": rho}

def check_combined_bending_web_crippling(M_Ed, F_Ed, M_Rd, R_w_Rd):
    """6.1.11 - Combined bending and local load"""
    if M_Rd <= 0 or R_w_Rd <= 0:
        return {"interaction": float("inf"), "status": "NOT OK", "M_ratio": 0, "F_ratio": 0}
    M_ratio = M_Ed / M_Rd
    F_ratio = F_Ed / R_w_Rd
    interaction = M_ratio + F_ratio
    return {"interaction": interaction, "utilization": interaction / 1.25,
            "status": "OK" if interaction <= 1.25 else "NOT OK",
            "M_ratio": M_ratio, "F_ratio": F_ratio}

# ============================================================================
# PLOTTING
# ============================================================================

def plot_internal_forces(x, M, V, spans):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.fill_between(x, 0, V, where=(V >= 0), alpha=0.3, color="blue")
    ax1.fill_between(x, 0, V, where=(V < 0), alpha=0.3, color="red")
    ax1.plot(x, V, "k-", linewidth=1.5)
    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.set_ylabel("Shear V [kN/m]")
    ax1.set_title("Shear Force (ULS)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(x, 0, M, where=(M >= 0), alpha=0.3, color="red")
    ax2.fill_between(x, 0, M, where=(M < 0), alpha=0.3, color="blue")
    ax2.plot(x, M, "k-", linewidth=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.invert_yaxis()
    ax2.set_ylabel("Moment M [kNm/m]")
    ax2.set_xlabel("Position x [m]")
    ax2.set_title("Bending Moment (ULS) - Positive DOWN", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_deflection(x, defl, spans, defl_limit):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, 0, defl, alpha=0.3, color="orange")
    ax.plot(x, defl, "b-", linewidth=2, label="Deflection")
    ax.axhline(y=defl_limit, color="red", linestyle="--", linewidth=2, label=f"L/200 = {defl_limit:.1f} mm")
    idx_max = np.argmax(np.abs(defl))
    d_max = defl[idx_max]
    ax.plot(x[idx_max], d_max, "ro", markersize=10)
    ax.annotate(f"max = {d_max:.2f} mm", xy=(x[idx_max], d_max),
                xytext=(x[idx_max] + 0.3, d_max + defl_limit*0.1), fontsize=10, color="red")
    ax.invert_yaxis()
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Deflection [mm]")
    ax.set_title("Deflection (SLS)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Steel Sheet v3.4", page_icon="S", layout="wide")
    st.title("Steel Sheet Calculator v3.4")
    st.markdown("**EN 1993-1-3 | Profiled Steel Sheet Design & Verification**")
    
    if "copied_profile" not in st.session_state:
        st.session_state.copied_profile = None
    
    # === SIDEBAR ===
    st.sidebar.header("Input")
    
    # Profile
    st.sidebar.subheader("1. Profile")
    profile_mode = st.sidebar.radio("Mode", ["Catalog", "Copy & Modify", "Manual"])
    
    if profile_mode == "Catalog":
        profile_types = get_profile_types()
        selected_type = st.sidebar.selectbox("Type", profile_types)
        available_t = get_thicknesses_for_type(selected_type)
        selected_t = st.sidebar.selectbox("Thickness", available_t, format_func=lambda x: f"{x} mm")
        profile_name = f"{selected_type}-{selected_t}"
        props = copy.deepcopy(get_profile_database().get(profile_name, {}))
        if st.sidebar.button("Copy for modification"):
            st.session_state.copied_profile = copy.deepcopy(props)
            st.session_state.copied_profile["source_name"] = profile_name
            
    elif profile_mode == "Copy & Modify":
        if st.session_state.copied_profile is None:
            st.sidebar.warning("Copy a profile from Catalog first")
            props = list(get_profile_database().values())[0]
            profile_name = "Select profile"
        else:
            base = st.session_state.copied_profile
            st.sidebar.success(f"Base: {base.get('source_name')}")
            new_t = st.sidebar.number_input("New thickness [mm]", 0.4, 2.0, float(base["t"]), 0.05)
            t_ratio = new_t / base["t"]
            props = copy.deepcopy(base)
            props["t"] = new_t
            for key in ["I_pos", "I_neg", "W_pos", "W_neg", "A_eff"]:
                if key in props:
                    props[key] = base[key] * t_ratio
            profile_name = f"{base.get('type')}-{new_t} (mod)"
            if st.sidebar.button("Clear"): 
                st.session_state.copied_profile = None
                st.rerun()
    else:
        c1, c2 = st.sidebar.columns(2)
        props = {
            "h": c1.number_input("h [mm]", 20.0, 250.0, 70.0),
            "t": c2.number_input("t [mm]", 0.4, 2.0, 0.9),
            "I_pos": c1.number_input("I_pos [cm4/m]", 10.0, 1500.0, 100.0),
            "W_pos": c2.number_input("W_pos [cm3/m]", 5.0, 200.0, 30.0),
            "W_neg": c1.number_input("W_neg [cm3/m]", 5.0, 200.0, 27.0),
            "A_eff": c2.number_input("A_eff [mm2/m]", 500.0, 4000.0, 1200.0),
            "I_neg": 90.0, "E": 210000,
            "h_w": c1.number_input("h_w [mm]", 15.0, 240.0, 63.0),
            "phi": c2.number_input("phi [deg]", 45.0, 90.0, 70.0),
            "r": c1.number_input("r [mm]", 1.0, 15.0, 4.0),
            "num_webs": int(c2.number_input("Webs", 2, 10, 5)),
            "pitch": 200, "b_top": 55, "b_bottom": 50
        }
        profile_name = "Manual"
    
    # Geometry
    st.sidebar.subheader("2. Geometry")
    num_spans = st.sidebar.number_input("Spans", 1, 5, 1)
    spans = [st.sidebar.number_input(f"L{i+1} [m]", 0.5, 20.0, 6.0, key=f"L{i}") for i in range(int(num_spans))]
    
    # Material
    st.sidebar.subheader("3. Material")
    steel_grade = st.sidebar.selectbox("Grade", ["S350GD", "S320GD", "S280GD", "S250GD"])
    fy = {"S350GD": 350, "S320GD": 320, "S280GD": 280, "S250GD": 250}[steel_grade]
    
    # Loads
    st.sidebar.subheader("4. Loads")
    c1, c2 = st.sidebar.columns(2)
    g_k = c1.number_input("G_k [kN/m2]", 0.0, 10.0, 0.3)
    q_k = c2.number_input("Q_k [kN/m2]", 0.0, 20.0, 1.5)
    gamma_g = c1.number_input("gamma_G", 1.0, 1.5, 1.35)
    gamma_q = c2.number_input("gamma_Q", 1.0, 2.0, 1.5)
    gamma_m = st.sidebar.number_input("gamma_M", 1.0, 1.2, 1.0)
    udl_uls = gamma_g * g_k + gamma_q * q_k
    udl_sls = g_k + q_k
    
    # Point loads
    st.sidebar.subheader("5. Point Loads")
    num_pl = st.sidebar.number_input("Count", 0, 10, 0)
    point_loads = []
    for i in range(int(num_pl)):
        c1, c2 = st.sidebar.columns(2)
        p_q = c1.number_input(f"P{i+1} [kN]", 0.0, 50.0, 2.0, key=f"Pq{i}")
        pos = c2.number_input(f"x{i+1} [m]", 0.0, sum(spans), 3.0, key=f"x{i}")
        point_loads.append({"magnitude": gamma_q * p_q, "magnitude_uls": gamma_q * p_q, 
                           "magnitude_sls": p_q, "position": pos})
    
    # Web crippling params
    st.sidebar.subheader("6. Web Crippling (6.1.7)")
    s_s = st.sidebar.number_input("Bearing length s_s [mm]", 10.0, 200.0, 50.0)
    
    st.sidebar.markdown("""
    **Category determination (6.1.7.3):**
    
    `c` = distance from sheet edge to nearest support edge
    
    `h_w` = web height (inclined)
    
    - **Category 1 (End):** `c <= 1.5*h_w`
      - Sheet edge close to/over support
      - alpha = 0.075, l_a = 10 mm
    
    - **Category 2 (Internal):** `c > 1.5*h_w`  
      - Sheet extends well past support
      - alpha = 0.15, l_a = s_s
    """)
    
    h_w_val = props.get("h_w", props.get("h", 70) - 10)
    st.sidebar.info(f"h_w = {h_w_val:.0f} mm, 1.5*h_w = {1.5*h_w_val:.0f} mm")
    
    edge_dist = st.sidebar.number_input("Edge distance c [mm]", 0.0, 1000.0, 50.0,
        help="Distance from sheet edge to support edge")
    
    st.sidebar.markdown("""
    **Load type (Table 6.7):**
    """)
    
    load_type = st.sidebar.selectbox("Load type", ["IOF", "EOF", "ITF", "ETF"],
        format_func=lambda x: {"IOF": "IOF - Interior One Flange", 
                               "EOF": "EOF - End One Flange",
                               "ITF": "ITF - Interior Two Flanges",
                               "ETF": "ETF - End Two Flanges"}[x])
    
    with st.sidebar.expander("Load types explained"):
        st.markdown("""
        ### One Flange (OF) vs Two Flanges (TF)
        
        **One Flange (OF)** - force on ONE flange only:
        ```
        Single sheet on support:
        
            /````\\
           /      \\
          /        \\
         +---------+  <- bottom flange
              ^
              |
           SUPPORT (force on bottom only)
        ```
        Typical case: sheet resting on purlin.
        
        ---
        
        **Two Flanges (TF)** - forces on BOTH flanges:
        ```
        Two sheets overlapping at support:
        
        Sheet 2:    /````\\
                   /      \\
                  v   v   v   <- presses DOWN on Sheet 1
        Sheet 1:    /````\\      (top flange loaded)
                   /      \\
                  +---------+
                       ^
                       |
                    SUPPORT    (bottom flange loaded)
        ```
        Web of Sheet 1 is squeezed from both sides.
        
        ---
        
        ### Interior (I) vs End (E)
        
        **End (E):** Support near sheet edge
        ```
        Sheet edge
             |
             v
        [====|  <- sheet ends here
             ^
          SUPPORT at end
        ```
        
        **Interior (I):** Support far from edges
        ```
        [==========|==========]
                   ^
            SUPPORT in middle
            (sheet continues both ways)
        ```
        
        ---
        
        ### RECOMMENDATION
        
        | Case | Choice |
        |------|--------|
        | Single sheet on support | **OF** |
        | Sheets overlapping | **TF** |
        | End support | **E** |
        | Middle support | **I** |
        
        **Typical: EOF or IOF**
        
        **Safest: EOF** (lowest resistance)
        """)
    
    st.sidebar.markdown("""
    **Effective webs:**
    - Support reaction: All webs
    - Point load: 1-2 webs
    """)
    n_webs_opt = st.sidebar.radio("Effective webs", ["All", "1", "2"])
    
    calc_btn = st.sidebar.button("CALCULATE", type="primary", use_container_width=True)
    
    # === MAIN ===
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile")
        fig = draw_profile_cross_section(props, profile_name)
        st.pyplot(fig)
        plt.close()
        
        with st.expander("Properties", expanded=True):
            st.markdown(f"""
            | Prop | Value |
            |------|-------|
            | h | {props.get('h', 0)} mm |
            | t | {props.get('t', 0)} mm |
            | h_w | {props.get('h_w', 0)} mm |
            | phi | {props.get('phi', 70)} deg |
            | r | {props.get('r', 4)} mm |
            | I_pos | {props.get('I_pos', 0):.1f} cm4/m |
            | W_pos | {props.get('W_pos', 0):.1f} cm3/m |
            | A_eff | {props.get('A_eff', 0):.0f} mm2/m |
            | Webs | {props.get('num_webs', 2)} |
            """)
    
    with col2:
        st.subheader("Loading")
        fig = draw_beam_diagram(spans, udl_uls, udl_sls, point_loads)
        st.pyplot(fig)
        plt.close()
        st.info(f"q_ULS = {udl_uls:.2f} kN/m2 | q_SLS = {udl_sls:.2f} kN/m2")
    
    if calc_btn:
        st.markdown("---")
        st.header("Results")
        
        I_calc = props.get("I_pos", 50)
        if num_spans == 1:
            x_uls, M_uls, V_uls, _ = calculate_single_span(spans[0], udl_uls, point_loads, props["E"], I_calc)
            pl_sls = [{"magnitude": pl["magnitude_sls"], "position": pl["position"]} for pl in point_loads]
            x_sls, _, _, defl_sls = calculate_single_span(spans[0], udl_sls, pl_sls, props["E"], I_calc)
            reactions = None
        else:
            x_uls, M_uls, V_uls, _, reactions = calculate_multi_span(spans, udl_uls, point_loads, props["E"], I_calc)
            pl_sls = [{"magnitude": pl["magnitude_sls"], "position": pl["position"]} for pl in point_loads]
            x_sls, _, _, defl_sls, _ = calculate_multi_span(spans, udl_sls, pl_sls, props["E"], I_calc)
        
        res = calculate_resistances(props, fy, gamma_m)
        M_Rd_pos, M_Rd_neg, V_Rd = res["M_Rd_pos"], res["M_Rd_neg"], res["V_Rd"]
        M_Ed_pos = np.max(M_uls)
        M_Ed_neg = abs(np.min(M_uls)) if np.min(M_uls) < 0 else 0
        V_Ed = np.max(np.abs(V_uls))
        defl_max = np.max(np.abs(defl_sls))
        defl_limit = max(spans) * 1000 / 200
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("EN 1993-1-3 Checks")
            
            # Basic checks table
            checks = []
            util_m = M_Ed_pos / M_Rd_pos
            checks.append(["6.1.4", "Bending M+", f"{M_Ed_pos:.2f}/{M_Rd_pos:.2f}", f"{util_m*100:.0f}%", "OK" if util_m<=1 else "FAIL"])
            util_v = V_Ed / V_Rd
            checks.append(["6.1.5", "Shear", f"{V_Ed:.2f}/{V_Rd:.2f}", f"{util_v*100:.0f}%", "OK" if util_v<=1 else "FAIL"])
            comb_mv = check_combined_bending_shear(M_Ed_pos, V_Ed, M_Rd_pos, V_Rd)
            checks.append(["6.1.10", "M+V", f"rho={comb_mv['rho']:.2f}", f"{comb_mv['utilization']*100:.0f}%", comb_mv["status"]])
            util_d = defl_max / defl_limit
            checks.append(["7.3", "Deflection", f"{defl_max:.1f}/{defl_limit:.0f}mm", f"{util_d*100:.0f}%", "OK" if util_d<=1 else "FAIL"])
            
            # Additional checks
            shear_buck = check_web_shear_buckling(props, fy)
            checks.append(["6.1.7.1(2)", "Web buckling", f"s_w/t={shear_buck['ratio']:.0f}", f"lim={shear_buck['limit']:.0f}", shear_buck["status"]])
            
            flange_buck = check_flange_buckling(props, fy)
            checks.append(["5.5.3.2", "Flange eff.", f"rho={flange_buck['rho']:.2f}", f"-{flange_buck['reduction']:.0f}%", flange_buck["status"]])
            
            flange_defl = check_local_flange_deflection(props, udl_sls)
            checks.append(["7.2", "Local defl.", f"{flange_defl['delta_local']:.2f}mm", f"lim={flange_defl['limit']:.1f}mm", flange_defl["status"]])
            
            df = pd.DataFrame(checks, columns=["Clause", "Check", "Values", "Util.", "Status"])
            
            def color_status(val):
                if val == "OK": return "background-color: #90EE90"
                elif val in ["FAIL", "NOT OK"]: return "background-color: #FFB6C1"
                return "background-color: #FFFACD"
            
            st.dataframe(df.style.applymap(color_status, subset=["Status"]), hide_index=True, use_container_width=True)
            
            # Web crippling details
            if point_loads:
                st.subheader("Web Crippling (6.1.7.3)")
                
                n_webs_eff = props.get("num_webs", 2) if n_webs_opt == "All" else int(n_webs_opt)
                
                for i, pl in enumerate(point_loads):
                    wc = check_web_crippling(pl["magnitude_uls"], props, fy, gamma_m, s_s, 
                                            load_type, edge_dist, n_webs_eff)
                    comb = check_combined_bending_web_crippling(M_Ed_pos, pl["magnitude_uls"], M_Rd_pos, wc["R_w_Rd"])
                    
                    st.markdown(f"### P{i+1} = {pl['magnitude_uls']:.2f} kN")
                    
                    status_color = "green" if wc["status"]=="OK" else "red"
                    
                    st.markdown(f"""
                    **{wc['cat_desc']}**
                    
                    Load type: {wc['load_type']} | Validity: {wc['validity']} (r/t={wc['r']/wc['t']:.1f}<=10, phi={wc['phi']}>=45)
                    
                    **Formula (6.18):** R_w,Rd = alpha * t^2 * sqrt(f_yb*E) * k1 * k2 * k3 / gamma_M1
                    
                    | Parameter | Formula | Value |
                    |-----------|---------|-------|
                    | alpha | Cat.{wc['category']} coeff | **{wc['alpha']}** |
                    | t | thickness | {wc['t']} mm |
                    | l_a | {'10 (fixed)' if wc['category']==1 else 's_s'} | {wc['l_a']} mm |
                    | k1 | 1 - 0.1*sqrt(r/t) | {wc['k1']:.4f} |
                    | k2 | 0.5 + sqrt(0.02*l_a/t) | {wc['k2']:.4f} |
                    | k3 | 2.4 + (phi/90)^2 | {wc['k3']:.4f} |
                    | sqrt(f_yb*E) | sqrt({fy}*{props['E']}) | {wc['sqrt_fy_E']:.1f} |
                    
                    **R_w,Rd (1 web)** = {wc['alpha']}*{wc['t']}^2*{wc['sqrt_fy_E']:.1f}*{wc['k1']:.4f}*{wc['k2']:.4f}*{wc['k3']:.4f}/{gamma_m} = **{wc['R_w_Rd_single']*1000:.0f} N = {wc['R_w_Rd_single']:.3f} kN**
                    
                    **R_w,Rd ({wc['num_webs_eff']} webs)** = {wc['R_w_Rd_single']:.3f} * {wc['num_webs_eff']} = **{wc['R_w_Rd']:.2f} kN**
                    
                    **Utilization:** {pl['magnitude_uls']:.2f} / {wc['R_w_Rd']:.2f} = **{wc['utilization']*100:.1f}%** --> :{status_color}[**{wc['status']}**]
                    
                    ---
                    **6.1.11 Combined M+F:** {comb['M_ratio']:.3f} + {comb['F_ratio']:.3f} = {comb['interaction']:.3f} <= 1.25 --> **{comb['status']}**
                    """)
        
        with col2:
            st.subheader("Summary")
            all_ok = util_m<=1 and util_v<=1 and util_d<=1
            if all_ok:
                st.success("PASS")
            else:
                st.error("FAIL")
            st.metric("M_Ed", f"{M_Ed_pos:.2f} kNm/m")
            st.metric("V_Ed", f"{V_Ed:.2f} kN/m")
            st.metric("d_max", f"{defl_max:.1f} mm")
        
        st.markdown("---")
        tab1, tab2 = st.tabs(["Forces", "Deflection"])
        with tab1:
            fig = plot_internal_forces(x_uls, M_uls, V_uls, spans)
            st.pyplot(fig)
            plt.close()
        with tab2:
            fig = plot_deflection(x_sls, defl_sls, spans, defl_limit)
            st.pyplot(fig)
            plt.close()
    
    else:
        st.info("Set parameters and click CALCULATE")

if __name__ == "__main__":
    main()
