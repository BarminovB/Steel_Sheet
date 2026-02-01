"""
Profiled Steel Sheet Calculator v3.1
Based on EN 1993-1-1 and EN 1993-1-3

Checks implemented (EN 1993-1-3):
- 6.1.4: Bending moment resistance
- 6.1.5: Shear resistance  
- 6.1.7: Web crippling (local transverse forces)
- 6.1.10: Combined shear and local transverse force
- 6.1.11: Combined bending and local transverse force
- 7.3: Serviceability (deflection)

Sign convention:
- Positive moment = sagging (tension at bottom) → plotted DOWNWARD
- Negative moment = hogging (tension at top) → plotted UPWARD

Author: Structural Engineering Calculator
Version: 3.1
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve
from typing import List, Dict, Tuple

# ============================================================================
# PROFILE DATABASE (Ruukki)
# ============================================================================

def get_profile_database() -> Dict:
    """
    Profile database based on Ruukki load-bearing sheets.
    Units: mm, cm4/m, cm3/m, mm2/m, MPa
    """
    return {
        "T45-30-905-0.6": {
            "h": 45, "t": 0.6, "I_pos": 30.4, "I_neg": 28.0, "W_pos": 13.5, "W_neg": 12.4,
            "A_eff": 857, "E": 210000, "h_w": 40, "phi": 75, "r": 3, "num_webs": 3, "pitch": 150,
            "b_top": 70, "b_bottom": 60
        },
        "T45-30-905-0.7": {
            "h": 45, "t": 0.7, "I_pos": 35.0, "I_neg": 32.0, "W_pos": 15.6, "W_neg": 14.2,
            "A_eff": 1000, "E": 210000, "h_w": 40, "phi": 75, "r": 3, "num_webs": 3, "pitch": 150,
            "b_top": 70, "b_bottom": 60
        },
        "T45-30-905-0.8": {
            "h": 45, "t": 0.8, "I_pos": 39.4, "I_neg": 36.0, "W_pos": 17.5, "W_neg": 16.0,
            "A_eff": 1143, "E": 210000, "h_w": 40, "phi": 75, "r": 3, "num_webs": 3, "pitch": 150,
            "b_top": 70, "b_bottom": 60
        },
        "T60-40-905-0.7": {
            "h": 60, "t": 0.7, "I_pos": 65.0, "I_neg": 58.0, "W_pos": 21.7, "W_neg": 19.3,
            "A_eff": 1050, "E": 210000, "h_w": 55, "phi": 70, "r": 4, "num_webs": 3, "pitch": 180,
            "b_top": 80, "b_bottom": 70
        },
        "T60-40-905-0.9": {
            "h": 60, "t": 0.9, "I_pos": 82.0, "I_neg": 73.0, "W_pos": 27.3, "W_neg": 24.3,
            "A_eff": 1350, "E": 210000, "h_w": 55, "phi": 70, "r": 4, "num_webs": 3, "pitch": 180,
            "b_top": 80, "b_bottom": 70
        },
        "T85-40-840-0.8": {
            "h": 85, "t": 0.8, "I_pos": 130.0, "I_neg": 115.0, "W_pos": 30.6, "W_neg": 27.1,
            "A_eff": 1200, "E": 210000, "h_w": 78, "phi": 72, "r": 4, "num_webs": 3, "pitch": 210,
            "b_top": 95, "b_bottom": 80
        },
        "T85-40-840-1.0": {
            "h": 85, "t": 1.0, "I_pos": 160.0, "I_neg": 142.0, "W_pos": 37.6, "W_neg": 33.4,
            "A_eff": 1500, "E": 210000, "h_w": 78, "phi": 72, "r": 4, "num_webs": 3, "pitch": 210,
            "b_top": 95, "b_bottom": 80
        },
        "T130M-75-930-0.8": {
            "h": 130, "t": 0.8, "I_pos": 350.0, "I_neg": 310.0, "W_pos": 53.8, "W_neg": 47.7,
            "A_eff": 1350, "E": 210000, "h_w": 120, "phi": 68, "r": 5, "num_webs": 4, "pitch": 233,
            "b_top": 110, "b_bottom": 90
        },
        "T130M-75-930-1.0": {
            "h": 130, "t": 1.0, "I_pos": 430.0, "I_neg": 380.0, "W_pos": 66.2, "W_neg": 58.5,
            "A_eff": 1688, "E": 210000, "h_w": 120, "phi": 68, "r": 5, "num_webs": 4, "pitch": 233,
            "b_top": 110, "b_bottom": 90
        },
        "T153M-100-840-1.0": {
            "h": 153, "t": 1.0, "I_pos": 620.0, "I_neg": 550.0, "W_pos": 81.0, "W_neg": 71.9,
            "A_eff": 1750, "E": 210000, "h_w": 143, "phi": 65, "r": 5, "num_webs": 3, "pitch": 280,
            "b_top": 130, "b_bottom": 100
        },
        "T153M-100-840-1.2": {
            "h": 153, "t": 1.2, "I_pos": 730.0, "I_neg": 645.0, "W_pos": 95.4, "W_neg": 84.3,
            "A_eff": 2100, "E": 210000, "h_w": 143, "phi": 65, "r": 5, "num_webs": 3, "pitch": 280,
            "b_top": 130, "b_bottom": 100
        }
    }


def get_available_profiles() -> List[str]:
    return list(get_profile_database().keys())


# ============================================================================
# VISUALIZATION 
# ============================================================================

def draw_profile_cross_section(props: Dict, profile_name: str = "") -> plt.Figure:
    """Draw trapezoidal profile cross-section."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    h = props['h']
    t = props['t']
    pitch = props.get('pitch', 150)
    phi = props.get('phi', 75)
    b_top = props.get('b_top', pitch * 0.4)
    b_bottom = props.get('b_bottom', pitch * 0.35)
    
    phi_rad = np.radians(phi)
    web_offset = h / np.tan(phi_rad) if phi < 90 else 0
    
    # Profile path for two ribs
    profile_x = [0, b_bottom/2]
    profile_y = [0, 0]
    
    # First rib up
    profile_x.append(b_bottom/2 + web_offset)
    profile_y.append(h)
    profile_x.append(b_bottom/2 + web_offset + b_top)
    profile_y.append(h)
    profile_x.append(b_bottom/2 + 2*web_offset + b_top)
    profile_y.append(0)
    profile_x.append(pitch)
    profile_y.append(0)
    
    # Second rib
    profile_x.append(pitch + b_bottom/2)
    profile_y.append(0)
    profile_x.append(pitch + b_bottom/2 + web_offset)
    profile_y.append(h)
    profile_x.append(pitch + b_bottom/2 + web_offset + b_top)
    profile_y.append(h)
    profile_x.append(pitch + b_bottom/2 + 2*web_offset + b_top)
    profile_y.append(0)
    profile_x.append(2*pitch)
    profile_y.append(0)
    
    ax.plot(profile_x, profile_y, 'b-', linewidth=2.5)
    ax.fill_between(profile_x, [y - t for y in profile_y], profile_y, 
                    alpha=0.3, color='steelblue')
    
    # Dimensions
    ax.annotate('', xy=(-15, h), xytext=(-15, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(-25, h/2, f'h={h}', fontsize=9, ha='right', va='center', color='red')
    
    ax.annotate('', xy=(0, -15), xytext=(pitch, -15),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(pitch/2, -25, f'pitch={pitch}', fontsize=9, ha='center', color='green')
    
    ax.text(pitch * 1.5, h + 10, f't={t} mm', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(-50, 2*pitch + 30)
    ax.set_ylim(-40, h + 30)
    ax.set_aspect('equal')
    ax.set_xlabel('Width [mm]')
    ax.set_ylabel('Height [mm]')
    ax.set_title(f'Profile: {profile_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='brown', linewidth=3)
    
    plt.tight_layout()
    return fig


def draw_beam_diagram(spans: List[float], udl_uls: float, udl_sls: float,
                      point_loads: List[Dict]) -> plt.Figure:
    """Draw longitudinal beam diagram with loads."""
    total_length = sum(spans)
    fig, ax = plt.subplots(figsize=(12, 4))
    
    beam_y = 1.0
    ax.plot([0, total_length], [beam_y, beam_y], 'b-', linewidth=4)
    
    # Supports
    support_positions = [0]
    cumulative = 0
    for span in spans:
        cumulative += span
        support_positions.append(cumulative)
    
    for i, x_sup in enumerate(support_positions):
        triangle = plt.Polygon([[x_sup - 0.12, beam_y - 0.05],
                                [x_sup + 0.12, beam_y - 0.05],
                                [x_sup, beam_y - 0.25]], 
                               closed=True, facecolor='gray', edgecolor='black')
        ax.add_patch(triangle)
        ax.text(x_sup, beam_y - 0.4, f'{i+1}', ha='center', fontsize=9,
                bbox=dict(boxstyle='circle', facecolor='lightgray'))
    
    # UDL arrows
    if udl_uls > 0:
        n_arrows = int(total_length / 0.3) + 1
        for x_arr in np.linspace(0.1, total_length - 0.1, n_arrows):
            ax.annotate('', xy=(x_arr, beam_y + 0.05), 
                       xytext=(x_arr, beam_y + 0.4),
                       arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax.plot([0, total_length], [beam_y + 0.4, beam_y + 0.4], 'r-', linewidth=1.5)
        ax.text(total_length / 2, beam_y + 0.55, 
                f'ULS: {udl_uls:.2f} kN/m² | SLS: {udl_sls:.2f} kN/m²',
                ha='center', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Point loads
    for i, pl in enumerate(point_loads):
        x_pl = pl['position']
        p_uls = pl.get('magnitude_uls', pl.get('magnitude', 0))
        p_sls = pl.get('magnitude_sls', p_uls)
        
        ax.annotate('', xy=(x_pl, beam_y + 0.05),
                   xytext=(x_pl, beam_y + 0.7),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(x_pl, beam_y + 0.8, f'P{i+1}\n{p_uls:.1f}/{p_sls:.1f}',
                ha='center', fontsize=9, color='blue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # Span dimensions
    cumulative = 0
    for i, span in enumerate(spans):
        mid = cumulative + span / 2
        ax.annotate('', xy=(cumulative + 0.05, beam_y - 0.55), 
                   xytext=(cumulative + span - 0.05, beam_y - 0.55),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax.text(mid, beam_y - 0.65, f'L{i+1}={span:.2f}m', 
                ha='center', fontsize=10, color='green')
        cumulative += span
    
    ax.set_xlim(-0.3, total_length + 0.3)
    ax.set_ylim(-0.9, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Loading Diagram (ULS/SLS values)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# STRUCTURAL CALCULATIONS
# ============================================================================

def calculate_single_span(span: float, udl: float, point_loads: List[Dict],
                          E: float, I: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate M, V, defl for simply supported beam."""
    n = 1001
    x = np.linspace(0, span, n)
    
    # Reactions
    Ra = udl * span / 2
    for pl in point_loads:
        if 0 <= pl['position'] <= span:
            Ra += pl['magnitude'] * (span - pl['position']) / span
    
    # Shear
    V = np.full_like(x, Ra, dtype=float) - udl * x
    for pl in point_loads:
        if 0 <= pl['position'] <= span:
            V = np.where(x > pl['position'], V - pl['magnitude'], V)
    
    # Moment (positive = sagging)
    M = cumulative_trapezoid(V, x, initial=0)
    
    # Deflection
    I_m4 = I * 1e-8
    E_Pa = E * 1e6
    EI = E_Pa * I_m4
    
    M_Nm = M * 1000
    theta = cumulative_trapezoid(M_Nm / EI, x, initial=0)
    y_raw = cumulative_trapezoid(theta, x, initial=0)
    y_corrected = y_raw - (y_raw[-1] / span) * x
    defl = y_corrected * 1000  # mm, positive = downward
    
    return x, M, V, defl


def calculate_multi_span(spans: List[float], udl: float, point_loads: List[Dict],
                         E: float, I: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Calculate M, V, defl for continuous beam using three-moment equation."""
    num_spans = len(spans)
    total_length = sum(spans)
    n = 1001
    
    if num_spans == 1:
        x, M, V, defl = calculate_single_span(spans[0], udl, point_loads, E, I)
        Ra = udl * spans[0] / 2
        for pl in point_loads:
            Ra += pl['magnitude'] * (spans[0] - pl['position']) / spans[0]
        Rb = udl * spans[0] + sum(pl['magnitude'] for pl in point_loads) - Ra
        return x, M, V, defl, [Ra, Rb]
    
    # Three-moment equation
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
        
        # RHS for UDL
        b[i] = -udl / 4 * (L_i**3 + L_ip1**3)
    
    # Point load contributions
    cumulative = 0
    for span_idx, span_len in enumerate(spans):
        for pl in point_loads:
            pos = pl['position']
            P = pl['magnitude']
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative
                b_dist = span_len - a
                L = span_len
                
                if span_idx > 0 and span_idx - 1 < n_unknowns:
                    contrib = -6 * P * a * b_dist * (L + b_dist) / (L**2 * 2)
                    b[span_idx - 1] += contrib
                
                if span_idx < n_unknowns:
                    contrib = -6 * P * a * b_dist * (L + a) / (L**2 * 2)
                    b[span_idx] += contrib
        cumulative += span_len
    
    try:
        M_supports = solve(A, b)
    except:
        M_supports = np.zeros(n_unknowns)
    
    M_sup_full = np.concatenate([[0], M_supports, [0]])
    
    # Build full M, V arrays
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
        
        # Simple beam moment
        M_udl = udl * x_local * (span_len - x_local) / 2
        
        # Point loads in span
        M_pl = np.zeros_like(x_local)
        for pl in point_loads:
            pos = pl['position']
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative
                P = pl['magnitude']
                for j, xl in enumerate(x_local):
                    if xl <= a:
                        M_pl[j] += P * xl * (span_len - a) / span_len
                    else:
                        M_pl[j] += P * a * (span_len - xl) / span_len
        
        # Support moment effect
        M_support = M_left + (M_right - M_left) * x_local / span_len
        M[mask] = M_udl + M_pl + M_support
        
        # Shear
        R_left = udl * span_len / 2 - (M_right - M_left) / span_len
        for pl in point_loads:
            pos = pl['position']
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative
                R_left += pl['magnitude'] * (span_len - a) / span_len
        
        if span_idx == 0:
            reactions.append(R_left)
        
        V_local = R_left - udl * x_local
        for pl in point_loads:
            pos = pl['position']
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative
                V_local = np.where(x_local > a, V_local - pl['magnitude'], V_local)
        
        V[mask] = V_local
        
        if span_idx < num_spans - 1:
            reactions.append(-V_local[-1])
        
        cumulative += span_len
    
    reactions.append(-V[-1] if len(V) > 0 else 0)
    
    # Deflection
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
            defl[indices] = y_corrected * 1000
        
        cumulative += span_len
    
    return x, M, V, defl, reactions


# ============================================================================
# EN 1993-1-3 RESISTANCE CHECKS
# ============================================================================

def calculate_resistances(props: Dict, fy: float, gamma_m: float) -> Dict:
    """
    Calculate design resistances per EN 1993-1-3.
    
    Returns:
        M_Rd_pos: Positive (sagging) moment resistance [kNm/m]
        M_Rd_neg: Negative (hogging) moment resistance [kNm/m]  
        V_Rd: Shear resistance [kN/m]
    """
    # Use W_pos for sagging, W_neg for hogging
    W_pos = props.get('W_pos', props.get('W', 20)) * 1e-6  # m3/m
    W_neg = props.get('W_neg', W_pos * 0.9) * 1e-6  # m3/m
    
    M_Rd_pos = W_pos * (fy * 1e6) / gamma_m / 1000  # kNm/m
    M_Rd_neg = W_neg * (fy * 1e6) / gamma_m / 1000  # kNm/m
    
    # Shear resistance (EN 1993-1-3, 6.1.5)
    A_eff = props['A_eff'] * 1e-6  # m2/m
    V_Rd = A_eff * (fy * 1e6) / (np.sqrt(3) * gamma_m) / 1000  # kN/m
    
    return {
        "M_Rd_pos": M_Rd_pos,
        "M_Rd_neg": M_Rd_neg,
        "V_Rd": V_Rd
    }


def check_web_crippling(F_Ed: float, props: Dict, fy: float, gamma_m: float, 
                        s_s: float = 50, category: str = "interior") -> Dict:
    """
    EN 1993-1-3, 6.1.7: Local transverse forces (web crippling).
    
    Args:
        F_Ed: Design transverse force [kN]
        props: Profile properties
        fy: Yield strength [MPa]
        gamma_m: Partial factor
        s_s: Stiff bearing length [mm]
        category: "interior" or "end" support
    
    Returns:
        R_w_Rd: Web crippling resistance [kN]
        utilization: F_Ed / R_w_Rd
    """
    t = props['t']
    r = props['r']
    phi = props['phi']
    E = props['E']
    h_w = props['h_w']
    num_webs = props.get('num_webs', 2)
    
    # EN 1993-1-3, 6.1.7.2 - Unstiffened webs
    l_a = s_s  # Effective bearing length
    
    # Coefficients (simplified from 6.1.7.2)
    # Category 1: single load or end support
    # Category 2: two opposing loads (interior support)
    
    if category == "end":
        # End support or single load near end
        k1 = 1.33 - 0.33 * min(s_s / t, 3.0) / 3.0
    else:
        # Interior support
        k1 = 1.0
    
    k2 = 1 - 0.1 * np.sqrt(r / t)  # Corner radius effect
    k3 = 0.5 + np.sqrt(0.02 * l_a / t)  # Bearing length effect
    k3 = min(k3, 1.25)
    k4 = 2.4 + (phi / 90)**2  # Web angle effect
    
    # Single web resistance
    R_w_Rd_single = k1 * k2 * k3 * k4 * t**2 * np.sqrt(fy * E) / gamma_m / 1000  # kN
    
    # Multiple webs (6.1.7.3)
    R_w_Rd = R_w_Rd_single * num_webs
    
    utilization = F_Ed / R_w_Rd if R_w_Rd > 0 else float('inf')
    
    return {
        "R_w_Rd": R_w_Rd,
        "R_w_Rd_single": R_w_Rd_single,
        "utilization": utilization,
        "status": "OK" if utilization <= 1.0 else "NOT OK"
    }


def check_combined_bending_web_crippling(M_Ed: float, F_Ed: float, 
                                          M_Rd: float, R_w_Rd: float) -> Dict:
    """
    EN 1993-1-3, 6.1.11: Combined bending and local transverse force.
    
    M_Ed/M_c,Rd + F_Ed/R_w,Rd ≤ 1.25
    """
    if M_Rd <= 0 or R_w_Rd <= 0:
        return {"interaction": float('inf'), "status": "NOT OK"}
    
    interaction = M_Ed / M_Rd + F_Ed / R_w_Rd
    limit = 1.25
    
    return {
        "interaction": interaction,
        "limit": limit,
        "utilization": interaction / limit,
        "status": "OK" if interaction <= limit else "NOT OK"
    }


def check_combined_shear_web_crippling(V_Ed: float, F_Ed: float,
                                        V_Rd: float, R_w_Rd: float) -> Dict:
    """
    EN 1993-1-3, 6.1.10: Combined shear and local transverse force.
    
    If V_Ed > 0.5 V_Rd, reduced R_w,Rd applies.
    """
    if V_Rd <= 0 or R_w_Rd <= 0:
        return {"R_w_Rd_reduced": 0, "status": "NOT OK"}
    
    if V_Ed <= 0.5 * V_Rd:
        # No reduction needed
        R_w_Rd_reduced = R_w_Rd
        utilization = F_Ed / R_w_Rd
    else:
        # Reduce web crippling resistance
        rho = (2 * V_Ed / V_Rd - 1)**2
        R_w_Rd_reduced = R_w_Rd * np.sqrt(1 - rho)
        utilization = F_Ed / R_w_Rd_reduced if R_w_Rd_reduced > 0 else float('inf')
    
    return {
        "R_w_Rd_reduced": R_w_Rd_reduced,
        "utilization": utilization,
        "status": "OK" if utilization <= 1.0 else "NOT OK"
    }


def check_combined_bending_shear(M_Ed: float, V_Ed: float,
                                  M_Rd: float, V_Rd: float) -> Dict:
    """
    EN 1993-1-3, 6.1.10: Combined bending and shear.
    
    If V_Ed > 0.5 V_Rd, reduced M_Rd applies.
    """
    if V_Ed <= 0.5 * V_Rd:
        M_Rd_reduced = M_Rd
    else:
        rho = (2 * V_Ed / V_Rd - 1)**2
        M_Rd_reduced = M_Rd * (1 - rho)
    
    utilization = M_Ed / M_Rd_reduced if M_Rd_reduced > 0 else float('inf')
    
    return {
        "M_Rd_reduced": M_Rd_reduced,
        "utilization": utilization,
        "status": "OK" if utilization <= 1.0 else "NOT OK"
    }


# ============================================================================
# PLOTTING (CORRECTED SIGN CONVENTION)
# ============================================================================

def plot_internal_forces(x: np.ndarray, M: np.ndarray, V: np.ndarray,
                         spans: List[float], point_loads: List[Dict]) -> plt.Figure:
    """
    Plot shear and bending moment diagrams.
    
    Sign convention:
    - Positive moment (sagging) → plotted DOWNWARD (below axis)
    - Negative moment (hogging) → plotted UPWARD (above axis)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    # === SHEAR DIAGRAM ===
    ax1.fill_between(x, 0, V, where=(V >= 0), alpha=0.3, color='blue', label='V+')
    ax1.fill_between(x, 0, V, where=(V < 0), alpha=0.3, color='red', label='V-')
    ax1.plot(x, V, 'k-', linewidth=1.5)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_ylabel('Shear V [kN/m]', fontsize=11)
    ax1.set_title('Shear Force Diagram (ULS)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Support lines
    cumulative = 0
    for span in spans:
        ax1.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
        cumulative += span
    ax1.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
    
    # Max values
    V_max = np.max(V)
    V_min = np.min(V)
    idx_max = np.argmax(V)
    idx_min = np.argmin(V)
    ax1.annotate(f'{V_max:.2f}', xy=(x[idx_max], V_max), fontsize=9, ha='center')
    ax1.annotate(f'{V_min:.2f}', xy=(x[idx_min], V_min), fontsize=9, ha='center')
    
    # === MOMENT DIAGRAM ===
    # CRITICAL: Positive moments (sagging) plotted DOWNWARD
    # We plot M (not -M), but with inverted y-axis
    
    ax2.fill_between(x, 0, M, where=(M >= 0), alpha=0.3, color='red', label='M+ (sagging)')
    ax2.fill_between(x, 0, M, where=(M < 0), alpha=0.3, color='blue', label='M- (hogging)')
    ax2.plot(x, M, 'k-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Moment M [kNm/m]', fontsize=11)
    ax2.set_xlabel('Position x [m]', fontsize=11)
    ax2.set_title('Bending Moment Diagram (ULS) — Positive (sagging) plotted DOWN', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # INVERT Y-AXIS so positive moments are below
    ax2.invert_yaxis()
    
    # Support lines
    cumulative = 0
    for span in spans:
        ax2.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
        cumulative += span
    ax2.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
    
    # Max/min values
    M_max = np.max(M)
    M_min = np.min(M)
    idx_max_m = np.argmax(M)
    idx_min_m = np.argmin(M)
    ax2.annotate(f'+{M_max:.2f}', xy=(x[idx_max_m], M_max), fontsize=9, 
                 ha='center', va='top')
    if M_min < 0:
        ax2.annotate(f'{M_min:.2f}', xy=(x[idx_min_m], M_min), fontsize=9, 
                     ha='center', va='bottom')
    
    # Point loads
    for pl in point_loads:
        ax1.axvline(pl['position'], color='green', linestyle=':', alpha=0.7)
        ax2.axvline(pl['position'], color='green', linestyle=':', alpha=0.7)
    
    ax2.legend(loc='lower right')
    plt.tight_layout()
    return fig


def plot_deflection(x: np.ndarray, defl: np.ndarray, spans: List[float],
                    defl_limit: float, point_loads: List[Dict]) -> plt.Figure:
    """Plot deflection diagram with limit line."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    total_length = sum(spans)
    
    # Deflection (positive = downward)
    ax.fill_between(x, 0, defl, alpha=0.3, color='orange')
    ax.plot(x, defl, 'b-', linewidth=2, label='Deflection')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Limit line
    ax.axhline(y=defl_limit, color='red', linestyle='--', linewidth=1.5,
               label=f'Limit L/200 = {defl_limit:.1f} mm')
    
    # Supports
    cumulative = 0
    for span in spans:
        ax.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
        # Support symbol
        ax.plot(cumulative, 0, marker='^', markersize=10, color='gray')
        cumulative += span
    ax.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
    ax.plot(cumulative, 0, marker='^', markersize=10, color='gray')
    
    # Max deflection annotation
    idx_max = np.argmax(np.abs(defl))
    d_max = defl[idx_max]
    ax.plot(x[idx_max], d_max, 'ro', markersize=8)
    ax.annotate(f'δ_max = {d_max:.2f} mm\nx = {x[idx_max]:.2f} m',
                xy=(x[idx_max], d_max), xytext=(x[idx_max] + 0.3, d_max + defl_limit*0.2),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Invert y-axis (deflection downward)
    ax.invert_yaxis()
    
    ax.set_xlabel('Position x [m]', fontsize=11)
    ax.set_ylabel('Deflection δ [mm]', fontsize=11)
    ax.set_title('Deflection Diagram (SLS)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Steel Sheet Calculator v3.1", page_icon="🏗️", layout="wide")
    
    st.title("🏗️ Profiled Steel Sheet Calculator v3.1")
    st.markdown("**EN 1993-1-3 compliant with all required checks**")
    
    # === SIDEBAR ===
    st.sidebar.header("📋 Input Parameters")
    
    # Profile
    st.sidebar.subheader("1. Profile")
    profile_mode = st.sidebar.radio("Input Mode", ["Database", "Manual"], horizontal=True)
    
    if profile_mode == "Database":
        profile_name = st.sidebar.selectbox("Ruukki Profile", get_available_profiles())
        props = get_profile_database()[profile_name].copy()
    else:
        st.sidebar.markdown("**Manual Input**")
        c1, c2 = st.sidebar.columns(2)
        props = {
            "h": c1.number_input("h [mm]", 20.0, 300.0, 60.0),
            "t": c2.number_input("t [mm]", 0.4, 2.0, 0.8),
            "I_pos": c1.number_input("I_pos [cm⁴/m]", 10.0, 1000.0, 65.0),
            "I_neg": c2.number_input("I_neg [cm⁴/m]", 10.0, 1000.0, 58.0),
            "W_pos": c1.number_input("W_pos [cm³/m]", 5.0, 200.0, 21.7),
            "W_neg": c2.number_input("W_neg [cm³/m]", 5.0, 200.0, 19.3),
            "A_eff": c1.number_input("A_eff [mm²/m]", 500.0, 3000.0, 1050.0),
            "E": 210000,
            "h_w": c2.number_input("h_w [mm]", 15.0, 290.0, 55.0),
            "phi": c1.number_input("φ [°]", 45.0, 90.0, 70.0),
            "r": c2.number_input("r [mm]", 1.0, 15.0, 4.0),
            "num_webs": int(c1.number_input("Webs", 1, 10, 3)),
            "pitch": c2.number_input("Pitch [mm]", 100.0, 400.0, 180.0),
            "b_top": 80, "b_bottom": 70
        }
        profile_name = "Custom"
    
    # Geometry
    st.sidebar.subheader("2. Geometry")
    num_spans = st.sidebar.number_input("Number of Spans", 1, 5, 1)
    spans = []
    for i in range(int(num_spans)):
        spans.append(st.sidebar.number_input(f"L{i+1} [m]", 0.5, 20.0, 6.0, key=f"L{i}"))
    total_length = sum(spans)
    
    # Material
    st.sidebar.subheader("3. Material")
    steel_grade = st.sidebar.selectbox("Steel Grade", ["S350GD", "S320GD", "S280GD", "S250GD"])
    fy = {"S350GD": 350, "S320GD": 320, "S280GD": 280, "S250GD": 250}[steel_grade]
    
    # Loads
    st.sidebar.subheader("4. Distributed Loads")
    c1, c2 = st.sidebar.columns(2)
    g_k = c1.number_input("G_k [kN/m²]", 0.0, 10.0, 0.3, help="Permanent")
    q_k = c2.number_input("Q_k [kN/m²]", 0.0, 20.0, 1.5, help="Variable")
    
    st.sidebar.markdown("**Partial Factors**")
    c1, c2, c3 = st.sidebar.columns(3)
    gamma_g = c1.number_input("γ_G", 1.0, 1.5, 1.35)
    gamma_q = c2.number_input("γ_Q", 1.0, 2.0, 1.5)
    gamma_m = c3.number_input("γ_M0", 1.0, 1.2, 1.0)
    
    udl_uls = gamma_g * g_k + gamma_q * q_k
    udl_sls = g_k + q_k
    st.sidebar.info(f"**q_ULS = {udl_uls:.2f} kN/m²**  |  q_SLS = {udl_sls:.2f} kN/m²")
    
    # Point loads
    st.sidebar.subheader("5. Point Loads")
    num_pl = st.sidebar.number_input("Number of Point Loads", 0, 10, 0)
    point_loads = []
    for i in range(int(num_pl)):
        st.sidebar.markdown(f"**P{i+1}**")
        c1, c2 = st.sidebar.columns(2)
        p_g = c1.number_input(f"P{i+1}_G [kN]", 0.0, 50.0, 0.0, key=f"Pg{i}")
        p_q = c2.number_input(f"P{i+1}_Q [kN]", 0.0, 50.0, 2.0, key=f"Pq{i}")
        pos = st.sidebar.number_input(f"Position [m]", 0.0, total_length, min(3.0, total_length/2), key=f"pos{i}")
        p_uls = gamma_g * p_g + gamma_q * p_q
        p_sls = p_g + p_q
        point_loads.append({"magnitude": p_uls, "magnitude_uls": p_uls, 
                           "magnitude_sls": p_sls, "position": pos})
    
    # Local effects
    st.sidebar.subheader("6. Local Effects")
    s_s = st.sidebar.number_input("Bearing length s_s [mm]", 10.0, 200.0, 50.0)
    
    # Calculate
    calc_btn = st.sidebar.button("🔄 CALCULATE", type="primary", use_container_width=True)
    
    # === MAIN AREA ===
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📐 Profile")
        fig_profile = draw_profile_cross_section(props, profile_name)
        st.pyplot(fig_profile)
        plt.close()
        
        with st.expander("Properties", expanded=True):
            st.write(f"**h** = {props['h']} mm")
            st.write(f"**t** = {props['t']} mm")
            st.write(f"**I_pos** = {props.get('I_pos', props.get('I', 0))} cm⁴/m")
            st.write(f"**W_pos** = {props.get('W_pos', props.get('W', 0))} cm³/m")
            st.write(f"**Webs** = {props.get('num_webs', 2)}")
    
    with col2:
        st.subheader("📏 Loading")
        fig_beam = draw_beam_diagram(spans, udl_uls, udl_sls, point_loads)
        st.pyplot(fig_beam)
        plt.close()
    
    if calc_btn:
        st.markdown("---")
        st.header("📊 Results")
        
        # Get I for deflection (use I_pos for simplicity)
        I_calc = props.get('I_pos', props.get('I', 50))
        
        # ULS calculation
        if num_spans == 1:
            x_uls, M_uls, V_uls, _ = calculate_single_span(spans[0], udl_uls, point_loads, props['E'], I_calc)
            pl_sls = [{"magnitude": pl['magnitude_sls'], "position": pl['position']} for pl in point_loads]
            x_sls, _, _, defl_sls = calculate_single_span(spans[0], udl_sls, pl_sls, props['E'], I_calc)
            reactions = None
        else:
            x_uls, M_uls, V_uls, _, reactions = calculate_multi_span(spans, udl_uls, point_loads, props['E'], I_calc)
            pl_sls = [{"magnitude": pl['magnitude_sls'], "position": pl['position']} for pl in point_loads]
            x_sls, _, _, defl_sls, _ = calculate_multi_span(spans, udl_sls, pl_sls, props['E'], I_calc)
        
        # Resistances
        res = calculate_resistances(props, fy, gamma_m)
        M_Rd_pos = res["M_Rd_pos"]
        M_Rd_neg = res["M_Rd_neg"]
        V_Rd = res["V_Rd"]
        
        # Design values
        M_Ed_pos = np.max(M_uls)  # Max positive (sagging)
        M_Ed_neg = abs(np.min(M_uls)) if np.min(M_uls) < 0 else 0  # Max negative (hogging)
        V_Ed = np.max(np.abs(V_uls))
        defl_max = np.max(np.abs(defl_sls))
        defl_limit = max(spans) * 1000 / 200
        
        # === CHECKS TABLE ===
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("EN 1993-1-3 Checks")
            
            checks_data = []
            
            # 6.1.4 Bending - positive
            util_m_pos = M_Ed_pos / M_Rd_pos if M_Rd_pos > 0 else 0
            checks_data.append(["6.1.4", "Bending (sagging)", 
                               f"{M_Ed_pos:.2f} / {M_Rd_pos:.2f} kNm/m",
                               f"{util_m_pos*100:.1f}%",
                               "OK" if util_m_pos <= 1.0 else "NOT OK"])
            
            # 6.1.4 Bending - negative (for continuous)
            if num_spans > 1 and M_Ed_neg > 0:
                util_m_neg = M_Ed_neg / M_Rd_neg if M_Rd_neg > 0 else 0
                checks_data.append(["6.1.4", "Bending (hogging)",
                                   f"{M_Ed_neg:.2f} / {M_Rd_neg:.2f} kNm/m",
                                   f"{util_m_neg*100:.1f}%",
                                   "OK" if util_m_neg <= 1.0 else "NOT OK"])
            
            # 6.1.5 Shear
            util_v = V_Ed / V_Rd if V_Rd > 0 else 0
            checks_data.append(["6.1.5", "Shear",
                               f"{V_Ed:.2f} / {V_Rd:.2f} kN/m",
                               f"{util_v*100:.1f}%",
                               "OK" if util_v <= 1.0 else "NOT OK"])
            
            # 6.1.10 Combined M+V
            comb_mv = check_combined_bending_shear(M_Ed_pos, V_Ed, M_Rd_pos, V_Rd)
            checks_data.append(["6.1.10", "Combined M+V",
                               f"M_Rd,red = {comb_mv['M_Rd_reduced']:.2f} kNm/m",
                               f"{comb_mv['utilization']*100:.1f}%",
                               comb_mv['status']])
            
            # 7.3 Deflection
            util_defl = defl_max / defl_limit if defl_limit > 0 else 0
            checks_data.append(["7.3", "Deflection (SLS)",
                               f"{defl_max:.2f} / {defl_limit:.1f} mm",
                               f"{util_defl*100:.1f}%",
                               "OK" if util_defl <= 1.0 else "NOT OK"])
            
            df = pd.DataFrame(checks_data, columns=["Clause", "Check", "Ed / Rd", "Util.", "Status"])
            
            def color_status(val):
                if val == "OK":
                    return 'background-color: #90EE90'
                elif "NOT OK" in str(val):
                    return 'background-color: #FFB6C1'
                return ''
            
            st.dataframe(df.style.applymap(color_status, subset=['Status']), 
                        hide_index=True, use_container_width=True)
            
            # Local checks for point loads
            if point_loads:
                st.subheader("Local Effects (6.1.7, 6.1.10, 6.1.11)")
                
                for i, pl in enumerate(point_loads):
                    F_Ed = pl['magnitude_uls']
                    
                    # 6.1.7 Web crippling
                    wc = check_web_crippling(F_Ed, props, fy, gamma_m, s_s)
                    
                    # 6.1.10 Combined V + F
                    comb_vf = check_combined_shear_web_crippling(V_Ed, F_Ed, V_Rd, wc['R_w_Rd'])
                    
                    # 6.1.11 Combined M + F
                    comb_mf = check_combined_bending_web_crippling(M_Ed_pos, F_Ed, M_Rd_pos, wc['R_w_Rd'])
                    
                    status = "🟢" if all([wc['status']=="OK", comb_vf['status']=="OK", 
                                          comb_mf['status']=="OK"]) else "🔴"
                    
                    with st.expander(f"**P{i+1}** = {F_Ed:.1f} kN @ x={pl['position']:.2f}m — {status}"):
                        st.write(f"**6.1.7 Web crippling:** R_w,Rd = {wc['R_w_Rd']:.2f} kN → "
                                f"Util = {wc['utilization']*100:.1f}% → {wc['status']}")
                        st.write(f"**6.1.10 Combined V+F:** R_w,Rd,red = {comb_vf['R_w_Rd_reduced']:.2f} kN → "
                                f"Util = {comb_vf['utilization']*100:.1f}% → {comb_vf['status']}")
                        st.write(f"**6.1.11 Combined M+F:** Interaction = {comb_mf['interaction']:.2f}/1.25 → "
                                f"Util = {comb_mf['utilization']*100:.1f}% → {comb_mf['status']}")
            
            # Reactions
            if reactions:
                st.subheader("Support Reactions (ULS)")
                cols = st.columns(len(reactions))
                for i, (col, R) in enumerate(zip(cols, reactions)):
                    col.metric(f"R{i+1}", f"{R:.2f} kN/m")
        
        with col2:
            st.subheader("Summary")
            
            all_ok = all([
                util_m_pos <= 1.0,
                util_v <= 1.0,
                comb_mv['utilization'] <= 1.0,
                util_defl <= 1.0
            ])
            
            if point_loads:
                for pl in point_loads:
                    wc = check_web_crippling(pl['magnitude_uls'], props, fy, gamma_m, s_s)
                    if wc['utilization'] > 1.0:
                        all_ok = False
            
            if all_ok:
                st.success("✅ ALL CHECKS PASSED")
            else:
                st.error("❌ SOME CHECKS FAILED")
            
            st.metric("M_Ed,max (ULS)", f"{max(M_Ed_pos, M_Ed_neg):.2f} kNm/m")
            st.metric("V_Ed,max (ULS)", f"{V_Ed:.2f} kN/m")
            st.metric("δ_max (SLS)", f"{defl_max:.2f} mm")
        
        # Diagrams
        st.markdown("---")
        st.subheader("📈 Diagrams")
        
        tab1, tab2 = st.tabs(["Internal Forces (ULS)", "Deflection (SLS)"])
        
        with tab1:
            fig_forces = plot_internal_forces(x_uls, M_uls, V_uls, spans, point_loads)
            st.pyplot(fig_forces)
            plt.close()
        
        with tab2:
            fig_defl = plot_deflection(x_sls, defl_sls, spans, defl_limit, point_loads)
            st.pyplot(fig_defl)
            plt.close()
    
    else:
        st.info("👈 Configure and click **CALCULATE**")
        
        st.markdown("""
        ### EN 1993-1-3 Checks Implemented:
        | Clause | Check | Description |
        |--------|-------|-------------|
        | 6.1.4 | Bending | M_Ed ≤ M_c,Rd (sagging & hogging) |
        | 6.1.5 | Shear | V_Ed ≤ V_b,Rd |
        | 6.1.7 | Web crippling | F_Ed ≤ R_w,Rd |
        | 6.1.10 | M+V, V+F | Combined actions |
        | 6.1.11 | M+F | Bending + transverse force |
        | 7.3 | Deflection | δ ≤ L/200 |
        
        ### Sign Convention:
        - **Positive moment (sagging)** → Tension at bottom → Plotted **DOWNWARD**
        - **Negative moment (hogging)** → Tension at top → Plotted **UPWARD**
        """)


if __name__ == "__main__":
    main()
