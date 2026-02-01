"""
Profiled Steel Sheet Calculator v3.0
Based on EN 1993-1-1 and EN 1993-1-3

Features:
- Profile cross-section visualization
- Longitudinal beam diagram with loads
- Multi-span continuous beam analysis
- Manual profile input
- Separate ULS/SLS load inputs
- Deflection diagram visualization

Author: Structural Engineering Calculator
Version: 3.0
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Arc, Rectangle, FancyArrowPatch
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.linalg import solve
from typing import List, Dict, Tuple, Optional
import io

# ============================================================================
# PROFILE DATABASE
# ============================================================================

def get_profile_database() -> Dict:
    """
    Profile database based on Ruukki Poimu data.
    Units:
        - h, t, h_w, r: mm
        - I: cm4/m (moment of inertia per meter width)
        - W: cm3/m (section modulus per meter width)  
        - A_eff: mm2/m (effective area per meter width)
        - E: MPa
        - pitch: mm (rib spacing)
        - b_top, b_bottom: mm (flange widths)
    """
    return {
        "T45-30-905-0.6": {
            "h": 45, "t": 0.6, "I": 30.4, "W": 13.5, "A_eff": 857, "E": 210000,
            "h_w": 40, "phi": 75, "r": 3, "num_webs": 3, "pitch": 150,
            "b_top": 70, "b_bottom": 60
        },
        "T45-30-905-0.7": {
            "h": 45, "t": 0.7, "I": 35.0, "W": 15.6, "A_eff": 1000, "E": 210000,
            "h_w": 40, "phi": 75, "r": 3, "num_webs": 3, "pitch": 150,
            "b_top": 70, "b_bottom": 60
        },
        "T45-30-905-0.8": {
            "h": 45, "t": 0.8, "I": 39.4, "W": 17.5, "A_eff": 1143, "E": 210000,
            "h_w": 40, "phi": 75, "r": 3, "num_webs": 3, "pitch": 150,
            "b_top": 70, "b_bottom": 60
        },
        "T60-40-905-0.7": {
            "h": 60, "t": 0.7, "I": 65.0, "W": 21.7, "A_eff": 1050, "E": 210000,
            "h_w": 55, "phi": 70, "r": 4, "num_webs": 3, "pitch": 180,
            "b_top": 80, "b_bottom": 70
        },
        "T60-40-905-0.9": {
            "h": 60, "t": 0.9, "I": 82.0, "W": 27.3, "A_eff": 1350, "E": 210000,
            "h_w": 55, "phi": 70, "r": 4, "num_webs": 3, "pitch": 180,
            "b_top": 80, "b_bottom": 70
        },
        "T85-40-840-0.8": {
            "h": 85, "t": 0.8, "I": 130.0, "W": 30.6, "A_eff": 1200, "E": 210000,
            "h_w": 78, "phi": 72, "r": 4, "num_webs": 3, "pitch": 210,
            "b_top": 95, "b_bottom": 80
        },
        "T85-40-840-1.0": {
            "h": 85, "t": 1.0, "I": 160.0, "W": 37.6, "A_eff": 1500, "E": 210000,
            "h_w": 78, "phi": 72, "r": 4, "num_webs": 3, "pitch": 210,
            "b_top": 95, "b_bottom": 80
        },
        "T130M-75-930-0.8": {
            "h": 130, "t": 0.8, "I": 350.0, "W": 53.8, "A_eff": 1350, "E": 210000,
            "h_w": 120, "phi": 68, "r": 5, "num_webs": 4, "pitch": 233,
            "b_top": 110, "b_bottom": 90
        },
        "T130M-75-930-1.0": {
            "h": 130, "t": 1.0, "I": 430.0, "W": 66.2, "A_eff": 1688, "E": 210000,
            "h_w": 120, "phi": 68, "r": 5, "num_webs": 4, "pitch": 233,
            "b_top": 110, "b_bottom": 90
        },
        "T153M-100-840-1.0": {
            "h": 153, "t": 1.0, "I": 620.0, "W": 81.0, "A_eff": 1750, "E": 210000,
            "h_w": 143, "phi": 65, "r": 5, "num_webs": 3, "pitch": 280,
            "b_top": 130, "b_bottom": 100
        },
        "T153M-100-840-1.2": {
            "h": 153, "t": 1.2, "I": 730.0, "W": 95.4, "A_eff": 2100, "E": 210000,
            "h_w": 143, "phi": 65, "r": 5, "num_webs": 3, "pitch": 280,
            "b_top": 130, "b_bottom": 100
        }
    }


def get_available_profiles() -> List[str]:
    """Return list of available profiles."""
    return list(get_profile_database().keys())


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_profile_cross_section(props: Dict, profile_name: str = "") -> plt.Figure:
    """
    Draw trapezoidal profile cross-section.
    Shows one complete pitch of the profile.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    h = props['h']
    t = props['t']
    pitch = props.get('pitch', 150)
    phi = props.get('phi', 75)
    r = props.get('r', 3)
    b_top = props.get('b_top', pitch * 0.4)
    b_bottom = props.get('b_bottom', pitch * 0.35)
    
    # Calculate web angle offset
    phi_rad = np.radians(phi)
    web_offset = h / np.tan(phi_rad)
    
    # Draw profile outline (one pitch)
    # Starting from bottom left
    x_start = 0
    
    # Create profile path
    profile_x = []
    profile_y = []
    
    # Bottom flange (left side)
    profile_x.extend([x_start, x_start + b_bottom/2])
    profile_y.extend([0, 0])
    
    # Left web going up
    profile_x.append(x_start + b_bottom/2 + web_offset)
    profile_y.append(h)
    
    # Top flange
    profile_x.append(x_start + b_bottom/2 + web_offset + b_top)
    profile_y.append(h)
    
    # Right web going down
    profile_x.append(x_start + b_bottom/2 + 2*web_offset + b_top)
    profile_y.append(0)
    
    # Bottom flange (right side) to complete one pitch
    profile_x.append(x_start + pitch)
    profile_y.append(0)
    
    # Second rib
    profile_x.append(x_start + pitch + b_bottom/2)
    profile_y.append(0)
    
    profile_x.append(x_start + pitch + b_bottom/2 + web_offset)
    profile_y.append(h)
    
    profile_x.append(x_start + pitch + b_bottom/2 + web_offset + b_top)
    profile_y.append(h)
    
    profile_x.append(x_start + pitch + b_bottom/2 + 2*web_offset + b_top)
    profile_y.append(0)
    
    profile_x.append(x_start + 2*pitch)
    profile_y.append(0)
    
    # Draw the profile
    ax.plot(profile_x, profile_y, 'b-', linewidth=2.5)
    
    # Fill the steel area
    ax.fill_between(profile_x, [y - t for y in profile_y], profile_y, 
                    alpha=0.3, color='steelblue')
    
    # Dimension annotations
    # Height dimension
    ax.annotate('', xy=(-15, h), xytext=(-15, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(-25, h/2, f'h={h}', fontsize=9, ha='right', va='center', color='red')
    
    # Pitch dimension  
    mid_pitch = pitch / 2
    ax.annotate('', xy=(0, -15), xytext=(pitch, -15),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(mid_pitch, -25, f'pitch={pitch}', fontsize=9, ha='center', va='top', color='green')
    
    # Thickness annotation
    ax.text(pitch * 1.5, h + 10, f't={t} mm', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Settings
    ax.set_xlim(-50, 2*pitch + 30)
    ax.set_ylim(-40, h + 30)
    ax.set_aspect('equal')
    ax.set_xlabel('Width [mm]')
    ax.set_ylabel('Height [mm]')
    ax.set_title(f'Profile Cross-Section: {profile_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='brown', linewidth=3, label='Support')
    
    plt.tight_layout()
    return fig


def draw_beam_diagram(spans: List[float], udl_uls: float, udl_sls: float,
                      point_loads: List[Dict], support_type: str = "simple") -> plt.Figure:
    """
    Draw longitudinal beam diagram with applied loads.
    """
    total_length = sum(spans)
    num_spans = len(spans)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Draw beam
    beam_y = 1.0
    ax.plot([0, total_length], [beam_y, beam_y], 'b-', linewidth=4)
    
    # Draw supports
    support_positions = [0]
    cumulative = 0
    for span in spans:
        cumulative += span
        support_positions.append(cumulative)
    
    for i, x_sup in enumerate(support_positions):
        # Triangle support
        triangle = plt.Polygon([[x_sup - 0.15, beam_y - 0.05],
                                [x_sup + 0.15, beam_y - 0.05],
                                [x_sup, beam_y - 0.35]], 
                               closed=True, fill=True, 
                               facecolor='gray', edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)
        
        # Roller or pin
        if i == 0 or (support_type == "continuous" and i == len(support_positions) - 1):
            # Pin support (first and last for continuous)
            ax.plot(x_sup, beam_y - 0.35, 'ko', markersize=8)
        else:
            # Roller support
            circle = plt.Circle((x_sup, beam_y - 0.42), 0.07, 
                               fill=False, color='black', linewidth=1.5)
            ax.add_patch(circle)
        
        # Support label
        ax.text(x_sup, beam_y - 0.6, f'S{i+1}', ha='center', fontsize=9)
    
    # Draw UDL
    if udl_uls > 0:
        arrow_spacing = total_length / 20
        for x_arr in np.arange(0.1, total_length - 0.05, arrow_spacing):
            ax.annotate('', xy=(x_arr, beam_y + 0.05), 
                       xytext=(x_arr, beam_y + 0.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1))
        
        # UDL line
        ax.plot([0, total_length], [beam_y + 0.5, beam_y + 0.5], 'r-', linewidth=1.5)
        
        # UDL label
        ax.text(total_length / 2, beam_y + 0.7, 
                f'q_ULS = {udl_uls:.2f} kN/m²\nq_SLS = {udl_sls:.2f} kN/m²',
                ha='center', va='bottom', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Draw point loads
    for i, pl in enumerate(point_loads):
        x_pl = pl['position']
        p_uls = pl.get('magnitude_uls', pl.get('magnitude', 0))
        p_sls = pl.get('magnitude_sls', p_uls)
        
        # Arrow
        ax.annotate('', xy=(x_pl, beam_y + 0.05),
                   xytext=(x_pl, beam_y + 0.8),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        # Label
        ax.text(x_pl, beam_y + 0.9, 
                f'P{i+1}\nULS:{p_uls:.1f}kN\nSLS:{p_sls:.1f}kN',
                ha='center', va='bottom', fontsize=9, color='blue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Span labels
    cumulative = 0
    for i, span in enumerate(spans):
        mid_span = cumulative + span / 2
        ax.annotate('', xy=(cumulative + 0.1, beam_y - 0.75), 
                   xytext=(cumulative + span - 0.1, beam_y - 0.75),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax.text(mid_span, beam_y - 0.85, f'L{i+1} = {span:.2f} m', 
                ha='center', va='top', fontsize=10, color='green')
        cumulative += span
    
    # Settings
    ax.set_xlim(-0.5, total_length + 0.5)
    ax.set_ylim(-1.2, 2.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Beam Loading Diagram', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def draw_deflection_diagram(x: np.ndarray, defl: np.ndarray, 
                           spans: List[float], defl_limit: float,
                           point_loads: List[Dict]) -> plt.Figure:
    """
    Draw deflection diagram with beam visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    total_length = sum(spans)
    
    # Scale deflection for visualization
    max_defl = max(abs(defl)) if max(abs(defl)) > 0 else 1
    scale = 0.3 * total_length / max_defl  # Visual scale
    defl_scaled = defl * scale / 1000  # Convert mm to visual units
    
    # Draw undeformed beam (dashed)
    ax.plot([0, total_length], [0, 0], 'k--', linewidth=1, alpha=0.5, label='Undeformed')
    
    # Draw deformed beam
    ax.fill_between(x, 0, -defl_scaled, alpha=0.3, color='orange')
    ax.plot(x, -defl_scaled, 'b-', linewidth=2, label='Deflected shape')
    
    # Draw supports
    support_positions = [0]
    cumulative = 0
    for span in spans:
        cumulative += span
        support_positions.append(cumulative)
    
    for x_sup in support_positions:
        triangle = plt.Polygon([[x_sup - 0.1, -0.02],
                                [x_sup + 0.1, -0.02],
                                [x_sup, -0.15]], 
                               closed=True, fill=True, 
                               facecolor='gray', edgecolor='black')
        ax.add_patch(triangle)
    
    # Mark maximum deflection
    idx_max = np.argmax(np.abs(defl))
    x_max = x[idx_max]
    d_max = defl[idx_max]
    
    ax.plot(x_max, -defl_scaled[idx_max], 'ro', markersize=10)
    ax.annotate(f'δ_max = {d_max:.2f} mm\n@ x = {x_max:.2f} m',
                xy=(x_max, -defl_scaled[idx_max]),
                xytext=(x_max + 0.3, -defl_scaled[idx_max] - 0.1),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw deflection limit line
    defl_limit_scaled = defl_limit * scale / 1000
    ax.axhline(y=-defl_limit_scaled, color='red', linestyle='--', linewidth=1.5,
               label=f'Limit L/200 = {defl_limit:.1f} mm')
    
    # Point load positions
    for pl in point_loads:
        ax.axvline(x=pl['position'], color='green', linestyle=':', alpha=0.5)
    
    # Settings
    ax.set_xlim(-0.3, total_length + 0.3)
    y_range = max(abs(min(-defl_scaled)), abs(max(-defl_scaled)), defl_limit_scaled) * 1.5
    ax.set_ylim(-y_range - 0.2, 0.2)
    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Deflection (scaled)')
    ax.set_title('Deflection Diagram', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STRUCTURAL CALCULATIONS - SINGLE SPAN
# ============================================================================

def calculate_single_span(span: float, udl: float, point_loads: List[Dict],
                          E: float, I: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate internal forces and deflections for simply supported beam.
    """
    n_points = 1001
    x = np.linspace(0, span, n_points)
    
    # Left reaction
    Ra = udl * span / 2
    for pl in point_loads:
        if 0 <= pl['position'] <= span:
            Ra += pl['magnitude'] * (span - pl['position']) / span
    
    # Shear force
    V = np.full_like(x, Ra, dtype=float) - udl * x
    for pl in point_loads:
        if 0 <= pl['position'] <= span:
            V = np.where(x > pl['position'], V - pl['magnitude'], V)
    
    # Bending moment
    M = cumulative_trapezoid(V, x, initial=0)
    
    # Deflection with boundary conditions
    I_m4 = I * 1e-8
    E_Pa = E * 1e6
    EI = E_Pa * I_m4
    
    M_Nm = M * 1000
    theta = cumulative_trapezoid(M_Nm / EI, x, initial=0)
    y_raw = cumulative_trapezoid(theta, x, initial=0)
    y_corrected = y_raw - (y_raw[-1] / span) * x
    defl = y_corrected * 1000  # mm
    
    return x, M, V, defl


# ============================================================================
# STRUCTURAL CALCULATIONS - MULTI-SPAN (CONTINUOUS BEAM)
# ============================================================================

def calculate_multi_span(spans: List[float], udl: float, point_loads: List[Dict],
                         E: float, I: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Calculate internal forces for continuous beam using Three-Moment Equation.
    
    Returns:
        x: Position array [m]
        M: Bending moment array [kNm/m]
        V: Shear force array [kN/m]
        defl: Deflection array [mm]
        reactions: Support reactions [kN/m]
    """
    num_spans = len(spans)
    total_length = sum(spans)
    n_points = 1001
    
    if num_spans == 1:
        x, M, V, defl = calculate_single_span(spans[0], udl, point_loads, E, I)
        Ra = udl * spans[0] / 2
        for pl in point_loads:
            Ra += pl['magnitude'] * (spans[0] - pl['position']) / spans[0]
        Rb = udl * spans[0] - Ra + sum(pl['magnitude'] for pl in point_loads)
        return x, M, V, defl, [Ra, Rb]
    
    # Three-Moment Equation for continuous beams
    # M_i-1 * L_i + 2*M_i*(L_i + L_i+1) + M_i+1 * L_i+1 = -6 * (A_i*a_i/L_i + A_i+1*b_i+1/L_i+1)
    
    # For UDL: A*a/L = q*L^3/24 and A*b/L = q*L^3/24
    # So RHS = -6 * (q*L_i^3/24 + q*L_i+1^3/24) = -q/4 * (L_i^3 + L_i+1^3)
    
    # Build coefficient matrix for internal support moments
    # M_0 = M_n = 0 (simply supported ends)
    n_unknowns = num_spans - 1  # Internal support moments
    
    if n_unknowns == 0:
        # Single span
        x, M, V, defl = calculate_single_span(spans[0], udl, point_loads, E, I)
        return x, M, V, defl, []
    
    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)
    
    for i in range(n_unknowns):
        L_i = spans[i]
        L_ip1 = spans[i + 1]
        
        # Diagonal
        A[i, i] = 2 * (L_i + L_ip1)
        
        # Off-diagonals
        if i > 0:
            A[i, i-1] = L_i
        if i < n_unknowns - 1:
            A[i, i+1] = L_ip1
        
        # RHS for UDL
        b[i] = -udl / 4 * (L_i**3 + L_ip1**3)
    
    # Add point load contributions to RHS
    cumulative = 0
    for span_idx, span_len in enumerate(spans):
        for pl in point_loads:
            pos = pl['position']
            P = pl['magnitude']
            
            # Check if point load is in this span
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative  # Distance from left support
                b_dist = span_len - a  # Distance from right support
                L = span_len
                
                # Contribution to moments at supports i and i+1
                # For left internal support (i = span_idx - 1)
                if span_idx > 0:
                    # M_A/L term
                    contrib = -6 * P * a * b_dist * (L + b_dist) / (L**2 * 2)
                    b[span_idx - 1] += contrib
                
                # For right internal support (i = span_idx)
                if span_idx < n_unknowns:
                    # M_B/L term  
                    contrib = -6 * P * a * b_dist * (L + a) / (L**2 * 2)
                    b[span_idx] += contrib
        
        cumulative += span_len
    
    # Solve for internal moments
    try:
        M_supports = solve(A, b)
    except:
        M_supports = np.zeros(n_unknowns)
    
    # Full support moments array (including zeros at ends)
    M_sup_full = np.concatenate([[0], M_supports, [0]])
    
    # Calculate M, V, defl for entire beam
    x = np.linspace(0, total_length, n_points)
    M = np.zeros(n_points)
    V = np.zeros(n_points)
    
    # Process each span
    cumulative = 0
    reactions = []
    
    for span_idx, span_len in enumerate(spans):
        M_left = M_sup_full[span_idx]
        M_right = M_sup_full[span_idx + 1]
        
        # Span indices
        mask = (x >= cumulative) & (x <= cumulative + span_len)
        x_local = x[mask] - cumulative
        
        # Simple beam moment from UDL
        M_udl = udl * x_local * (span_len - x_local) / 2
        
        # Add point load moments in this span
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
        
        # Superimpose support moments (linear variation)
        M_support_effect = M_left + (M_right - M_left) * x_local / span_len
        
        M[mask] = M_udl + M_pl + M_support_effect
        
        # Shear from equilibrium
        # V = dM/dx (numerical differentiation)
        
        # Reaction at left support of this span
        R_left = (udl * span_len / 2 - (M_right - M_left) / span_len)
        for pl in point_loads:
            pos = pl['position']
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative
                R_left += pl['magnitude'] * (span_len - a) / span_len
        
        if span_idx == 0:
            reactions.append(R_left)
        
        # Shear calculation
        V_local = R_left - udl * x_local
        for pl in point_loads:
            pos = pl['position']
            if cumulative <= pos < cumulative + span_len:
                a = pos - cumulative
                V_local = np.where(x_local > a, V_local - pl['magnitude'], V_local)
        
        V[mask] = V_local
        
        # Interior reaction (sum of shears at support)
        if span_idx < num_spans - 1:
            V_right = V_local[-1]
            reactions.append(-V_right)  # Will add next span's left reaction
        
        cumulative += span_len
    
    # Add last reaction
    reactions.append(-V[-1] if len(V) > 0 else 0)
    
    # Deflection calculation (numerical integration with BC at each support)
    I_m4 = I * 1e-8
    E_Pa = E * 1e6
    EI = E_Pa * I_m4
    
    defl = np.zeros(n_points)
    cumulative = 0
    
    for span_idx, span_len in enumerate(spans):
        mask = (x >= cumulative) & (x <= cumulative + span_len)
        indices = np.where(mask)[0]
        
        if len(indices) > 1:
            x_span = x[indices]
            M_span = M[indices]
            
            M_Nm = M_span * 1000
            dx = x_span[1] - x_span[0] if len(x_span) > 1 else 0.001
            
            theta = cumulative_trapezoid(M_Nm / EI, x_span, initial=0)
            y_raw = cumulative_trapezoid(theta, x_span, initial=0)
            
            # Apply BC: y=0 at both ends of span
            y_corrected = y_raw - (y_raw[-1] / span_len) * (x_span - cumulative)
            
            defl[indices] = y_corrected * 1000  # mm
        
        cumulative += span_len
    
    return x, M, V, defl, reactions


# ============================================================================
# CAPACITY CALCULATIONS
# ============================================================================

def calculate_capacities(props: Dict, fy: float, gamma_m: float) -> Dict:
    """Calculate design resistances per EN 1993-1-3."""
    W_m3 = props['W'] * 1e-6
    M_Rd = W_m3 * (fy * 1e6) / gamma_m / 1000  # kNm/m
    
    A_eff_m2 = props['A_eff'] * 1e-6
    V_Rd = A_eff_m2 * (fy * 1e6) / (np.sqrt(3) * gamma_m) / 1000  # kN/m
    
    return {"M_Rd": M_Rd, "V_Rd": V_Rd}


def check_uls_sls(M_uls: np.ndarray, V_uls: np.ndarray, defl_sls: np.ndarray,
                  props: Dict, fy: float, total_span: float,
                  gamma_m: float) -> Tuple[Dict, Dict, Dict]:
    """
    Perform ULS and SLS checks.
    M_uls, V_uls: Already factored for ULS
    defl_sls: Calculated with SLS loads
    """
    M_Ed = np.max(np.abs(M_uls))
    V_Ed = np.max(np.abs(V_uls))
    defl_max = np.max(np.abs(defl_sls))
    
    capacities = calculate_capacities(props, fy, gamma_m)
    M_Rd = capacities["M_Rd"]
    V_Rd = capacities["V_Rd"]
    
    # Deflection limit: L/200 for single span, L/250 for continuous
    defl_limit = total_span * 1000 / 200
    
    util_M = M_Ed / M_Rd if M_Rd > 0 else float('inf')
    util_V = V_Ed / V_Rd if V_Rd > 0 else float('inf')
    util_defl = defl_max / defl_limit if defl_limit > 0 else float('inf')
    
    uls_results = {
        "Moment": "OK" if util_M <= 1.0 else "NOT OK",
        "Shear": "OK" if util_V <= 1.0 else "NOT OK"
    }
    
    sls_results = {
        "Deflection": "OK" if util_defl <= 1.0 else "NOT OK"
    }
    
    values = {
        "M_Ed": M_Ed, "M_Rd": M_Rd, "util_M": util_M,
        "V_Ed": V_Ed, "V_Rd": V_Rd, "util_V": util_V,
        "defl_max": defl_max, "defl_limit": defl_limit, "util_defl": util_defl
    }
    
    return uls_results, sls_results, values


def check_local_effects(point_load: Dict, props: Dict, fy: float,
                        gamma_m0: float = 1.0, s_s: float = 50) -> Dict:
    """Check local web crippling per EN 1993-1-3 6.1.7."""
    F_Ed = point_load.get('magnitude_uls', point_load.get('magnitude', 0))
    t = props['t']
    r = props['r']
    phi = props['phi']
    E = props['E']
    num_webs = props.get('num_webs', 1)
    
    l_a = s_s
    k1 = 1 - 0.1 * np.sqrt(r / t)
    k2 = min(0.5 + np.sqrt(0.02 * l_a / t), 1.25)
    k3 = 2.4 + (phi / 90)**2
    
    R_w_Rd_single = k1 * k2 * k3 * t**2 * np.sqrt(fy * E) / gamma_m0 / 1000
    R_w_Rd = R_w_Rd_single * num_webs
    
    utilization = F_Ed / R_w_Rd if R_w_Rd > 0 else float('inf')
    status = "OK" if utilization <= 1.0 else "NOT OK"
    
    return {
        "status": status,
        "utilization": utilization,
        "R_w_Rd": R_w_Rd,
        "F_Ed": F_Ed
    }


# ============================================================================
# RESULTS TABLE
# ============================================================================

def generate_results_table(uls: Dict, sls: Dict, values: Dict) -> pd.DataFrame:
    """Generate formatted results table."""
    data = [
        ["Bending Moment (ULS)", 
         f"{values['M_Ed']:.2f} / {values['M_Rd']:.2f} kNm/m", 
         f"{values['util_M']*100:.1f}%", 
         uls["Moment"]],
        ["Shear Force (ULS)", 
         f"{values['V_Ed']:.2f} / {values['V_Rd']:.2f} kN/m", 
         f"{values['util_V']*100:.1f}%", 
         uls["Shear"]],
        ["Deflection (SLS)", 
         f"{values['defl_max']:.2f} / {values['defl_limit']:.2f} mm", 
         f"{values['util_defl']*100:.1f}%", 
         sls["Deflection"]]
    ]
    return pd.DataFrame(data, columns=["Check", "Ed / Rd", "Utilization", "Status"])


def plot_efforts(x: np.ndarray, M: np.ndarray, V: np.ndarray,
                 spans: List[float], point_loads: List[Dict]) -> plt.Figure:
    """Plot bending moment and shear force diagrams."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Shear
    ax1.fill_between(x, 0, V, alpha=0.3, color='red')
    ax1.plot(x, V, 'r-', linewidth=1.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Shear V [kN/m]')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Shear Force Diagram (ULS)')
    
    # Support lines
    cumulative = 0
    for span in spans:
        ax1.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
        cumulative += span
    ax1.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
    
    # Moment
    ax2.fill_between(x, 0, -M, alpha=0.3, color='blue')
    ax2.plot(x, -M, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Moment M [kNm/m]')
    ax2.set_xlabel('Position x [m]')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    ax2.set_title('Bending Moment Diagram (ULS)')
    
    # Support lines
    cumulative = 0
    for span in spans:
        ax2.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
        cumulative += span
    ax2.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
    
    # Point loads
    for pl in point_loads:
        ax1.axvline(pl['position'], color='green', linestyle=':', alpha=0.7)
        ax2.axvline(pl['position'], color='green', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Steel Sheet Calculator v3.0",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Profiled Steel Sheet Calculator v3.0")
    st.markdown("**Based on EN 1993-1-1 and EN 1993-1-3**")
    
    # ========== SIDEBAR ==========
    st.sidebar.header("📋 Input Parameters")
    
    # Profile selection
    st.sidebar.subheader("1. Profile Selection")
    profile_mode = st.sidebar.radio(
        "Input Mode",
        ["Database", "Manual"],
        horizontal=True
    )
    
    if profile_mode == "Database":
        profile_name = st.sidebar.selectbox(
            "Select Profile",
            get_available_profiles()
        )
        properties = get_profile_database()[profile_name]
    else:
        st.sidebar.markdown("**Manual Profile Properties**")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            h = st.number_input("Height h [mm]", min_value=20.0, value=60.0, step=5.0)
            t = st.number_input("Thickness t [mm]", min_value=0.4, max_value=2.0, value=0.8, step=0.1)
            I = st.number_input("Inertia I [cm⁴/m]", min_value=10.0, value=65.0, step=5.0)
            W = st.number_input("Modulus W [cm³/m]", min_value=5.0, value=21.7, step=1.0)
            A_eff = st.number_input("A_eff [mm²/m]", min_value=500.0, value=1050.0, step=50.0)
        with col2:
            E = st.number_input("E [MPa]", min_value=190000.0, max_value=220000.0, value=210000.0, step=1000.0)
            h_w = st.number_input("Web height h_w [mm]", min_value=15.0, value=55.0, step=5.0)
            phi = st.number_input("Web angle φ [°]", min_value=45.0, max_value=90.0, value=70.0, step=5.0)
            r = st.number_input("Radius r [mm]", min_value=1.0, max_value=10.0, value=4.0, step=1.0)
            num_webs = st.number_input("Number of webs", min_value=1, max_value=10, value=3)
        
        pitch = st.sidebar.number_input("Pitch [mm]", min_value=100.0, value=180.0, step=10.0)
        b_top = st.sidebar.number_input("Top flange width [mm]", min_value=30.0, value=80.0, step=5.0)
        b_bottom = st.sidebar.number_input("Bottom flange width [mm]", min_value=30.0, value=70.0, step=5.0)
        
        properties = {
            "h": h, "t": t, "I": I, "W": W, "A_eff": A_eff, "E": E,
            "h_w": h_w, "phi": phi, "r": r, "num_webs": int(num_webs),
            "pitch": pitch, "b_top": b_top, "b_bottom": b_bottom
        }
        profile_name = "Custom Profile"
    
    # Beam geometry
    st.sidebar.subheader("2. Beam Geometry")
    num_spans = st.sidebar.number_input("Number of Spans", min_value=1, max_value=5, value=1)
    
    spans = []
    for i in range(int(num_spans)):
        span = st.sidebar.number_input(
            f"Span L{i+1} [m]",
            min_value=0.5, max_value=20.0, value=6.0, step=0.5,
            key=f"span_{i}"
        )
        spans.append(span)
    
    total_length = sum(spans)
    
    # Steel grade
    st.sidebar.subheader("3. Material")
    steel_grade = st.sidebar.selectbox(
        "Steel Grade",
        ["S350GD", "S320GD", "S280GD", "S250GD"]
    )
    fy_dict = {"S350GD": 350, "S320GD": 320, "S280GD": 280, "S250GD": 250}
    fy = fy_dict[steel_grade]
    
    # Loads - ULS and SLS separate
    st.sidebar.subheader("4. Distributed Loads")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        g_k = st.number_input("G_k (dead) [kN/m²]", min_value=0.0, value=0.3, step=0.1,
                              help="Characteristic permanent load")
    with col2:
        q_k = st.number_input("Q_k (live) [kN/m²]", min_value=0.0, value=1.2, step=0.1,
                              help="Characteristic variable load")
    
    st.sidebar.markdown("**Partial Factors**")
    col1, col2, col3 = st.sidebar.columns(3)
    gamma_g = col1.number_input("γ_G", value=1.35, step=0.05)
    gamma_q = col2.number_input("γ_Q", value=1.5, step=0.05)
    gamma_m = col3.number_input("γ_M0", value=1.0, step=0.05)
    
    # Calculate combined loads
    udl_uls = gamma_g * g_k + gamma_q * q_k  # kN/m² for ULS
    udl_sls = g_k + q_k  # kN/m² for SLS (characteristic)
    
    st.sidebar.info(f"**ULS:** q_Ed = {udl_uls:.2f} kN/m²\n\n**SLS:** q_k = {udl_sls:.2f} kN/m²")
    
    # Point loads
    st.sidebar.subheader("5. Point Loads")
    num_points = st.sidebar.number_input("Number of Point Loads", min_value=0, max_value=10, value=0)
    
    point_loads = []
    for i in range(int(num_points)):
        st.sidebar.markdown(f"**Point Load {i+1}**")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            p_g = st.number_input(f"P{i+1}_G [kN]", min_value=0.0, value=0.0, step=0.5, key=f"pg_{i}",
                                  help="Permanent component")
            p_q = st.number_input(f"P{i+1}_Q [kN]", min_value=0.0, value=2.0, step=0.5, key=f"pq_{i}",
                                  help="Variable component")
        with col2:
            position = st.number_input(f"Position [m]", min_value=0.0, max_value=total_length,
                                       value=min(3.0, total_length/2), step=0.5, key=f"pos_{i}")
        
        p_uls = gamma_g * p_g + gamma_q * p_q
        p_sls = p_g + p_q
        
        point_loads.append({
            "magnitude": p_uls,  # For calculations
            "magnitude_uls": p_uls,
            "magnitude_sls": p_sls,
            "position": position,
            "P_G": p_g,
            "P_Q": p_q
        })
    
    # Local effects
    st.sidebar.subheader("6. Local Effects")
    s_s = st.sidebar.number_input("Bearing length s_s [mm]", min_value=10.0, value=50.0, step=10.0)
    
    # Calculate button
    calculate = st.sidebar.button("🔄 CALCULATE", type="primary", use_container_width=True)
    
    # ========== MAIN CONTENT ==========
    
    # Always show profile info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📐 Profile Cross-Section")
        fig_profile = draw_profile_cross_section(properties, profile_name)
        st.pyplot(fig_profile)
        plt.close()
        
        # Profile properties table
        with st.expander("Profile Properties", expanded=True):
            props_display = {
                "Height h": f"{properties['h']} mm",
                "Thickness t": f"{properties['t']} mm",
                "Inertia I": f"{properties['I']} cm⁴/m",
                "Modulus W": f"{properties['W']} cm³/m",
                "A_eff": f"{properties['A_eff']} mm²/m",
                "Webs": f"{properties.get('num_webs', 2)}",
                "Web angle φ": f"{properties.get('phi', 75)}°"
            }
            for key, val in props_display.items():
                st.write(f"**{key}:** {val}")
    
    with col2:
        st.subheader("📏 Beam Loading Diagram")
        fig_beam = draw_beam_diagram(spans, udl_uls, udl_sls, point_loads,
                                     "continuous" if num_spans > 1 else "simple")
        st.pyplot(fig_beam)
        plt.close()
    
    # Run calculations
    if calculate:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Calculate for ULS
        if num_spans == 1:
            x_uls, M_uls, V_uls, _ = calculate_single_span(
                spans[0], udl_uls, point_loads, properties['E'], properties['I']
            )
            # Calculate for SLS
            point_loads_sls = [{"magnitude": pl['magnitude_sls'], "position": pl['position']} 
                              for pl in point_loads]
            x_sls, M_sls, V_sls, defl_sls = calculate_single_span(
                spans[0], udl_sls, point_loads_sls, properties['E'], properties['I']
            )
            reactions = None
        else:
            x_uls, M_uls, V_uls, _, reactions = calculate_multi_span(
                spans, udl_uls, point_loads, properties['E'], properties['I']
            )
            point_loads_sls = [{"magnitude": pl['magnitude_sls'], "position": pl['position']} 
                              for pl in point_loads]
            x_sls, M_sls, V_sls, defl_sls, _ = calculate_multi_span(
                spans, udl_sls, point_loads_sls, properties['E'], properties['I']
            )
        
        # Check results
        uls_results, sls_results, values = check_uls_sls(
            M_uls, V_uls, defl_sls, properties, fy, max(spans), gamma_m
        )
        
        # Results layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Results table
            st.subheader("Global Checks")
            results_df = generate_results_table(uls_results, sls_results, values)
            
            def color_status(val):
                if val == "OK":
                    return 'background-color: #90EE90'
                elif "NOT OK" in str(val):
                    return 'background-color: #FFB6C1'
                return ''
            
            styled_df = results_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            
            # Local effects
            if point_loads:
                st.subheader("Local Effects (Web Crippling)")
                for i, pl in enumerate(point_loads):
                    local_res = check_local_effects(pl, properties, fy, gamma_m, s_s)
                    status_icon = "🟢" if local_res['status'] == "OK" else "🔴"
                    st.markdown(
                        f"**P{i+1}** ({pl['magnitude_uls']:.1f} kN @ {pl['position']:.2f} m): "
                        f"{status_icon} {local_res['status']} — "
                        f"Util: **{local_res['utilization']*100:.1f}%** — "
                        f"R_w,Rd = {local_res['R_w_Rd']:.2f} kN"
                    )
            
            # Reactions
            if reactions and len(reactions) > 0:
                st.subheader("Support Reactions (ULS)")
                cols = st.columns(len(reactions))
                for i, (col, R) in enumerate(zip(cols, reactions)):
                    col.metric(f"R{i+1}", f"{R:.2f} kN/m")
        
        with col2:
            st.subheader("Summary")
            
            all_ok = all([
                uls_results["Moment"] == "OK",
                uls_results["Shear"] == "OK",
                sls_results["Deflection"] == "OK"
            ])
            
            if point_loads:
                for pl in point_loads:
                    local_res = check_local_effects(pl, properties, fy, gamma_m, s_s)
                    if local_res['status'] != "OK":
                        all_ok = False
            
            if all_ok:
                st.success("✅ ALL CHECKS PASSED")
            else:
                st.error("❌ SOME CHECKS FAILED")
            
            st.metric("M_Ed (ULS)", f"{values['M_Ed']:.2f} kNm/m")
            st.metric("V_Ed (ULS)", f"{values['V_Ed']:.2f} kN/m")
            st.metric("δ_max (SLS)", f"{values['defl_max']:.2f} mm")
            st.metric("Limit L/200", f"{values['defl_limit']:.1f} mm")
        
        # Diagrams
        st.markdown("---")
        st.subheader("📈 Diagrams")
        
        tab1, tab2 = st.tabs(["📊 Internal Forces (ULS)", "📉 Deflection (SLS)"])
        
        with tab1:
            fig_forces = plot_efforts(x_uls, M_uls, V_uls, spans, point_loads)
            st.pyplot(fig_forces)
            plt.close()
        
        with tab2:
            defl_limit = max(spans) * 1000 / 200
            fig_defl = draw_deflection_diagram(x_sls, defl_sls, spans, defl_limit, point_loads)
            st.pyplot(fig_defl)
            plt.close()
    
    else:
        st.info("👈 Configure parameters and click **CALCULATE** to run analysis.")
        
        st.markdown("""
        ### Features
        - **Profile database**: Ruukki profiles with accurate properties
        - **Manual profile input**: Enter custom profile dimensions
        - **Multi-span analysis**: Up to 5 continuous spans
        - **Separate ULS/SLS loads**: Proper load combination per EN 1990
        - **Visual diagrams**: Profile, loading, forces, deflection
        - **Local effects**: Web crippling per EN 1993-1-3
        
        ### Standards
        - EN 1990: Basis of structural design
        - EN 1991-1-1: Actions on structures
        - EN 1993-1-1: General rules for steel
        - EN 1993-1-3: Cold-formed steel
        """)


if __name__ == "__main__":
    main()
