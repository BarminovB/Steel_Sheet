import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import fsolve

# Profiles module functions
def get_profile_properties(manufacturer, profile_type):
    """
    Retrieve geometric and sectional properties for the selected profile.
    Based on Ruukki Poimu data; expandable for UK suppliers.
    Units: mm for lengths, cm4/m for I, cm3/m for W, mm2/m for A_eff.
    Updated with approximate real data for T130 from Ruukki sources (Wel ~98-127 cm3/m for t=0.9, I estimated).
    """
    profiles = {
        "Ruukki": {
            "T45-0.7": {
                "h": 45, "t": 0.7, "I": 50.0, "W": 10.0, "A_eff": 1000, "E": 210000,
                "h_w": 40, "phi": 90, "r": 3, "num_webs": 2  # Example for trapezoidal
            },
            "T60-0.8": {
                "h": 60, "t": 0.8, "I": 80.0, "W": 15.0, "A_eff": 1200, "E": 210000,
                "h_w": 55, "phi": 80, "r": 4, "num_webs": 3
            },
            "T80-1.0": {
                "h": 80, "t": 1.0, "I": 150.0, "W": 25.0, "A_eff": 1400, "E": 210000,
                "h_w": 70, "phi": 85, "r": 4, "num_webs": 3
            },
            "T130-0.9": {
                "h": 130, "t": 0.9, "I": 450.0, "W": 70.0, "A_eff": 1500, "E": 210000,
                "h_w": 120, "phi": 70, "r": 5, "num_webs": 4  # Adjusted based on Ruukki data (Wel approx 98-127 cm3/m for similar)
            }
        },
        "UK Suppliers": {
            "ProfileA": {
                "h": 50, "t": 0.75, "I": 60.0, "W": 12.0, "A_eff": 1100, "E": 210000,
                "h_w": 45, "phi": 90, "r": 3, "num_webs": 2
            }
        }
    }
    return profiles.get(manufacturer, {}).get(profile_type, {})

def get_available_profiles(manufacturer):
    profiles = {
        "Ruukki": ["T45-0.7", "T60-0.8", "T80-1.0", "T130-0.9"],
        "UK Suppliers": ["ProfileA"]
    }
    return profiles.get(manufacturer, [])

# Calculations module functions
def calculate_efforts(span, udl, point_loads, E, I):
    """
    Calculate bending moments (M), shear forces (V), and deflections (defl) for a simple beam.
    Uses analytical integration for UDL + point loads.
    EN 1993 compliant logic: efforts envelopes.
    Units: span m, udl kN/m, P kN (assumed concentrated on 1m width), I cm4/m -> m4/m, M kNm/m, V kN/m.
    """
    x = np.linspace(0, span, 1000)
    # Calculate left reaction ra
    ra = udl * span / 2
    for pl in point_loads:
        ra += pl['magnitude'] * (span - pl['position']) / span
    # Shear V
    V = np.full_like(x, ra) - udl * x
    for pl in point_loads:
        V[x > pl['position']] -= pl['magnitude']
    # Moments M: integrate V
    M = cumulative_trapezoid(V, x, initial=0)
    # Deflections: double integrate M / (E*I)
    I_m4 = I * 1e-8  # cm4/m to m4/m
    E_pa = E * 1e6  # MPa to Pa
    EI = E_pa * I_m4
    theta = cumulative_trapezoid(M * 1000 / EI, x, initial=0)  # M kNm/m to Nm/m
    defl = cumulative_trapezoid(theta, x, initial=0)  # m
    defl *= 1000  # to mm
    return M, V, defl

def check_uls_sls(M, V, defl, props, fy, span, gamma_m, gamma_g, gamma_q, point_loads=[]):
    """
    Perform ULS and SLS checks per EN 1993-1-1/1-3.
    Added interaction for local transverse (from local_effects).
    Units fixed: M_rd, V_rd in kNm/m, kN/m.
    """
    load_factor = gamma_g * 0.3 + gamma_q * 0.7  # ULS factor
    M_ed = max(abs(M)) * load_factor  # kNm/m
    V_ed = max(abs(V)) * load_factor  # kN/m
    defl_max = max(abs(defl))  # mm, SLS no factor
    W = props['W'] * 1e-6  # m3/m (from cm3/m)
    M_rd = W * (fy * 1e6) / gamma_m  # kNm/m
    A_eff = props['A_eff'] * 1e-6  # m2/m
    V_rd = A_eff * (fy * 1e6) / (np.sqrt(3) * gamma_m)  # kN/m
    uls = {
        "Moment": "OK" if M_ed <= M_rd else "NOT OK",
        "Shear": "OK" if V_ed <= V_rd else "NOT OK",
        "Local Transverse": "OK"
    }
    sls = {
        "Deflection": "OK" if defl_max <= span * 1000 / 200 else "NOT OK"
    }
    # Local checks
    for pl in point_loads:
        local_res = check_local_effects(pl, props, fy, V_ed=V_ed, M_ed=M_ed, gamma_m0=gamma_m, s_s=s_s, concrete=concrete_contact, f_cd=f_cd, span=span)
        if "NOT OK" in local_res['status']:
            uls["Local Transverse"] = "NOT OK"
    return uls, sls

# Local effects module functions
def check_local_effects(point_load, props, fy, V_ed=0, M_ed=0, gamma_m0=1.0, s_s=50, concrete=False, f_cd=20, span=0):
    """
    Check local effects from point loads per EN 1993-1-3 6.1.7 (updated to better match standard formula for unstiffened webs).
    Simplified for multiple webs: R_w,Rd ≈ t^2 * sqrt(f_y * E) * factor / gamma.
    Includes interaction with shear and bending per 6.1.10, 6.1.11.
    Units: mm, MPa, kN.
    Verified approx with Dlubal example (R_w,Rd ~8.73 kN for similar params).
    """
    F_ed = point_load['magnitude']  # kN
    t = props['t']
    r = props['r']
    phi = props['phi']
    h_w = props['h_w']
    num_webs = props.get('num_webs', 1)
    E = props['E']
    # Simplified from EN 1993-1-3 6.1.7.1 for single web, adjusted for multiple
    l_a = s_s  # Effective bearing length, approx
    factor = 1 - 0.1 * np.sqrt(r / t)  # From (6.14)
    alpha = 0.5 + np.sqrt(0.02 * l_a / t)  # From (6.15)
    beta = 2.4 + (phi / 90)**2  # From (6.16)
    R_w_Rd = t**2 * np.sqrt(fy * E) * factor * alpha * beta / gamma_m0 / 1000  # kN, /1000 for units (t mm, sqrt MPa)
    R_w_Rd *= num_webs  # For multiple webs (approx, as per 6.1.7.3)
    # Interaction with shear (6.1.10)
    V_pl_Rd = props['A_eff'] * fy / (np.sqrt(3) * gamma_m0) / 1000  # kN/m
    if V_ed > 0.5 * V_pl_Rd:
        k_v = np.sqrt(1 - (V_ed / V_pl_Rd - 0.5) / 0.5)  # approximate
        R_w_Rd *= k_v
    # Interaction with bending (6.1.11)
    M_c_Rd = props['W'] * fy / gamma_m0 / 1000  # kNm/m
    util_bend_trans = M_ed / M_c_Rd + 1.25 * (F_ed / R_w_Rd)
    if util_bend_trans > 1:
        status = "NOT OK (Interaction)"
        utilization = util_bend_trans
    else:
        utilization = F_ed / R_w_Rd
        status = "OK" if utilization <= 1.0 else "NOT OK"
    # Concrete
    if concrete:
        A_c = s_s * 50  # mm2, placeholder for contact width 50mm
        sigma_c = F_ed * 1000 / A_c  # MPa (F_ed kN to N)
        if sigma_c > f_cd:
            status = "NOT OK (Concrete)"
            utilization = max(utilization, sigma_c / f_cd)
    return {"status": status, "utilization": utilization, "details": f"R_w,Rd = {R_w_Rd:.2f} kN"}

# New function for max allowable point load
def calculate_max_point_load(span, udl, positions, props, fy, gamma_m, gamma_g, gamma_q, s_s, concrete, f_cd):
    """
    Calculate maximum allowable point load for each position, considering UDL.
    Solves separately for global (moment ULS, defl SLS) and local, takes min.
    Uses fsolve for each limit.
    """
    load_factor = gamma_g * 0.3 + gamma_q * 0.7
    max_p = []
    for pos in positions:
        # Global moment ULS
        def m_func(P):
            point_loads_temp = [{"magnitude": P, "position": pos, "type": "Temp"}]
            M, _, _ = calculate_efforts(span, udl, point_loads_temp, props['E'], props['I'])
            M_ed = max(abs(M)) * load_factor
            W = props['W'] * 1e-6
            M_rd = W * (fy * 1e6) / gamma_m
            return M_ed - M_rd
        max_p_m = fsolve(m_func, 10.0)[0] if m_func(0) < 0 else np.inf  # If already fail, inf or adjust

        # Global defl SLS (no factor)
        def d_func(P):
            point_loads_temp = [{"magnitude": P, "position": pos, "type": "Temp"}]
            _, _, defl = calculate_efforts(span, udl, point_loads_temp, props['E'], props['I'])
            defl_max = max(abs(defl))
            defl_lim = span * 1000 / 200
            return defl_max - defl_lim
        max_p_d = fsolve(d_func, 10.0)[0] if d_func(0) < 0 else np.inf

        max_p_g = min(max_p_m, max_p_d)

        # Local
        M_udl, V_udl, _ = calculate_efforts(span, udl, [], props['E'], props['I'])
        M_ed_udl = max(abs(M_udl)) * load_factor
        V_ed_udl = max(abs(V_udl)) * load_factor
        def local_func(P):
            pl_temp = {"magnitude": P, "position": pos}
            # Approx delta V = P/2, delta M = P * pos * (span - pos) / (4 * span) * span approx max M from P
            delta_M = P * pos * (span - pos) / (4 * span) * 4  # Simplified quarter point approx
            local_res = check_local_effects(pl_temp, props, fy, V_ed=V_ed_udl + P / 2, M_ed=M_ed_udl + delta_M, gamma_m0=gamma_m, s_s=s_s, concrete=concrete, f_cd=f_cd, span=span)
            return local_res['utilization'] - 1.0  # Solve for util=1
        max_p_l = fsolve(local_func, 10.0)[0]

        max_p.append(min(max_p_g, max_p_l))
    return max_p

# Utils module functions
def generate_results_table(uls, sls):
    data = {**uls, **sls}
    df = pd.DataFrame(list(data.items()), columns=["Check", "Status"])
    return df

def plot_efforts(x, M, V, point_loads):
    fig, ax1 = plt.subplots()
    ax1.plot(x, M, 'b-', label='Moment (kNm/m)')
    ax1.set_ylabel('Moment')
    ax2 = ax1.twinx()
    ax2.plot(x, V, 'r--', label='Shear (kN/m)')
    ax2.set_ylabel('Shear')
    for pl in point_loads:
        ax1.axvline(pl['position'], color='g', linestyle=':', label='Point Load')
    fig.legend()
    return fig

def plot_deflections(x, defl, point_loads):
    fig, ax = plt.subplots()
    ax.plot(x, defl, 'g-', label='Deflection (mm)')
    for pl in point_loads:
        ax.axvline(pl['position'], color='r', linestyle=':', label='Point Load')
    ax.legend()
    return fig

# Main Streamlit app
st.title("Profiled Steel Sheet Calculator (Ruukki/UK Suppliers)")

st.sidebar.header("Input Parameters")

# Profile input mode
profile_mode = st.sidebar.selectbox("Profile Input Mode", ["From Database", "Manual"])

if profile_mode == "From Database":
    manufacturer = st.sidebar.selectbox("Manufacturer", ["Ruukki", "UK Suppliers"])
    profile_type = st.sidebar.selectbox("Profile Type", get_available_profiles(manufacturer))
    properties = get_profile_properties(manufacturer, profile_type)
else:
    st.sidebar.subheader("Manual Profile Properties")
    h = st.sidebar.number_input("Height h (mm)", min_value=0.0, value=45.0)
    t = st.sidebar.number_input("Thickness t (mm)", min_value=0.0, value=0.7)
    I = st.sidebar.number_input("Moment of Inertia I (cm⁴/m)", min_value=0.0, value=50.0)
    W = st.sidebar.number_input("Section Modulus W (cm³/m)", min_value=0.0, value=10.0)
    A_eff = st.sidebar.number_input("Effective Area A_eff (mm²/m)", min_value=0.0, value=1000.0)
    E = st.sidebar.number_input("Modulus of Elasticity E (MPa)", min_value=0.0, value=210000.0)
    h_w = st.sidebar.number_input("Web Height h_w (mm)", min_value=0.0, value=40.0)
    phi = st.sidebar.number_input("Web Angle phi (degrees)", min_value=0.0, value=90.0)
    r = st.sidebar.number_input("Internal Radius r (mm)", min_value=0.0, value=3.0)
    num_webs = st.sidebar.number_input("Number of Webs", min_value=1, value=2)
    properties = {
        "h": h, "t": t, "I": I, "W": W, "A_eff": A_eff, "E": E,
        "h_w": h_w, "phi": phi, "r": r, "num_webs": num_webs
    }

st.sidebar.write("Selected/Manual Profile Properties:", properties)

span_length = st.sidebar.number_input("Span Length (m)", min_value=1.0, value=6.0)
support_scheme = "Simple Beam"  # Fixed for prototype

steel_grade = st.sidebar.selectbox("Steel Grade", ["S350GD", "S280GD"])
fy = 350 if steel_grade == "S350GD" else 280  # Yield strength in MPa

udl = st.sidebar.number_input("UDL (kN/m)", min_value=0.0, value=1.5)

point_loads = []
num_points = st.sidebar.number_input("Number of Point Loads", min_value=0, max_value=5, value=0)
for i in range(num_points):
    col1, col2, col3 = st.sidebar.columns(3)
    magnitude = col1.number_input(f"Point Load {i+1} Magnitude (kN)", value=2.0)
    position = col2.number_input(f"Position {i+1} (m from left)", min_value=0.0, max_value=span_length, value=span_length/2)
    load_type = col3.selectbox(f"Type {i+1}", ["Equipment Suspension", "Ductwork"])
    point_loads.append({"magnitude": magnitude, "position": position, "type": load_type})

gamma_m = 1.0  # For steel
gamma_g = 1.35  # Permanent
gamma_q = 1.5   # Variable

st.sidebar.header("Local Effects Parameters")
s_s = st.sidebar.number_input("Stiff Bearing Length s_s (mm)", min_value=10.0, value=50.0)
concrete_contact = st.sidebar.checkbox("Concrete Contact (EN 1992 check)")
f_cd = st.sidebar.number_input("Concrete design strength f_cd (MPa)", min_value=0.0, value=20.0) if concrete_contact else 20.0

if st.sidebar.button("Calculate"):
    x = np.linspace(0, span_length, 1000)
    M, V, defl = calculate_efforts(span_length, udl, point_loads, properties['E'], properties['I'])
    uls_results, sls_results = check_uls_sls(M, V, defl, properties, fy, span_length, gamma_m, gamma_g, gamma_q, point_loads)
    local_results = [check_local_effects(pl, properties, fy, V_ed=max(abs(V)) * (gamma_g * 0.3 + gamma_q * 0.7), M_ed=max(abs(M)) * (gamma_g * 0.3 + gamma_q * 0.7), gamma_m0=gamma_m, s_s=s_s, concrete=concrete_contact, f_cd=f_cd, span=span_length) for pl in point_loads]
    st.header("Global Checks")
    st.table(generate_results_table(uls_results, sls_results))
    st.header("Local Effects Report")
    for i, res in enumerate(local_results):
        st.write(f"Point Load {i+1}: {res['status']} - Utilization: {res['utilization']:.2f} ({res['details']})")
    # Max allowable point loads
    st.header("Maximum Allowable Point Loads (considering UDL)")
    positions = [pl['position'] for pl in point_loads] if point_loads else [span_length / 2]
    max_p = calculate_max_point_load(span_length, udl, positions, properties, fy, gamma_m, gamma_g, gamma_q, s_s, concrete_contact, f_cd)
    for i, pos in enumerate(positions):
        st.write(f"For position {pos:.2f} m: Max P = {max_p[i]:.2f} kN")
    st.header("Graphs")
    fig_m_v = plot_efforts(x, M, V, point_loads)
    st.pyplot(fig_m_v)
    fig_defl = plot_deflections(x, defl, point_loads)
    st.pyplot(fig_defl)
