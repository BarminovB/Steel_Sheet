import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from profiles import get_profile_properties
from calculations import calculate_efforts, check_uls_sls
from local_effects import check_local_effects
from utils import plot_efforts, plot_deflections, generate_results_table

# Streamlit app title
st.title("Profiled Steel Sheet Calculator (Ruukki/UK Suppliers)")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Manufacturer selection
manufacturer = st.sidebar.selectbox("Manufacturer", ["Ruukki", "UK Suppliers"])

# Profile selection (expandable)
profile_type = st.sidebar.selectbox("Profile Type", ["T45-0.7", "T60-0.8"])  # Example profiles

# Get profile properties
properties = get_profile_properties(manufacturer, profile_type)
st.sidebar.write("Selected Profile Properties:", properties)

# Span and support scheme (simple beam for prototype)
span_length = st.sidebar.number_input("Span Length (m)", min_value=1.0, value=6.0)
support_scheme = "Simple Beam"  # Fixed for prototype

# Material
steel_grade = st.sidebar.selectbox("Steel Grade", ["S350GD", "S280GD"])
fy = 350 if steel_grade == "S350GD" else 280  # Yield strength in MPa

# Uniformly Distributed Load (UDL)
udl = st.sidebar.number_input("UDL (kN/m)", min_value=0.0, value=1.5)

# Point loads (multiple)
point_loads = []
num_points = st.sidebar.number_input("Number of Point Loads", min_value=0, max_value=5, value=0)
for i in range(num_points):
    col1, col2, col3 = st.sidebar.columns(3)
    magnitude = col1.number_input(f"Point Load {i+1} Magnitude (kN)", value=2.0)
    position = col2.number_input(f"Position {i+1} (m from left)", min_value=0.0, max_value=span_length, value=span_length/2)
    load_type = col3.selectbox(f"Type {i+1}", ["Equipment Suspension", "Ductwork"])
    point_loads.append({"magnitude": magnitude, "position": position, "type": load_type})

# Gamma factors (partial safety factors, per Eurocode)
gamma_m = 1.0  # For steel
gamma_g = 1.35  # Permanent
gamma_q = 1.5   # Variable

# Run calculation
if st.sidebar.button("Calculate"):
    # Combine loads
    x = np.linspace(0, span_length, 1000)  # Discretize span
    
    # Calculate efforts (M, V, defl)
    M, V, defl = calculate_efforts(span_length, udl, point_loads, properties['E'], properties['I'])
    
    # Checks ULS/SLS
    uls_results, sls_results = check_uls_sls(M, V, defl, properties, fy, span_length, gamma_m, gamma_g, gamma_q)
    
    # Local effects
    local_results = [check_local_effects(pl, properties, fy) for pl in point_loads]
    
    # Display results
    st.header("Global Checks")
    st.table(generate_results_table(uls_results, sls_results))
    
    st.header("Local Effects Report")
    for i, res in enumerate(local_results):
        st.write(f"Point Load {i+1}: {res['status']} - Utilization: {res['utilization']:.2f}")
    
    st.header("Graphs")
    fig_m_v = plot_efforts(x, M, V, point_loads)
    st.pyplot(fig_m_v)
    
    fig_defl = plot_deflections(x, defl, point_loads)
    st.pyplot(fig_defl)