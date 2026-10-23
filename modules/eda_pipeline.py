import os
import json
import pandas as pd
import streamlit as st

def render_eda_dashboard():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
            <h1 style="color:white; margin:0; font-size:2.2rem;">📊 Exploratory Data Analysis & ETL Pipeline</h1>
            <p style="color:#94A3B8; margin-top:0.4rem; font-size:1rem;">Automated Schema Validation, ETL Execution Metrics & Agronomic Data Analytics</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Pipeline Metrics Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Validated Crop Classes", value="36 Classes", delta="Pydantic Enforced")
    with col2:
        st.metric(label="ETL Data Source", value="OpenWeather API", delta="Live JSON Stream")
    with col3:
        st.metric(label="Database Target", value="SQLite / JSON DB", delta="Active Sync")

    st.markdown("---")

    # 2. Agronomic Database Overview & Analytics
    st.subheader("📋 1. Agronomic Dataset Summary")
    
    agri_db_path = os.path.join("database", "crop_agronomic_db.json")
    if os.path.exists(agri_db_path):
        with open(agri_db_path, "r") as f:
            raw_data = json.load(f)
            
        df_raw = pd.DataFrame(raw_data)
        st.dataframe(df_raw, use_container_width=True)
        
        st.subheader("📈 2. Interactive Exploratory Visualizations")
        
        # Transform data structure into clean rows for visualization
        rows = []
        for crop_name, metrics in raw_data.items():
            if isinstance(metrics, dict):
                rows.append({
                    "Crop": crop_name,
                    "Optimal Soil pH": float(metrics.get("optimal_ph", 6.5)),
                    "Min Soil pH": float(metrics.get("ph_min", 6.0)),
                    "Max Soil pH": float(metrics.get("ph_max", 7.0))
                })
        
        plot_df = pd.DataFrame(rows)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Optimal Soil pH Comparison Across Crops**")
            st.bar_chart(plot_df.set_index("Crop")[["Optimal Soil pH"]])
            
        with c2:
            st.write("**Soil pH Ranges (Min vs Max)**")
            st.line_chart(plot_df.set_index("Crop")[["Min Soil pH", "Max Soil pH"]])
            
    else:
        st.warning("`database/crop_agronomic_db.json` not found.")