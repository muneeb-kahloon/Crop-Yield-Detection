from modules.eda_pipeline import render_eda_dashboard
import os
import streamlit as st
from PIL import Image
import numpy as np

# Import existing backend modules
from modules.disease_detection import predict_crop_disease
from modules.soil_fertilizer import evaluate_soil_and_fertilizer
from modules.weather_advisory import fetch_live_weather, generate_weather_advisory
from modules.crop_suitability import rank_crop_suitability
from modules.smart_assistant import process_assistant_query

# 1. Page Configuration
st.set_page_config(
    page_title="AgriVision AI - Enterprise Smart Agriculture Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Custom CSS Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Page Entry Transition */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes growBar {
        from { width: 0; }
    }

    .main .block-container {
        animation: fadeInUp 0.38s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    .stApp {
        background:
            radial-gradient(circle at 100% 0%, rgba(34, 197, 94, 0.06) 0%, transparent 38%),
            radial-gradient(circle at 0% 100%, rgba(79, 70, 229, 0.05) 0%, transparent 40%),
            #F8FAFC;
    }

    /* Base Cards */
    .agri-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.6rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 10px -2px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.25rem;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease, border-color 0.25s ease;
    }

    .agri-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 26px -6px rgba(0, 0, 0, 0.08);
    }

    /* Pill Badges */
    .badge-mint {
        background-color: #DDF5E5; color: #123D2A; padding: 0.38rem 0.85rem;
        border-radius: 20px; font-weight: 700; font-size: 0.85rem; display: inline-block; border: 1px solid #A7F3D0;
    }

    /* ---------- New professional UI components ---------- */
    .hero-light {
        position: relative;
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 70%);
        border: 1px solid #BBF7D0;
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 28px -10px rgba(22, 163, 74, 0.18);
        overflow: hidden;
    }
    .hero-light::after {
        content: "";
        position: absolute; right: -60px; top: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, rgba(34,197,94,0.18) 0%, transparent 70%);
    }
    .hero-light h1 { margin: 0; font-size: 2.2rem; font-weight: 800; color: #15803D; }
    .hero-light p { margin: 0.45rem 0 0 0; font-size: 1.05rem; color: #3F6B4F; max-width: 760px; }

    .stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 1.8rem; }
    .stat-tile {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
        padding: 1rem 1.2rem; box-shadow: 0 2px 6px rgba(15,23,42,0.03);
    }
    .stat-tile .stat-value { font-size: 1.5rem; font-weight: 800; color: #0F172A; line-height: 1.2; }
    .stat-tile .stat-label { font-size: 0.82rem; font-weight: 600; color: #64748B; margin-top: 2px; }

    .section-title {
        font-size: 1.25rem; font-weight: 800; color: #0F172A; margin: 0.4rem 0 0.9rem 0;
        display: flex; align-items: center; gap: 10px;
    }
    .section-title::after { content: ""; flex: 1; height: 1px; background: #E2E8F0; }

    .module-card h3 { font-size: 1.1rem; font-weight: 800; }
    .module-card p { color: #475569; font-size: 0.95rem; margin-bottom: 0; }

    .flow-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 0.5rem; }
    .flow-step {
        background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 14px; padding: 1rem 1.2rem;
    }
    .flow-step b { color: #0F172A; }
    .flow-step span { display: block; color: #64748B; font-size: 0.88rem; margin-top: 4px; }

    .meter-track { background: #EEF2F7; border-radius: 999px; height: 10px; overflow: hidden; margin-top: 6px; }
    .meter-fill { height: 100%; border-radius: 999px; animation: growBar 0.9s cubic-bezier(0.16, 1, 0.3, 1); }

    .ph-scale {
        position: relative; height: 12px; border-radius: 999px; margin: 1.2rem 0 0.4rem 0;
        background: linear-gradient(90deg, #F87171 0%, #FBBF24 25%, #4ADE80 50%, #38BDF8 75%, #818CF8 100%);
    }
    .ph-marker {
        position: absolute; top: -6px; width: 6px; height: 24px; background: #0F172A;
        border-radius: 4px; border: 2px solid #FFFFFF; box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    .ph-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; font-weight: 600; }

    .side-brand {
        display: flex; align-items: center; gap: 12px; padding: 0.9rem 1rem;
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
        border: 1px solid #BBF7D0; border-radius: 14px; margin-bottom: 0.6rem;
    }
    .side-brand .brand-name { font-size: 1.15rem; font-weight: 800; color: #14532D; line-height: 1.1; }
    .side-brand .brand-sub { font-size: 0.75rem; color: #4D7C5F; font-weight: 600; }

    .status-dot {
        display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #22C55E;
        margin-right: 6px; box-shadow: 0 0 0 4px rgba(34,197,94,0.18);
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    div[data-testid="stSidebarUserContent"] {
        padding-top: 1rem;
    }

    /* Customizing Streamlit Radio into Big Interactive Buttons */
    div[data-testid="stSidebarUserContent"] .stRadio > div {
        gap: 10px !important;
    }

    div[data-testid="stSidebarUserContent"] .stRadio label {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 1rem !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        display: flex !important;
        align-items: center !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Radio Circle Hide for Clean Button Look */
    div[data-testid="stSidebarUserContent"] .stRadio label > div:first-child {
        display: none !important;
    }

    /* Hover State for Sidebar Buttons */
    div[data-testid="stSidebarUserContent"] .stRadio label:hover {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        transform: translateX(4px) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06) !important;
        color: #0F172A !important;
    }

    /* Selected Active Button State */
    div[data-testid="stSidebarUserContent"] .stRadio label[data-checked="true"] {
        background: #FFFFFF !important;
        border-left: 5px solid #16A34A !important;
        border-color: #CBD5E1 !important;
        color: #15803D !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 18px rgba(22, 163, 74, 0.14) !important;
    }

    /* ---------- Sidebar navigation buttons ---------- */
    .nav-heading { font-size: 0.8rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.04em; margin: 0.2rem 0 0.6rem 0.3rem; }
    section[data-testid="stSidebar"] .stButton { margin-bottom: -0.35rem; }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important; min-height: 48px !important;
        justify-content: flex-start !important; text-align: left !important;
        padding: 0.65rem 1rem !important; border-radius: 12px !important;
        font-size: 0.95rem !important; font-weight: 600 !important;
        background: #FFFFFF !important; color: #334155 !important;
        border: 1px solid #E5E9F0 !important; border-left: 4px solid #E5E9F0 !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04) !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button > div,
    section[data-testid="stSidebar"] .stButton > button p {
        justify-content: flex-start !important; text-align: left !important; width: 100%;
        color: inherit !important; margin: 0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #F0FDF4 !important; color: #15803D !important;
        border-color: #BBF7D0 !important; border-left-color: #4ADE80 !important;
        transform: translateX(3px) !important; box-shadow: 0 8px 16px -6px rgba(22,163,74,0.2) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #15803D 0%, #22C55E 100%) !important;
        color: #FFFFFF !important; border-color: transparent !important; border-left: 4px solid #BBF7D0 !important;
        box-shadow: 0 10px 20px -8px rgba(22,163,74,0.55) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
    section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
        color: #FFFFFF !important; transform: none !important;
    }

    /* ---------- Brand logo ---------- */
    .brand-wrap {
        display: flex; align-items: center; gap: 14px; padding: 1.1rem 1.1rem;
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
        border: 1px solid #BBF7D0; border-radius: 18px; margin-bottom: 0.8rem;
        box-shadow: 0 10px 22px -12px rgba(22,163,74,0.35);
    }
    .brand-wrap svg { flex-shrink: 0; filter: drop-shadow(0 8px 10px rgba(22,163,74,0.30)); }
    .brand-title { font-size: 1.55rem; font-weight: 800; color: #14532D; line-height: 1.05; letter-spacing: -0.01em; display: flex; align-items: center; gap: 7px; }
    .brand-pill {
        font-size: 0.72rem; font-weight: 800; color: #FFFFFF; padding: 0.18rem 0.5rem; border-radius: 8px;
        background: linear-gradient(135deg, #15803D 0%, #22C55E 100%);
    }
    .brand-tag { font-size: 0.78rem; font-weight: 600; color: #4D7C5F; margin-top: 5px; line-height: 1.3; }

    /* ---------- Step cards (How to get started) ---------- */
    .tool-grid .tool-card.step-card {
        background: var(--bg); color: var(--text); border: 2px dashed var(--accent);
        box-shadow: 0 10px 20px -14px var(--accent);
    }
    .tool-card.step-card::after { background: var(--accent); opacity: 0.12; }
    .tool-card.step-card .tool-icon { background: var(--accent); color: #FFFFFF; }

    /* ---------- Dashboard sync (sizes, radius, gaps) ---------- */
    .stat-strip { grid-template-columns: repeat(3, 1fr); gap: 18px; }
    .stat-tile { border-radius: 18px; padding: 1.2rem 1.5rem; box-shadow: 0 8px 18px -12px rgba(15,23,42,0.25); }
    .hero-light { border-radius: 18px; }
    .tool-grid .tool-card { min-height: 175px; }
    .tool-link { text-decoration: none !important; color: inherit !important; display: block; }
    .tool-link:hover { text-decoration: none !important; }
    .tool-cta {
        display: inline-block; margin-top: 0.9rem; font-size: 0.8rem; font-weight: 700;
        padding: 0.3rem 0.8rem; border-radius: 999px; background: rgba(255,255,255,0.25);
        position: relative; z-index: 1;
    }

    /* ---------- Professional navigation ---------- */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 8px !important; width: 100% !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        width: 100% !important; box-sizing: border-box !important;
        padding: 0.8rem 1rem !important; border-radius: 12px !important;
        background: #F8FAFC !important; border: 1px solid #E5E9F0 !important;
        border-left: 4px solid transparent !important;
        transition: all 0.2s ease !important; margin: 0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-of-type { display: none !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #F0FDF4 !important; border-left-color: #86EFAC !important;
        transform: translateX(3px) !important; box-shadow: 0 6px 14px rgba(22,163,74,0.10) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #15803D 0%, #22C55E 100%) !important;
        border-color: transparent !important; border-left-color: #BBF7D0 !important;
        box-shadow: 0 10px 20px -6px rgba(22,163,74,0.45) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * { color: #FFFFFF !important; font-weight: 700 !important; }

    /* ---------- Dashboard tool cards ---------- */
    .tool-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 1.8rem; }
    .tool-card {
        position: relative; overflow: hidden; border-radius: 18px; padding: 1.5rem; min-height: 190px;
        color: #FFFFFF; box-shadow: 0 14px 26px -12px rgba(15,23,42,0.35);
        transition: transform 0.25s cubic-bezier(0.16,1,0.3,1), box-shadow 0.25s ease;
    }
    .tool-card::after {
        content: ""; position: absolute; right: -40px; bottom: -40px; width: 140px; height: 140px;
        border-radius: 50%; background: rgba(255,255,255,0.14);
    }
    .tool-card:hover { transform: translateY(-5px); box-shadow: 0 22px 32px -12px rgba(15,23,42,0.4); }
    .tool-icon {
        width: 48px; height: 48px; border-radius: 14px; background: rgba(255,255,255,0.25);
        display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin-bottom: 0.9rem;
    }
    .tool-card h3 { margin: 0 0 0.4rem 0; font-size: 1.08rem; font-weight: 800; color: inherit; }
    .tool-card p { margin: 0; font-size: 0.9rem; line-height: 1.5; color: inherit; opacity: 0.93; position: relative; z-index: 1; }
    @media (max-width: 1100px) { .tool-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 700px) { .tool-grid { grid-template-columns: 1fr; } }

    /* Selectbox Dropdown Customization */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Form containers & uploader polish */
    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px;
        padding: 1.4rem 1.4rem 1rem 1.4rem;
        box-shadow: 0 4px 10px -2px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 14px;
        border: 2px dashed #86EFAC;
        background: #F7FEF9;
    }

    @media (max-width: 900px) {
        .stat-strip { grid-template-columns: repeat(2, 1fr); }
        .flow-row { grid-template-columns: 1fr; }
    }
    </style>
""", unsafe_allow_html=True)

# Helper: quick check that an uploaded image contains plant/leaf/tree-like colours
def looks_like_plant(img, min_ratio=0.10):
    try:
        small = img.convert("RGB").resize((128, 128)).convert("HSV")
        hsv = np.asarray(small, dtype=np.int32)
        h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # PIL hue scale 0-255: ~20-120 covers yellow-green-green tones (≈28°-170°)
        veg = (h >= 20) & (h <= 120) & (sat >= 50) & (val >= 40)
        return float(veg.mean()) >= min_ratio
    except Exception:
        return True

# Helper function for soil texture styling (light palette)
def get_soil_color_style(soil_type):
    color_map = {
        "Sandy": {"bg": "#FEF9E7", "border": "#F6D98B", "text": "#7A5A12", "desc": "Coarse Mineral Sand Matrix"},
        "Loam": {"bg": "#F6EDE3", "border": "#DDC3A8", "text": "#5C3A1E", "desc": "Rich Dark Earth Humus"},
        "Clay Loam": {"bg": "#FAEAE2", "border": "#E6BCA8", "text": "#7C3A24", "desc": "Terracotta Red-Brown Clay"},
        "Sandy Loam": {"bg": "#FCF1DD", "border": "#EBCB93", "text": "#7A4F12", "desc": "Golden Mineral Silt Mix"},
        "Heavy Clay": {"bg": "#F4E5D6", "border": "#D6B390", "text": "#6B3F1D", "desc": "Dense Clay Matrix"},
        "Silt Loam": {"bg": "#EAEFF4", "border": "#BFCBD8", "text": "#34465A", "desc": "Dark Slate Mineral Silt Soil"}
    }
    return color_map.get(soil_type, {"bg": "#F6EDE3", "border": "#DDC3A8", "text": "#5C3A1E", "desc": "Standard Soil Matrix"})

# 3. Sidebar Navigation Panel
with st.sidebar:
    st.markdown("""
        <div class="brand-wrap">
            <svg width="68" height="68" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-label="AgriVision AI logo">
                <defs>
                    <linearGradient id="avg" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%" stop-color="#22C55E"/>
                        <stop offset="100%" stop-color="#065F46"/>
                    </linearGradient>
                </defs>
                <rect width="64" height="64" rx="19" fill="url(#avg)"/>
                <circle cx="32" cy="32" r="22" fill="none" stroke="#FFFFFF" stroke-opacity="0.28" stroke-width="1.6" stroke-dasharray="3 4"/>
                <path d="M17 45 C16 28 28 17 47 17 C48 35 38 46 21 45 Z" fill="#FFFFFF"/>
                <path d="M20 43 L38 26" stroke="#16A34A" stroke-width="2.4" stroke-linecap="round"/>
                <path d="M29 36 L29 28 M34 31 L41 31" stroke="#16A34A" stroke-width="1.8" stroke-linecap="round"/>
                <circle cx="49" cy="46" r="3.2" fill="#BBF7D0"/>
                <circle cx="15" cy="20" r="2.4" fill="#BBF7D0"/>
            </svg>
            <div>
                <div class="brand-title">AgriVision <span class="brand-pill">AI</span></div>
                <div class="brand-tag">Enterprise Smart<br>Agriculture Platform</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav_items = [
        "📊  Dashboard Overview",
        "🤖  AI Farming Assistant",
        "🌿  Crop Disease Detection",
        "🧪  Soil pH & Health Analysis",
        "💊  Fertilizer Recommendations",
        "🌤️  Climate & Weather Advice",
        "🌾  Crop Suitability Guidance",
        "ℹ️  About Project"
        "📊 EDA & Data Pipeline"
    ]
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = nav_items[0]

    # Dashboard cards link to pages through ?page=<key>
    card_pages = {"assistant": 1, "disease": 2, "soil": 3, "fertilizer": 4, "weather": 5, "suitability": 6}
    requested = st.query_params.get("page")
    if requested in card_pages:
        st.session_state.nav_page = nav_items[card_pages[requested]]
        st.query_params.clear()

    st.markdown('<div class="nav-heading">Main Menu</div>', unsafe_allow_html=True)
    for i, item in enumerate(nav_items):
        is_active = st.session_state.nav_page == item
        if st.button(item, key=f"nav_btn_{i}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav_page = item
            st.rerun()
    navigation = st.session_state.nav_page
    st.markdown("---")
    st.markdown("""
        <div style="background-color: #F1F5F9; border: 1px solid #CBD5E1; padding: 0.85rem; border-radius: 12px; color: #334155; font-size: 0.85rem; line-height: 1.7;">
            <span class="status-dot"></span><b>System Status:</b> Operational<br>
            <b>Model Core:</b> MobileNetV2
        </div>
    """, unsafe_allow_html=True)

# Clean Navigation Key Match
nav_key = navigation.strip()

# -----------------------------------------------------------------------------
# 1. DASHBOARD OVERVIEW (GREEN TITLE)
# -----------------------------------------------------------------------------
if "Dashboard Overview" in nav_key:
    st.markdown("""
        <div class="hero-light">
            <h1>📊 Smart Agriculture Dashboard</h1>
            <p>Integrated AI platform for disease management, soil analysis, fertilizer planning & climate action</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stat-strip">
            <div class="stat-tile"><div class="stat-value">6</div><div class="stat-label">Integrated AI modules</div></div>
            <div class="stat-tile"><div class="stat-value">MobileNetV2</div><div class="stat-label">Vision model core</div></div>
            <div class="stat-tile"><div class="stat-value">~20,638</div><div class="stat-label">PlantVillage training images</div></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Enterprise Agricultural Tools</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="tool-grid">
            <a class="tool-link" href="?page=assistant" target="_self">
                <div class="tool-card" style="background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);">
                    <div class="tool-icon">🤖</div>
                    <h3>Natural Language AI Assistant</h3>
                    <p>Interactive conversational bot to diagnose agricultural issues and route queries in real-time.</p>
                    <span class="tool-cta">Open module</span>
                </div>
            </a>
            <a class="tool-link" href="?page=disease" target="_self">
                <div class="tool-card" style="background: linear-gradient(135deg, #15803D 0%, #22C55E 100%);">
                    <div class="tool-icon">🌿</div>
                    <h3>Crop Disease Detection</h3>
                    <p>Computer vision AI leaf scanner to identify visual pathogens and display targeted remedies.</p>
                    <span class="tool-cta">Open module</span>
                </div>
            </a>
            <a class="tool-link" href="?page=soil" target="_self">
                <div class="tool-card" style="background: linear-gradient(135deg, #FBE3D0 0%, #F2BE98 100%); color:#6B3417;">
                    <div class="tool-icon" style="background: rgba(255,255,255,0.55);">🧪</div>
                    <h3>Soil pH & Texture Analysis</h3>
                    <p>Chemical analysis tool to evaluate crop pH tolerances and visual soil texture profiles.</p>
                    <span class="tool-cta" style="background: rgba(255,255,255,0.6);">Open module</span>
                </div>
            </a>
            <a class="tool-link" href="?page=fertilizer" target="_self">
                <div class="tool-card" style="background: linear-gradient(135deg, #FBF6E8 0%, #EBDDB6 100%); color:#5E4A1F; border: 1px solid #E7D8B4;">
                    <div class="tool-icon" style="background: rgba(255,255,255,0.6);">💊</div>
                    <h3>Fertilizer Recommendations</h3>
                    <p>Nutrient deficiency calculator providing tailored NPK fertilizer schedules and safety tips.</p>
                    <span class="tool-cta" style="background: rgba(255,255,255,0.6);">Open module</span>
                </div>
            </a>
            <a class="tool-link" href="?page=weather" target="_self">
                <div class="tool-card" style="background: linear-gradient(135deg, #0369A1 0%, #38BDF8 100%);">
                    <div class="tool-icon">🌤️</div>
                    <h3>Climate & Weather Advice</h3>
                    <p>Live weather forecast integration and climate extreme mitigation strategies.</p>
                    <span class="tool-cta">Open module</span>
                </div>
            </a>
            <a class="tool-link" href="?page=suitability" target="_self">
                <div class="tool-card" style="background: linear-gradient(135deg, #F1FAE3 0%, #CDEBA6 100%); color:#3F6212; border: 1px solid #C5E5A0;">
                    <div class="tool-icon" style="background: rgba(255,255,255,0.65);">🌾</div>
                    <h3>Crop Suitability Guidance</h3>
                    <p>Multi-parameter agronomic decision matrix ranking candidate crops for land optimization.</p>
                    <span class="tool-cta" style="background: rgba(255,255,255,0.65);">Open module</span>
                </div>
            </a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">How to get started</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="tool-grid">
            <div class="tool-card step-card" style="--bg:#E9F8F6; --accent:#2BB3A5; --text:#115E59;">
                <div class="tool-icon" style="font-weight:800;">1</div>
                <h3>Diagnose</h3>
                <p>Upload a leaf photo or describe the problem to the assistant.</p>
            </div>
            <div class="tool-card step-card" style="--bg:#ECF9F0; --accent:#4CC474; --text:#14532D;">
                <div class="tool-icon" style="font-weight:800;">2</div>
                <h3>Analyze</h3>
                <p>Check soil pH, texture and nutrient levels for your crop.</p>
            </div>
            <div class="tool-card step-card" style="--bg:#EAF5FD; --accent:#4AA9E4; --text:#0C4A6E;">
                <div class="tool-icon" style="font-weight:800;">3</div>
                <h3>Plan</h3>
                <p>Get fertilizer, weather and crop suitability guidance.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AI FARMING ASSISTANT (TEAL / EMERALD THEME)
# -----------------------------------------------------------------------------
elif "AI Farming Assistant" in navigation or "AI Farming Assistant" in st.session_state.nav_page:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
            <h2 style="color:white; margin:0;">🤖 Interactive AI Farming Assistant</h2>
            <p style="color:#E6FFFA; margin-top:0.4rem;">Ask questions in plain language to get instant agronomic guidance and system routing.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am your AI Agronomist. Ask me any question about crops, fertilizers, weather, or leaf diseases!"}
        ]
        
    # Display Chat Messages
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Input Query
    user_query = st.chat_input("Type your farming question here...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        q_lower = user_query.lower()
        
        # ---------------------------------------------------------------------
        # DYNAMIC SMART RESPONSE ROUTER
        # ---------------------------------------------------------------------
        # 1. Bacterial vs Fungal vs Viral Diseases
        if any(k in q_lower for k in ["bacterial", "fungal", "viral", "difference between"]):
            response = """🔬 **Comparison of Plant Pathogens:**

• **Fungal Diseases (e.g., Blight, Mildew, Rust):**
  - *Symptoms:* Powdery coating, leaf spots with rings, wilting.
  - *Control:* Copper fungicides, Neem oil, reducing foliage moisture.

• **Bacterial Diseases (e.g., Bacterial Canker, Leaf Spot):**
  - *Symptoms:* Water-soaked lesions, slimy ooze, leaf scorching.
  - *Control:* Bactericides, clean tools, removing infected plants immediately.

• **Viral Diseases (e.g., Mosaic Virus, Leaf Curl):**
  - *Symptoms:* Mottled/stunted growth, curled leaves, yellow patterns.
  - *Control:* No direct cure; control insect vectors (whiteflies/aphids) and use resistant varieties."""

        # 2. Unclear Symptoms Guidance
        elif any(k in q_lower for k in ["unclear", "not sure", "confused", "identify"]):
            response = "🔍 **Unclear Symptoms Guidance:**\nIf symptoms are unclear:\n1. Capture clear photos of leaf surface and underside.\n2. Go to the **'Crop Disease Detection'** tab and upload the image.\n3. MobileNetV2 vision model will analyze and detect exact disease class."

        # 3. Fertilizer & NPK Dosage
        elif any(k in q_lower for k in ["fertilizer", "npk", "urea", "amount", "dosage"]):
            response = "🌱 **Fertilizer Recommendation:**\n- **Vegetative Phase:** Higher Nitrogen (Urea / NPK 20-10-10) for leaf growth.\n- **Flowering/Fruiting Phase:** Higher Phosphorus & Potassium.\n- Use the **'Fertilizer Recommendations'** tab for exact field dosage calculations!"

        # 4. Yellow Leaves & Tomato Specific Diseases
        elif any(k in q_lower for k in ["yellow", "brown spot", "early blight", "tomato"]):
            response = "🍂 **Tomato Leaf Spot Diagnosis:**\nYellow leaves with dark spots usually indicate **Early Blight** or **Septoria Leaf Spot**.\n- Prune lower infected leaves.\n- Avoid overhead sprinkler irrigation.\n- Apply copper-based fungicide or organic Neem oil."

        # 5. Soil pH & Health
        elif any(k in q_lower for k in ["ph", "soil", "acid", "alkaline"]):
            response = "🧪 **Soil pH Guidance:**\nOptimal soil pH for most crops is **6.0 - 7.0**.\n- **pH < 6.0 (Acidic):** Add agricultural lime.\n- **pH > 7.5 (Alkaline):** Add elemental sulfur or organic compost."

        # 6. Weather & Climate
        elif any(k in q_lower for k in ["weather", "rain", "temperature", "heat", "water"]):
            response = "🌤️ **Climate Guidance:**\nMonitor weather changes closely. In high temperatures, irrigate early morning to avoid evaporation. Check the **'Climate & Weather Advice'** tab for live weather updates!"

        # 7. Fallback General Response
        else:
            response = f"🌿 **Agronomic Advice:**\nRegarding *'{user_query}'*:\nMaintain optimal soil moisture, apply balanced fertilizers, and check the **'Crop Disease Detection'** tab to analyze leaf health."

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
# -----------------------------------------------------------------------------
# 3. CROP DISEASE DETECTION
# -----------------------------------------------------------------------------
elif "Crop Disease Detection" in nav_key:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #14532D 0%, #16A34A 60%, #22C55E 100%); padding: 2.2rem 2.5rem; border-radius: 18px; color: white; margin-bottom: 2rem; box-shadow: 0 12px 28px -6px rgba(22, 163, 74, 0.28);">
            <h1 style="color:white; margin:0; font-size:2.2rem; font-weight:800;">🌿 AI Leaf Disease Diagnosis</h1>
            <p style="color:#DCFCE7; margin-top:0.4rem; font-size:1.05rem;">Deep learning computer vision inference powered by MobileNetV2</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        uploaded_file = st.file_uploader("Upload plant leaf photo...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Target Sample Photo")
            
    with col2:
        if uploaded_file:
            with st.spinner("Executing MobileNetV2 Neural Network..."):
                try:
                    res = predict_crop_disease(image)
                    
                    # Case 1: Non-Plant Diagram / Screenshot
                    if res.get("is_non_plant", False):
                        st.markdown(f"""
                            <div class="agri-card" style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-left: 6px solid #EF4444;">
                                <h3 style="color: #991B1B; margin-top: 0;">{res['warning_title']}</h3>
                                <p style="color: #7F1D1D; font-size: 1.02rem; margin-top: 0.5rem;">{res['warning_message']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Case 2: Unsupported Crop (e.g. Rice, Wheat)
                    elif res.get("is_unsupported_crop", False):
                        st.markdown(f"""
                            <div class="agri-card" style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 6px solid #D97706;">
                                <h3 style="color: #92400E; margin-top: 0;">{res['warning_title']}</h3>
                                <p style="color: #78350F; font-size: 1.02rem; margin-top: 0.5rem;">{res['warning_message']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Case 3: Supported Crop Matched
                    else:
                        st.markdown(f"""
                            <div class="agri-card" style="border-left: 6px solid #16A34A;">
                                <h3 style="color:#15803D; margin-top:0;">Inference Summary</h3>
                                <p><b>Predicted Diagnosis:</b> <span class="badge-mint">{res['full_disease_name']}</span></p>
                                <p><b>Confidence Level:</b> <b>{res['confidence']:.2f}%</b></p>
                                <hr style="margin: 1rem 0; border-color: #E2E8F0;">
                                <p><b>🔍 Visual Symptoms:</b> {res['symptoms']}</p>
                                <p><b>🦠 Pathogen Agent:</b> {res['cause']}</p>
                                <p><b>🛠 Recommended Management:</b> {res['treatment']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Inference Error: {e}")
# -----------------------------------------------------------------------------
# 4. SOIL pH & HEALTH ANALYSIS (LIGHT CLAY EARTH THEME)
# -----------------------------------------------------------------------------
elif "Soil pH & Health Analysis" in nav_key:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #FFF4EA 0%, #FDE6D3 60%, #FAD7BB 100%); border: 1px solid #F3C9A8; padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 2rem; box-shadow: 0 12px 28px -10px rgba(194, 113, 63, 0.22);">
            <h1 style="color:#9A3412; margin:0; font-size:2.2rem; font-weight:800;">🧪 Soil Chemical Profile & Texture Visualizer</h1>
            <p style="color:#B45F2E; margin-top:0.4rem; font-size:1.05rem;">Target crop pH tolerance checking and visual soil texture matrix analysis</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        div[data-testid="stNumberInputStepDown"], div[data-testid="stNumberInputStepUp"], .stButton>button, div[data-testid="stForm"] button {
            background: linear-gradient(135deg, #D97F4E 0%, #EBA070 100%) !important;
            box-shadow: 0 4px 10px rgba(194, 113, 63, 0.25) !important;
        }
        div[data-baseweb="select"] svg { fill: #C2713F !important; }
        </style>
    """, unsafe_allow_html=True)

    with st.form("soil_form"):
        c1, c2 = st.columns(2)
        with c1:
            crop_choice = st.selectbox("Select Target Crop:", ["Tomato", "Potato", "Pepper Bell", "Wheat", "Maize (Corn)"])
            soil_ph = st.number_input("Soil pH Reading:", min_value=3.0, max_value=11.0, value=6.5, step=0.1)
        with c2:
            soil_type = st.selectbox("Soil Texture Profile:", ["Loam", "Sandy Loam", "Clay Loam", "Sandy", "Heavy Clay", "Silt Loam"])
            growth_stage = st.selectbox("Crop Growth Phase:", ["Seedling / Transplant", "Vegetative", "Flowering / Fruiting"])

        submit_soil = st.form_submit_button("Analyze Soil Profile")

    if submit_soil:
        soil_res = evaluate_soil_and_fertilizer(crop_choice, soil_ph, soil_type, growth_stage=growth_stage)
        soil_visual = get_soil_color_style(soil_type)
        marker_pos = max(0, min(100, (soil_ph - 3.0) / 8.0 * 100))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="agri-card" style="background-color: {soil_visual['bg']}; border: 2px solid {soil_visual['border']}; color: {soil_visual['text']};">
                <h3 style="color: {soil_visual['text']}; margin:0;">🏞️ Visual Soil Color Profile: {soil_type}</h3>
                <p style="color: {soil_visual['text']}; margin-top:0.4rem;"><b>Texture Characteristics:</b> {soil_visual['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="agri-card" style="border-left: 6px solid #D97F4E;">
                <h3 style="color:#B45F2E; margin-top:0;">Soil Chemical Analysis for {soil_res['crop']}</h3>
                <p><b>pH Condition Status:</b> <span class="badge-mint">{soil_res['ph_status']}</span></p>
                <div class="ph-scale"><div class="ph-marker" style="left: calc({marker_pos}% - 3px);"></div></div>
                <div class="ph-labels"><span>3 Acidic</span><span>7 Neutral</span><span>11 Alkaline</span></div>
                <p style="margin-top:1rem;">{soil_res['ph_analysis']}</p>
            </div>
        """, unsafe_allow_html=True)
        if soil_res["ph_remedy"]:
            st.info(f"**Required Soil Amendment:** {soil_res['ph_remedy']['amendment']}\n\n*Guidance:* {soil_res['ph_remedy']['guidance']}")

# -----------------------------------------------------------------------------
# 5. FERTILIZER RECOMMENDATIONS (AMBER GOLD / BUTTERY YELLOW THEME)
# -----------------------------------------------------------------------------
elif "Fertilizer Recommendations" in nav_key:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #FFFDF7 0%, #F6EDD6 100%); border: 1px solid #E7D8B4; padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 2rem; box-shadow: 0 12px 28px -10px rgba(176, 141, 87, 0.25);">
            <h1 style="color:#7A5C1E; margin:0; font-size:2.2rem; font-weight:800;">💊 Nutrient Deficiency & Fertilizer Advice</h1>
            <p style="color:#9A8050; margin-top:0.4rem; font-size:1.05rem;">Custom NPK deficiency calculations and soil amendment guidance</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        div[data-testid="stNumberInputStepDown"], div[data-testid="stNumberInputStepUp"], .stButton>button, div[data-testid="stForm"] button {
            background: linear-gradient(135deg, #B8893A 0%, #D4B06A 100%) !important;
            box-shadow: 0 4px 10px rgba(176, 141, 87, 0.3) !important;
        }
        div[data-baseweb="select"] svg { fill: #B8893A !important; }
        </style>
    """, unsafe_allow_html=True)

    with st.form("fert_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            c_choice = st.selectbox("Crop Type:", ["Tomato", "Potato", "Pepper Bell", "Wheat", "Maize (Corn)"])
            s_ph = st.number_input("Soil pH Level:", min_value=3.0, max_value=11.0, value=6.5)
        with fc2:
            n_val = st.number_input("Nitrogen N (mg/kg):", min_value=0, max_value=500, value=120)
            p_val = st.number_input("Phosphorus P (mg/kg):", min_value=0, max_value=200, value=12)
        with fc3:
            k_val = st.number_input("Potassium K (mg/kg):", min_value=0, max_value=600, value=150)

        submit_fert = st.form_submit_button("Calculate Fertilizer Plan")

    if submit_fert:
        fert_res = evaluate_soil_and_fertilizer(c_choice, s_ph, n_val=n_val, p_val=p_val, k_val=k_val)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommended Fertilizer Interventions</div>', unsafe_allow_html=True)
        for rec in fert_res["fertilizer_recommendations"]:
            st.markdown(f"""
                <div class="agri-card" style="background-color: #FFFCF5; border: 1px solid #EADFC4; border-left: 6px solid #C9A45C;">
                    <h3 style="color:#8A6A2A; margin:0;">💊 {rec['fertilizer']}</h3>
                    <p style="margin-top:0.5rem; color:#5E4A1F;"><b>Reason:</b> {rec['reason']}</p>
                    <p style="color:#5E4A1F;"><b>Application Guidance:</b> {rec['application']}</p>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. CLIMATE & WEATHER ADVICE (SKY BLUE THEME) - colors unchanged
# -----------------------------------------------------------------------------
elif "Climate & Weather Advice" in navigation or "Climate & Weather Advice" in st.session_state.nav_page:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
            <h2 style="color:white; margin:0;">🌤️ Real-Time Climate & Weather Guidance</h2>
            <p style="color:#E0F2FE; margin-top:0.4rem;">Fetch live location weather metrics and receive actionable crop protection advisories.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Inputs Section
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        city_input = st.text_input("Enter City / Region Name:", value="Lahore")
        fetch_btn = st.button("Fetch Live Weather Data", type="primary")
        
    with col_w2:
        season_cond = st.selectbox("Select Current / Forecasted Weather Event:", ["Normal / Seasonal Conditions", "Heavy Rain / Flood", "Heatwave", "Drought / Dry Spell", "Frost / Cold Snap"])
        soil_m = st.selectbox("Soil Moisture Status:", ["Optimal", "Low", "Excessive / Waterlogged"])

    # 2. Session State Initialization
    if "live_weather_data" not in st.session_state:
        st.session_state.live_weather_data = None

    # 3. Handle Button Click & Fetch Weather
    if fetch_btn:
        with st.spinner(f"Fetching real-time weather metrics for {city_input}..."):
            weather_res = fetch_live_weather(city_name=city_input)
            st.session_state.live_weather_data = weather_res
            st.success(f"Weather metrics updated for {city_input}!")

    # 4. Display Weather Data & Generated Advisories
    if st.session_state.live_weather_data:
        w_data = st.session_state.live_weather_data
        
        st.markdown("---")
        st.subheader(f"📍 Live Weather Metrics: {w_data.get('city', city_input)}")
        
        # Display Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Temperature", f"{w_data.get('temperature', 25.0)} °C")
        m2.metric("Humidity", f"{w_data.get('humidity', 60.0)} %")
        m3.metric("Wind Speed", f"{w_data.get('wind_speed', 3.0)} m/s")
        m4.metric("Condition", w_data.get('description', 'Clear'))

        st.markdown("---")
        st.subheader("💡 Agronomic Weather Advisory")
        
        # Generate and display advisory list
        advisories = generate_weather_advisory(
            crop_type="General Crops",
            manual_weather=season_cond,
            soil_moisture=soil_m,
            live_weather=w_data
        )
        
        for adv in advisories:
            st.info(adv)
elif "Crop Suitability Guidance" in navigation or "Crop Suitability Guidance" in st.session_state.nav_page:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #15803D 0%, #22C55E 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
            <h2 style="color:white; margin:0;">🌾 Crop Suitability & Growth Guidance</h2>
            <p style="color:#DCFCE7; margin-top:0.4rem;">Evaluate soil parameters, temperature ranges, and water requirements to find ideal crops for your farm.</p>
        </div>
    """, unsafe_allow_html=True)

    # Input Parameters Form
    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        target_ph = st.slider("Select Soil pH Level:", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
        soil_type_selected = st.selectbox("Select Soil Texture:", ["Loam", "Sandy Loam", "Clay Loam", "Silt Loam"])

    with col_cs2:
        temp_val = st.number_input("Average Temperature (°C):", min_value=0.0, max_value=50.0, value=25.0)
        water_avail = st.selectbox("Water Availability:", ["High", "Medium", "Low"])

    st.markdown("---")
    st.subheader("📋 Suitable Crop Recommendations")

    # Recommendation Logic
    recommended_crops = []

    # Tomato Logic
    if 5.8 <= target_ph <= 7.2 and 15 <= temp_val <= 32:
        recommended_crops.append({
            "name": "🍅 Tomato",
            "suitability": "High (Optimal Conditions)",
            "ph_range": "6.0 - 6.8",
            "water": "Medium to High",
            "advice": "Ideal conditions for tomato cultivation. Ensure proper drainage and support stakes."
        })

    # Potato Logic
    if 4.8 <= target_ph <= 6.8 and 12 <= temp_val <= 26:
        recommended_crops.append({
            "name": "🥔 Potato",
            "suitability": "High (Optimal Conditions)",
            "ph_range": "5.0 - 6.5",
            "water": "Medium",
            "advice": "Slightly acidic soil favors potato tuber development. Avoid waterlogging."
        })

    # Pepper Bell Logic
    if 5.8 <= target_ph <= 7.2 and 18 <= temp_val <= 33:
        recommended_crops.append({
            "name": "🫑 Pepper Bell (Capsicum)",
            "suitability": "High (Optimal Conditions)",
            "ph_range": "6.0 - 7.0",
            "water": "Medium",
            "advice": "Requires warm soil with uniform moisture levels for optimal yield."
        })

    # Maize / Corn Logic
    if 5.5 <= target_ph <= 7.5 and 18 <= temp_val <= 35:
        recommended_crops.append({
            "name": "🌽 Maize / Corn",
            "suitability": "Moderate to High",
            "ph_range": "5.8 - 7.0",
            "water": "High",
            "advice": "Thrives well in loam soil with good nitrogen management."
        })

    # Display Results
    if recommended_crops:
        for crop in recommended_crops:
            with st.expander(f"{crop['name']} - Suitability: {crop['suitability']}", expanded=True):
                st.write(f"**Ideal pH Range:** {crop['ph_range']}")
                st.write(f"**Water Need:** {crop['water']}")
                st.info(crop['advice'])
    else:
        st.warning("⚠️ No exact high-suitability crop match found for the selected soil pH and temperature combination. Try adjusting parameters closer to pH 6.0 - 7.0 and 20°C - 30°C.")
# -----------------------------------------------------------------------------
# 8. ABOUT PROJECT (SLATE STEEL THEME)
# -----------------------------------------------------------------------------
elif "About Project" in nav_key or "About Project" in st.session_state.get("nav_page", ""):
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #475569 60%, #64748B 100%); padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
            <h1 style="color:white; margin:0; font-size:2.2rem; font-weight:800;">ℹ️ Project Architecture & EDA Pipeline</h1>
            <p style="color:#CBD5E1; margin-top:0.4rem; font-size:1.05rem;">Technical stack breakdown, system design, and exploratory data analytics</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="agri-card" style="border-left: 6px solid #475569;">
            <h3 style="color:#1E293B; margin-top:0;">AgriVision AI Architecture</h3>
            <p>An enterprise-grade smart agriculture platform designed for smallholders and digital extension services.</p>
            <br>
            <p><b>System Core Libraries & Frameworks:</b></p>
            <ul>
                <li><b>Computer Vision Model:</b> MobileNetV2 trained on PlantVillage (~20,638 images)</li>
                <li><b>UI Framework:</b> Streamlit with custom CSS SaaS styling & Keyframe Animations</li>
                <li><b>Agronomic Engines:</b> Rule-based Soil pH, NPK Deficiency, Weather API, and Crop Suitability logic</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Render EDA Pipeline Dashboard with Soil pH Bar & Line Charts
    from modules.eda_pipeline import render_eda_dashboard
    render_eda_dashboard()