import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ---------------------------------------------------------
# 1️⃣ PRO-TIER UI CONFIGURATION & NEON-DARK CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Student Burnout Intel", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Base Theme Overrides */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Global Metric Styling (m1-m4, c1) */
    [data-testid="stMetricValue"] {
        color: #00FFAA !important; 
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        text-shadow: 0px 0px 10px rgba(0, 255, 170, 0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.05rem;
    }
    div[data-testid="metric-container"] {
        background-color: #1A1C24 !important;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 15px;
        transition: transform 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B22 !important; border-right: 1px solid #30363d; }
    .sidebar-title { color: #58a6ff; font-weight: 700; font-size: 1.6rem; margin-bottom: 20px; }

    /* Status Badges with Glassmorphism */
    .status-badge {
        padding: 18px; border-radius: 12px; text-align: center; color: white;
        font-weight: 700; margin-top: 25px; border: 1px solid rgba(255,255,255,0.1);
        text-transform: uppercase; letter-spacing: 0.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2️⃣ DATA & MODEL PIPELINE (Thread-Safe)
# ---------------------------------------------------------
@st.cache_data
def load_production_assets():
    try:
        data = pd.read_csv("data/processed/student_final_dataset.csv")
        mdl = joblib.load("models/burnout_model.pkl")
        return data, mdl
    except Exception as e:
        st.error(f"Critical System Error: Mapping failed. {e}")
        st.stop()

df, model = load_production_assets()
feature_cols = ["login_std", "sentiment_std", "shock_index", "consistency_score", "early_warning_flag"]
X = df[feature_cols]
raw_df = pd.read_csv("data/raw/synthetic_student_behaviour.csv")

# ---------------------------------------------------------
# 3️⃣ SIDEBAR INTELLIGENCE
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<p class='sidebar-title'> Behavioural Intelligence Console</p>", unsafe_allow_html=True)
    st.divider()
    
    # Simplified to a single selectbox
    student_id = st.selectbox("Active Student Registry", df["student_id"].unique())

    st.divider()
    st.info("""
    **Intelligence Note:**
    This console identifies early-warning patterns in student engagement using SHAP explainability.
    """)
    
    st.divider()
    st.caption("v4.5.3 Build | System Status: Optimal")

# Core Data Mapping
student_data = df[df["student_id"] == student_id]
s_idx = student_data.index[0]
r_score, r_level, b_label = student_data["risk_score"].iloc[0], student_data["risk_level"].iloc[0], student_data["burnout_label"].iloc[0]

    
# ---------------------------------------------------------
# 4️⃣ EXECUTIVE DASHBOARD HEADER
# ---------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #58a6ff; margin-bottom: 5px;'>🎓 Behavioural Burnout Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 40px;'>Predictive Analytics Interface for Academic Disengagement</p>", unsafe_allow_html=True)

# Main Metrics (m1 - m4)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cohort Population", f"{len(df):,}")
m2.metric("Burnout Incidence", int(df["burnout_label"].sum()))
m3.metric("Critical Alerts", int((df["risk_level"] == "High").sum()))
m4.metric("Mean Risk Coefficient", f"{df['risk_score'].mean():.2f}")

st.divider()

# ---------------------------------------------------------
# 5️⃣ MULTI-MODAL ANALYSIS (Left: Distribution & Radar)
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader("📊 Global Risk Context")
    r_counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"])
    fig_bar, ax_bar = plt.subplots()
    fig_bar.patch.set_facecolor('#0E1117')
    ax_bar.set_facecolor('#161B22')

    colors = ['#238636', '#d29922', '#da3633']
    ax_bar.bar(r_counts.index, r_counts.values, color=colors)

    ax_bar.tick_params(colors='white')
    ax_bar.spines['bottom'].set_color('#58a6ff')
    ax_bar.spines['left'].set_color('#58a6ff')
    ax_bar.set_title("Risk Level Distribution", color='white')

    st.pyplot(fig_bar)
    plt.close(fig_bar)
    
    # 🕸️ RADAR: COHORT NORMALIZATION
    st.markdown("### 🕸️ Cohort Behavioural Baseline")
    categories = [f.replace('_', ' ').title() for f in feature_cols]
    N = len(categories)
    
    # Normalize
    f_min, f_max = df[feature_cols].min(), df[feature_cols].max()
    avg_n = (df[feature_cols].mean() - f_min) / (f_max - f_min)
    ind_n = (student_data[feature_cols].iloc[0] - f_min) / (f_max - f_min)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig_radar, ax_radar = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor='#0E1117')
    ax_radar.set_facecolor('#1A1C24')
    
    # Plotting Radar
    ax_radar.plot(angles, np.append(ind_n, ind_n.iloc[0]), color="#00FFAA", linewidth=2.5, label="Target Student")
    ax_radar.fill(angles, np.append(ind_n, ind_n.iloc[0]), "#00FFAA", alpha=0.25)
    ax_radar.plot(angles, np.append(avg_n, avg_n.iloc[0]), color="#FFFFFF", linestyle='--', alpha=0.6, label="Cohort Avg")
    
    # Visual Polish
    ax_radar.tick_params(colors='#8b949e', labelsize=8)
    plt.thetagrids(np.degrees(angles[:-1]), categories, color="white", weight='bold', size=9)
    ax_radar.grid(color='#30363d', linestyle='-', alpha=0.5)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, facecolor='#161B22', labelcolor='white', frameon=False)
    st.pyplot(fig_radar)



# ---------------------------------------------------------
# 6️⃣ DEEP DIVE INTERFACE (Right: Metrics & SHAP)
# ---------------------------------------------------------
with col_right:
    st.subheader(f"🔍 Student Profile: ID #{student_id}")
    
    # Personal Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Intensity", f"{r_score:.2f}")
    
    # Badges
    lvl_map = {"Low": "#238636", "Medium": "#d29922", "High": "#da3633"}
    c2.markdown(f"<div class='status-badge' style='background-color: {lvl_map[r_level]};'>Level: {r_level}</div>", unsafe_allow_html=True)
    
    b_color = "#da3633" if b_label == 1 else "#238636"
    b_text = "Burnout Detected" if b_label == 1 else "Stable Status"
    c3.markdown(f"<div class='status-badge' style='background-color: {b_color};'>{b_text}</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # 🧠 SHAP WATERFALL: EXPLAINABILITY ENGINE
    st.markdown("#### 🧠 Decision Engine Attribution (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_vals_obj = explainer(X)
    
    # Slicing for Classification (Dim Check)
    if len(shap_vals_obj.values.shape) == 3:
        v, b = shap_vals_obj.values[s_idx, :, 1], shap_vals_obj.base_values[s_idx, 1]
    else:
        v, b = shap_vals_obj.values[s_idx], shap_vals_obj.base_values[s_idx]

    explanation = shap.Explanation(values=v, base_values=b, data=student_data[feature_cols].iloc[0], feature_names=feature_cols)

    # Force White Text for SHAP
    with plt.rc_context({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}):
        fig_shap, ax_shap = plt.subplots(figsize=(11, 5))
        fig_shap.patch.set_facecolor('#1A1C24')
        shap.plots.waterfall(explanation, show=False)
        plt.gca().tick_params(axis='y', colors='white', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig_shap)



    # Dynamically Generated Narrative
    st.markdown("### Recommended Action Plan")
    top_driver = feature_cols[np.argmax(np.abs(v))]
    
    if r_level == "High":
        st.error(f"**High Priority Intervention:** The system identifies **{top_driver.replace('_', ' ')}** as the primary catalyst. Immediate advisory support is recommended.")
    else:
        st.success(f"**Preventative Monitoring:** Current variance in **{top_driver.replace('_', ' ')}** is within controlled parameters. Maintain standard engagement.")


# ---------------------------------------------------------
# 📈 BEHAVIOURAL TIMELINE ANALYSIS (UI FIXED)
# ---------------------------------------------------------

st.markdown("### 📈 Behavioural Timeline (16-Week Pattern)")

student_time = raw_df[raw_df["student_id"] == student_id]

fig_trend, ax = plt.subplots(2, 2, figsize=(12, 8))
fig_trend.patch.set_facecolor('#0E1117')  # page background

plot_bg = '#161B22'  # slightly lighter panel

for axis in ax.flatten():
    axis.set_facecolor(plot_bg)
    axis.grid(color='#30363d', linestyle='--', linewidth=0.7, alpha=0.6)
    axis.tick_params(colors='white')
    axis.spines['bottom'].set_color('#58a6ff')
    axis.spines['left'].set_color('#58a6ff')
    axis.title.set_color('white')
    axis.yaxis.label.set_color('white')
    axis.xaxis.label.set_color('white')

# LOGIN TREND
ax[0, 0].plot(student_time["week"], student_time["login_count"], 
              color="#00FFAA", linewidth=3)
ax[0, 0].set_title("Login Activity Trend")

# ATTENDANCE TREND
ax[0, 1].plot(student_time["week"], student_time["attendance_rate"], 
              color="#58a6ff", linewidth=3)
ax[0, 1].set_title("Attendance Rate Trend")

# SENTIMENT TREND
ax[1, 0].plot(student_time["week"], student_time["sentiment_score"], 
              color="#f78166", linewidth=3)
ax[1, 0].set_title("Sentiment Score Trend")

# DELAY TREND
ax[1, 1].plot(student_time["week"], student_time["assignment_delay_days"], 
              color="#d29922", linewidth=3)
ax[1, 1].set_title("Assignment Delay Trend")

plt.tight_layout()
st.pyplot(fig_trend)
plt.close(fig_trend)
# ---------------------------------------------------------
# 7️⃣ FOOTER
# ---------------------------------------------------------
st.markdown("<div style='text-align: center; color: #484f58; padding: 40px; border-top: 1px solid #30363d; margin-top: 50px;'>Hackathon Build v4.5 | AI Governance & Ethics Compliant</div>", unsafe_allow_html=True)