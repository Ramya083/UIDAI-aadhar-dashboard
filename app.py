#%%writefile aadhaar_streamlit/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


st.set_page_config(
    page_title="UIDAI Aadhaar Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #eaf3f1; }

section[data-testid="stSidebar"] {
    background-color: #1f2430;
}
section[data-testid="stSidebar"] * {
    color: white;
}

.topbar {
    background: white;
    padding: 18px 28px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

.kpi {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
.kpi h2 { margin: 0; font-size: 26px; }
.kpi p { color: gray; margin: 0; font-size: 14px; }

.panel {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(folder="aadhaar_streamlit/data"):
    dfs = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            df.columns = df.columns.str.strip().str.lower()
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_data()


df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

df["total_enrolments"] = (
    df["age_0_5"].fillna(0)
    + df["age_5_17"].fillna(0)
    + df["age_18_greater"].fillna(0)
)


st.sidebar.title("📊 UIDAI")
st.sidebar.markdown("Aadhaar Enrolment Dashboard")

state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["state"].unique())
)

filtered = df if state == "All" else df[df["state"] == state]


st.markdown("""
<div class="topbar">
    <h1>UIDAI Aadhaar Dashboard</h1>
    <p style="color:gray;">National Enrolment Analytics – 2025</p>
</div>
""", unsafe_allow_html=True)


total = filtered["total_enrolments"].sum()
children = filtered["age_0_5"].sum() + filtered["age_5_17"].sum()
adult = filtered["age_18_greater"].sum()
daily_avg = total / filtered["date"].nunique()

k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"<div class='kpi'><h2>{int(total):,}</h2><p>Total Enrolments</p></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'><h2>{int(children):,}</h2><p>Child Enrolments</p></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'><h2>{(adult/total)*100:.1f}%</h2><p>Adult Coverage</p></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi'><h2>{int(daily_avg):,}</h2><p>Daily Avg</p></div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.subheader("📈 Enrolment Trend")

trend = filtered.groupby("date")["total_enrolments"].sum()
fig1, ax1 = plt.subplots(figsize=(8,3))
trend.plot(ax=ax1)
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.subheader("📊 State-wise Enrolments")

state_data = filtered.groupby("state")["total_enrolments"].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(8,3))
state_data.head(10).plot(kind="bar", ax=ax2)
st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)


if state != "All":

    # ---- District ----
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader(f"🏙️ District-wise Enrolments – {state}")

    district_data = filtered.groupby("district")["total_enrolments"].sum().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8,3))
    district_data.head(10).plot(kind="bar", ax=ax3)
    st.pyplot(fig3)
    st.dataframe(district_data.reset_index(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- PINCODE HEATMAP (FIXED) ----
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("📍 Pincode Enrolment Heatmap")

    pincode_series = filtered.groupby("pincode")["total_enrolments"].sum().sort_values(ascending=False)

    top_n = 50
    values = pincode_series.head(top_n).values.astype(float)  # ✅ FIX

    cols = 10
    rows = int(np.ceil(len(values) / cols))

    padded = np.pad(values, (0, rows * cols - len(values)), constant_values=np.nan)
    heatmap = padded.reshape(rows, cols)

    fig4, ax4 = plt.subplots(figsize=(6,4))
    im = ax4.imshow(heatmap, cmap="YlOrRd")
    ax4.set_xticks([])
    ax4.set_yticks([])

    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Total Enrolments", rotation=270, labelpad=15)

    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.subheader("🤖 AI Summary Insights")

if st.button("Generate Insights"):
    top_state = df.groupby("state")["total_enrolments"].sum().idxmax()
    st.success(f"""
• Total Aadhaar enrolments recorded: {int(total):,}
• Child enrolments account for {children/total*100:.1f}% of total.
• Adult population dominates Aadhaar coverage.
• {top_state} leads in Aadhaar enrolments nationwide.
• Enrolment activity shows stable infrastructure performance.
""")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    "<p style='text-align:center;color:gray;'>UIDAI Aadhaar Enrolment Dashboard • 2025</p>",
    unsafe_allow_html=True
)
