import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ====================================================
# ⚙️ CONFIG PAGE
# ====================================================
st.set_page_config(
    page_title="Analisis Ekonomi Indonesia",
    layout="wide",
    page_icon="📊"
)

# ====================================================
# 🎨 CSS FIX: SIDEBAR + STYLE MODERN
# ====================================================
st.markdown("""
<style>
/* Main background */
.main {
    background-color: #f8fafc;
}

/* Sidebar area */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    color: white !important;
}

/* Sidebar text color */
[data-testid="stSidebar"] * {
    color: white !important;
    font-family: 'Poppins', sans-serif;
}

/* Titles and headers */
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif;
    color: #0f172a;
}

/* Dataframe style */
.dataframe {
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}

/* Divider */
hr {
    border: 1px solid #cbd5e1;
}

/* Metric cards */
[data-testid="stMetricValue"] {
    color: #2563eb;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ====================================================
# 📂 LOAD DATA
# ====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("API_IDN_DS2_en_csv_v2_893274.csv", skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df

df = load_data()

# ====================================================
# 🧭 SIDEBAR NAVIGATION
# ====================================================
st.sidebar.title("📊 Dashboard Ekonomi Indonesia")
st.sidebar.markdown("### Pilih Analisis")
menu = st.sidebar.radio("", ["GDP Growth", "Unemployment"])

st.sidebar.markdown("---")
st.sidebar.caption("Dibuat oleh [Nama Kamu] 💡")

# ====================================================
# 📈 GDP GROWTH ANALYSIS
# ====================================================
if menu == "GDP Growth":
    st.title("📈 Analisis Pertumbuhan Ekonomi (GDP Growth) Indonesia")

    indicators = [
        "SL.UEM.TOTL.ZS", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "NE.GDI.TOTL.ZS",
        "SL.TLF.CACT.ZS", "NE.TRD.GNFS.ZS", "NE.CON.GOVT.ZS", "SP.POP.GROW"
    ]

    df_idn = df[(df["Country Code"] == "IDN") & (df["Indicator Code"].isin(indicators))]
    df_long = df_idn.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="Year", value_name="Value"
    )
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Value", "Year"])
    df_final = df_long.pivot_table(index="Year", columns="Indicator Code", values="Value").reset_index()

    df_final.rename(columns={
        "SL.UEM.TOTL.ZS": "Unemployment",
        "NY.GDP.MKTP.KD.ZG": "GDP_Growth",
        "FP.CPI.TOTL.ZG": "Inflation",
        "NE.GDI.TOTL.ZS": "Investment",
        "SL.TLF.CACT.ZS": "Labor_Participation",
        "NE.TRD.GNFS.ZS": "Trade_Openness",
        "NE.CON.GOVT.ZS": "Government_Expenditure",
        "SP.POP.GROW": "Population_Growth"
    }, inplace=True)
    df_final = df_final.dropna()

    # Subset untuk visualisasi
    df_viz = df_final[df_final["Year"].between(2017, 2023)]
    numeric_cols = [
        "Unemployment", "GDP_Growth", "Inflation", "Investment",
        "Labor_Participation", "Trade_Openness", "Government_Expenditure", "Population_Growth"
    ]

    st.subheader("📊 Tren Indikator Ekonomi (2017–2023)")
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 16))
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df_viz["Year"], df_viz[col], marker='o', color='tab:blue', linewidth=2)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.subheader("🔥 Korelasi Antar Variabel (2017–2023)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_viz[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.divider()

    st.subheader("📈 Distribusi Variabel (2017–2023)")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[col], kde=True, bins=15, color="steelblue", ax=ax)
        ax.set_title(f"Distribusi: {col}", fontsize=12, fontweight='bold')
        st.pyplot(fig)
    st.divider()

    # === Model Random Forest
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[[col for col in numeric_cols if col != "GDP_Growth"]]
    y = df_model["GDP_Growth"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("🤖 Evaluasi Model Random Forest")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("RMSE", f"{rmse:.3f}")
    st.divider()

# ====================================================
# 📉 UNEMPLOYMENT ANALYSIS
# ====================================================
elif menu == "Unemployment":
    st.title("📉 Analisis Tingkat Pengangguran (Unemployment) Indonesia")

    indicators = [
        "SL.UEM.TOTL.ZS", "SL.TLF.CACT.ZS", "SL.TLF.TOTL.IN", "SL.IND.EMPL.ZS",
        "NY.GDP.MKTP.KD.ZG", "NE.GDI.TOTL.ZS", "NE.TRD.GNFS.ZS",
        "SE.TER.ENRR", "SP.POP.1564.TO.ZS"
    ]

    df_unemp = df[(df["Country Code"] == "IDN") & (df["Indicator Code"].isin(indicators))]
    df_long = df_unemp.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="Year", value_name="Value"
    )
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Value", "Year"])
    df_final = df_long.pivot_table(index="Year", columns="Indicator Code", values="Value").reset_index()

    df_final.rename(columns={
        "SL.UEM.TOTL.ZS": "Unemployment",
        "SL.TLF.CACT.ZS": "Labor_Participation",
        "SL.TLF.TOTL.IN": "Labor_Force_Total",
        "SL.IND.EMPL.ZS": "Employment_Industry",
        "NY.GDP.MKTP.KD.ZG": "GDP_Growth",
        "NE.GDI.TOTL.ZS": "Investment",
        "NE.TRD.GNFS.ZS": "Trade_Openness",
        "SE.TER.ENRR": "Enrollment_Tertiary",
        "SP.POP.1564.TO.ZS": "Population_Productive"
    }, inplace=True)
    df_final = df_final.dropna()

    df_viz = df_final[df_final["Year"].between(2017, 2023)]
    numeric_cols = ["Unemployment", "Labor_Participation", "Employment_Industry",
                    "GDP_Growth", "Investment", "Trade_Openness"]

    st.subheader("📊 Tren Variabel (2017–2023)")
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 16))
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df_viz["Year"], df_viz[col], marker='o', color='tab:blue')
        axes[i].set_title(col)
        axes[i].grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.subheader("🔥 Korelasi Antar Variabel (2017–2023)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_viz[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.divider()

    # Ridge Regression
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]]
    y = df_model["Unemployment"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("🤖 Evaluasi Ridge Regression")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("RMSE", f"{rmse:.3f}")
    st.divider()

# ====================================================
# 🧾 FOOTER
# ====================================================
st.markdown("""
---
<center>
    <b>Dibuat dengan ❤️ oleh [Nama Kamu]</b><br>
    Sumber data: World Bank Open Data (2024)
</center>
""", unsafe_allow_html=True)
