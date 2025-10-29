# ====================================================
# 📊 ANALISIS EKONOMI INDONESIA
# GDP GROWTH & UNEMPLOYMENT DASHBOARD
# ====================================================

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
# 🧭 Konfigurasi & Tema Visual
# ====================================================

st.set_page_config(page_title="Analisis Ekonomi Indonesia", layout="wide", page_icon="📈")

# 🌈 CSS modern + perbaikan sidebar
st.markdown("""
    <style>
        /* Latar utama */
        .main { background-color: #f9fafb; }

        /* Warna sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1e293b;
            color: white;
        }

        /* Teks di sidebar */
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] div {
            color: white !important;
        }

        /* Judul & heading */
        h1, h2, h3, h4 {
            color: #0f172a;
            font-family: "Poppins", sans-serif;
        }

        /* Garis pemisah */
        hr { border: 1px solid #d1d5db; }

        /* Tabel data */
        .dataframe { border-radius: 8px; border: 1px solid #d1d5db; }
    </style>
""", unsafe_allow_html=True)


# ====================================================
# 📂 Fungsi Memuat Data
# ====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("API_IDN_DS2_en_csv_v2_893274.csv", skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df

df = load_data()

# ====================================================
# 📊 Sidebar
# ====================================================

st.sidebar.title("📊 Dashboard Analisis Ekonomi")
var_choice = st.sidebar.radio("Pilih Analisis:", ["GDP Growth", "Unemployment"])

# ====================================================
# 📈 ANALISIS GDP GROWTH
# ====================================================

if var_choice == "GDP Growth":
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

    df_viz = df_final[df_final["Year"].between(2017, 2023)]
    numeric_cols = [
        "Unemployment", "GDP_Growth", "Inflation", "Investment",
        "Labor_Participation", "Trade_Openness",
        "Government_Expenditure", "Population_Growth"
    ]

    # === Visualisasi Tren
    st.subheader("📊 Tren Indikator Ekonomi (2017–2023)")
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 16))
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df_viz["Year"], df_viz[col], marker='o', color='tab:blue', linewidth=2)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Persentase (%)")
        axes[i].grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Tahun", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    # === Korelasi
    st.subheader("🔥 Korelasi Antar Variabel (2017–2023)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_viz[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.divider()

    # === Distribusi
    st.subheader("📈 Distribusi Variabel (2017–2023)")
    plt.style.use("seaborn-v0_8-whitegrid")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[col], kde=True, bins=15, color="steelblue", ax=ax)
        ax.set_title(f"Distribusi: {col}", fontsize=12, fontweight='bold')
        st.pyplot(fig)
    st.divider()

    # === Model
    st.subheader("🤖 Model Random Forest: Prediksi GDP Growth")
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[[col for col in numeric_cols if col != "GDP_Growth"]]
    y = df_model["GDP_Growth"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae, rmse, r2 = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (Test)", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("RMSE", f"{rmse:.3f}")
    st.divider()

    # === Prediksi 2024–2026
    st.subheader("🔮 Prediksi GDP Growth 2024–2026")
    recent = df_final[df_final["Year"].between(2021, 2023)]
    growth_rates = {col: recent[col].pct_change().mean() for col in X.columns}
    future = pd.DataFrame(columns=X.columns)
    for i, year in enumerate([2024, 2025, 2026]):
        row = recent.iloc[-1][X.columns] * [(1 + growth_rates[c])**(i + 1) for c in X.columns]
        future.loc[year] = row
    preds = model.predict(future)
    st.dataframe(pd.DataFrame({"Year": [2024, 2025, 2026], "Predicted GDP Growth": preds}))
    st.divider()

    # === Feature Importance
    st.subheader("💡 Feature Importance (2017–2023)")
    imp = permutation_importance(model, X_train, y_train, n_repeats=30, random_state=42)
    imp_df = pd.DataFrame({"Feature": X.columns, "Importance": imp.importances_mean}).sort_values(by="Importance", ascending=False)
    st.dataframe(imp_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
    st.pyplot(fig)

# ====================================================
# 📉 ANALISIS UNEMPLOYMENT
# ====================================================
elif var_choice == "Unemployment":
    st.title("📉 Analisis Tingkat Pengangguran (Unemployment) Indonesia")

    indicators = [
        "SL.UEM.TOTL.ZS", "SL.TLF.CACT.ZS", "SL.TLF.TOTL.IN", "SL.IND.EMPL.ZS",
        "NY.GDP.MKTP.KD.ZG", "NE.GDI.TOTL.ZS", "NE.TRD.GNFS.ZS",
        "SE.TER.ENRR", "SP.POP.1564.TO.ZS"
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

    st.subheader("📈 Distribusi Variabel (2017–2023)")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[col], kde=True, bins=10, color="steelblue", ax=ax)
        ax.set_title(f"Distribusi: {col}", fontsize=12)
        st.pyplot(fig)
    st.divider()

    st.subheader("🤖 Ridge Regression: Prediksi Unemployment")
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]]
    y = df_model["Unemployment"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, rmse, r2 = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (Test)", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("RMSE", f"{rmse:.3f}")

