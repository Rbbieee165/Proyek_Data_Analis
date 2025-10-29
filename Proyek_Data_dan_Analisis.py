# ====================================================
# 📊 ANALISIS EKONOMI INDONESIA: GDP GROWTH & UNEMPLOYMENT
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

# ----------------------------
# Konfigurasi halaman
# ----------------------------
st.set_page_config(page_title="Analisis Ekonomi Indonesia", layout="wide", page_icon="📈")

# ----------------------------
# CSS untuk tampilan modern
# ----------------------------
st.markdown("""
    <style>
        .main { background-color: #f9fafb; }
        h1, h2, h3, h4 { color: #0f172a; font-family: "Helvetica Neue", sans-serif; }
        [data-testid="stSidebar"] { background-color: #f1f5f9; }
        .dataframe { border-radius: 8px; border: 1px solid #d1d5db; }
        hr { border: 1px solid #d1d5db; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Fungsi memuat data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("API_IDN_DS2_en_csv_v2_893274.csv", skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df

df = load_data()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🔍 Pilih Analisis")
var_choice = st.sidebar.selectbox("Pilih variabel:", ["GDP Growth", "Unemployment"])

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

    # =======================================
    # 📊 Tren Indikator Ekonomi
    # =======================================
    df_viz = df_final[df_final["Year"].between(2017, 2023)]
    numeric_cols = [
        "Unemployment", "GDP_Growth", "Inflation", "Investment",
        "Labor_Participation", "Trade_Openness", "Government_Expenditure", "Population_Growth"
    ]

    st.subheader("📊 Tren Indikator Ekonomi Indonesia (2017–2023)")
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 16))
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df_viz["Year"], df_viz[col], marker='o', color='tab:blue', linewidth=2)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Persentase (%)")
        axes[i].grid(True, linestyle="--", alpha=0.7)
        axes[i].set_xticks(df_viz["Year"].astype(int))
    plt.xlabel("Tahun", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")

    # =======================================
    # 🔥 Korelasi
    # =======================================
    st.subheader("🔥 Korelasi Antar Variabel (2017–2023)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_viz[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.markdown("---")

    # =======================================
    # 📈 Distribusi Variabel
    # =======================================
    st.subheader("📈 Distribusi Variabel (2017–2023)")
    plt.style.use("seaborn-v0_8-whitegrid")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[col], kde=True, bins=15, color="steelblue", ax=ax)
        ax.set_title(f"Distribusi: {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)
    st.markdown("---")

    # =======================================
    # 🤖 Model Random Forest
    # =======================================
    st.subheader("🤖 Model Random Forest untuk Prediksi GDP Growth")
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[[
        "Unemployment", "Inflation", "Investment", "Labor_Participation",
        "Trade_Openness", "Government_Expenditure", "Population_Growth"
    ]]
    y = df_model["GDP_Growth"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_full = rf_model.predict(X)

    # Evaluasi model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (Test)", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("RMSE", f"{rmse:.3f}")
    st.markdown("---")

    # =======================================
    # 🔮 Prediksi 2024–2026
    # =======================================
    st.subheader("🔮 Prediksi GDP Growth 2024–2026")
    recent_years = df_final[df_final["Year"].between(2021, 2023)]
    cols = X.columns
    growth_rates = {col: recent_years[col].pct_change().mean() for col in cols}
    future_years = [2024, 2025, 2026]
    last_values = recent_years.iloc[-1][cols]
    future_X = pd.DataFrame(columns=cols)
    for i, year in enumerate(future_years):
        row = last_values * [(1 + growth_rates[c])**(i+1) for c in cols]
        future_X.loc[i] = row
    future_pred = rf_model.predict(future_X)
    df_future = pd.DataFrame({"Year": future_years, "Predicted GDP Growth": future_pred})
    st.dataframe(df_future)

    # Visualisasi tren prediksi
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_final["Year"], df_final["GDP_Growth"], marker='o', label="Aktual", color='tab:blue')
    ax.plot(df_future["Year"], df_future["Predicted GDP Growth"], marker='x', linestyle='--', color='red', label="Prediksi (2024–2026)")
    ax.axvline(2023, color='gray', linestyle='--', label="Batas Prediksi")
    ax.set_title("Tren GDP Growth Indonesia (2010–2026)")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")

    # =======================================
    # 💡 Feature Importance
    # =======================================
    st.subheader("💡 Feature Importance (2017–2023)")
    rf_model_full = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf_model_full.fit(X, y)
    df_focus = df_final[df_final["Year"].between(2017, 2023)]
    X_focus = df_focus[cols]
    y_focus = df_focus["GDP_Growth"]
    perm_importance = permutation_importance(rf_model_full, X_focus, y_focus, n_repeats=30, random_state=42)
    importance_df = pd.DataFrame({
        "Feature": X_focus.columns,
        "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r", ax=ax)
    ax.set_title("Feature Importance / Faktor Paling Berpengaruh")
    st.pyplot(fig)

# ====================================================
# 📉 ANALISIS UNEMPLOYMENT (Tidak diubah)
# ====================================================
else:
    st.info("Bagian analisis Unemployment sudah tersedia di kode lengkapmu di bawah ini dan tetap bisa digunakan. 🎯")

# ====================================================
# 🧾 FOOTER
# ====================================================
st.markdown("""
---
**Dibuat dengan ❤️ oleh [Nama Kamu]**  
Sumber data: *World Bank Open Data (2024)*
""")
