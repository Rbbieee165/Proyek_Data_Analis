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

st.set_page_config(page_title="Analisis GDP & Unemployment", layout="wide")

# ==============================
# Fungsi untuk memuat data
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("API_IDN_DS2_en_csv_v2_893274.csv", skiprows=4)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df

df = load_data()

# ==============================
# Pilihan Variabel di Sidebar
# ==============================
st.sidebar.title("🔍 Pilih Analisis")
var_choice = st.sidebar.selectbox("Pilih variabel:", ["GDP Growth", "Unemployment"])

# ==============================
# Analisis GDP Growth
# ==============================
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

    # Visualisasi tren dan korelasi
    df_viz = df_final[df_final["Year"].between(2017, 2023)]
    numeric_cols = [
        "Unemployment", "GDP_Growth", "Inflation", "Investment",
        "Labor_Participation", "Trade_Openness", "Government_Expenditure", "Population_Growth"
    ]

    st.subheader("Tren Indikator Ekonomi (2017–2023)")
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in numeric_cols:
        ax.plot(df_viz["Year"], df_viz[col], marker='o', label=col)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Korelasi antar variabel (2017–2023)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_viz[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Variabel (2017–2023)")
    plt.style.use("seaborn-v0_8-whitegrid")

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[col], kde=True, bins=15, color="steelblue", ax=ax)
        ax.set_title(f"Distribusi Variabel: {col} (2017–2023)", fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Frekuensi")
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)    

    # Model Random Forest
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[[
        "Unemployment", "Inflation", "Investment", "Labor_Participation",
        "Trade_Openness", "Government_Expenditure", "Population_Growth"
    ]]
    y = df_model["GDP_Growth"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_full = rf_model.predict(X)

    # Visualisasi hasil
    st.subheader("Prediksi vs Aktual (2010–2023)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_model["Year"], y, label="Aktual", marker='o')
    ax.plot(df_model["Year"], y_pred_full, label="Prediksi", linestyle='--', marker='x')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Forecast 2024–2026
    recent_years = df_final[df_final["Year"].between(2021, 2023)]
    cols = ["Unemployment", "Inflation", "Investment", "Labor_Participation",
            "Trade_Openness", "Government_Expenditure", "Population_Growth"]
    growth_rates = {col: recent_years[col].pct_change().mean() for col in cols}
    future_years = [2024, 2025, 2026]
    last_values = recent_years.iloc[-1][cols]
    future_X = pd.DataFrame(columns=cols)
    for i, year in enumerate(future_years):
        row = last_values * [(1 + growth_rates[c])**(i+1) for c in cols]
        future_X.loc[i] = row
    future_pred = rf_model.predict(future_X)

    df_future = pd.DataFrame({"Year": future_years, "GDP_Growth_Predicted": future_pred})
    st.subheader("Prediksi GDP Growth 2024–2026")
    st.dataframe(df_future)

        # Visualisasi tren aktual + prediksi GDP Growth
    st.subheader("Tren GDP Growth Aktual dan Prediksi (2010–2026)")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot data aktual
    ax.plot(df_final["Year"], df_final["GDP_Growth"], marker='o', label="Aktual", color='tab:blue')

    # Plot hasil prediksi masa depan
    ax.plot(df_future["Year"], df_future["GDP_Growth_Predicted"],
            marker='x', linestyle='--', color='red', label="Prediksi (2024–2026)")

    # Tambahkan garis vertikal pembatas
    ax.axvline(2023, color='gray', linestyle='--', label="Batas Prediksi")

    # Label dan grid
    ax.set_title("Tren GDP Growth Indonesia (2010–2026)")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluasi Model Random Forest")
    st.write(f"R² (Test): {r2:.3f}")
    st.write(f"MAE: {mae:.3f}")
    st.write(f"RMSE: {rmse:.3f}")

# ============================================================
    # Feature Importance (Permutation Importance 2017–2023)
# ============================================================
    from sklearn.inspection import permutation_importance

    st.subheader("Feature Importance (2017–2023) | Model 2010–2023")

    # Siapkan model dari 2010–2023
    X_full = df_final[[
    "Unemployment", "Inflation", "Investment",
    "Labor_Participation", "Trade_Openness",
    "Government_Expenditure", "Population_Growth"
    ]]
    y_full = df_final["GDP_Growth"]

    rf_model_full = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf_model_full.fit(X_full, y_full)

    # Filter data 2017–2023 untuk visualisasi importance
    df_focus = df_final[df_final["Year"].between(2017, 2023)]
    X_focus = df_focus[[
        "Unemployment", "Inflation", "Investment",
        "Labor_Participation", "Trade_Openness",
        "Government_Expenditure", "Population_Growth"
    ]]
    y_focus = df_focus["GDP_Growth"]

    # Hitung permutation importance
    perm_importance = permutation_importance(
        rf_model_full, X_focus, y_focus, n_repeats=30, random_state=42
    )

    importance_df = pd.DataFrame({
        "Feature": X_focus.columns,
        "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)

    # Tampilkan tabel
    st.dataframe(importance_df)

    # Visualisasi diagram batang
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("Fitur")
    ax.set_title("Feature Importance / Faktor Berpengaruh (2017–2023)")
    ax.invert_yaxis()
    st.pyplot(fig)



# ==============================
# Analisis Unemployment
# ==============================
elif var_choice == "Unemployment":
    st.title("📉 Analisis Tingkat Pengangguran (Unemployment) Indonesia")

    indicators = [
        "SL.TLF.CACT.ZS", "SL.TLF.TOTL.IN", "SL.IND.EMPL.ZS",
        "NY.GDP.MKTP.KD.ZG", "NE.GDI.TOTL.ZS", "NE.TRD.GNFS.ZS",
        "SE.TER.ENRR", "SP.POP.1564.TO.ZS", "SL.UEM.TOTL.ZS"
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

    # Visualisasi tren dan korelasi
    df_viz = df_final[df_final["Year"].between(2017, 2023)]
    numeric_cols = ["Unemployment", "Labor_Participation", "Employment_Industry",
                    "GDP_Growth", "Investment", "Trade_Openness"]

    st.subheader("Tren Variabel Ekonomi (2017–2023)")
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 16))
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df_viz["Year"], df_viz[col], marker='o')
        axes[i].set_title(col)
        axes[i].grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Korelasi antar variabel (2017–2023)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_viz[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # 🔹 Tambahan: Distribusi Variabel (Histogram + KDE)
    # ==============================
    st.subheader("Distribusi Variabel (2017–2023)")
    plt.style.use("seaborn-v0_8-whitegrid")

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_viz[col], kde=True, bins=10, color="steelblue", ax=ax)
        ax.set_title(f"Distribusi Variabel: {col} (2017–2023)", fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Frekuensi")
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

    # Ridge Regression
    df_model = df_final[df_final["Year"].between(2010, 2023)]
    X = df_model[["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]]
    y = df_model["Unemployment"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)

    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.subheader("Evaluasi Ridge Regression")
    st.write(f"R² (Test): {r2:.3f}")
    st.write(f"MAE: {mae:.3f}")
    st.write(f"RMSE: {rmse:.3f}")

    # Forecasting sederhana
    features = ["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]
    df_last2 = df_final[df_final["Year"].isin([2022, 2023])][features + ["Year"]].set_index("Year")
    growth_rates = (df_last2.loc[2023] - df_last2.loc[2022]) / df_last2.loc[2022]

    pred_years = [2024, 2025, 2026]
    pred_values = []
    last_values = df_final[df_final["Year"] == 2023][features].iloc[0].values
    for y in pred_years:
        next_values = last_values * (1 + growth_rates.values)
        next_unemp = ridge_model.predict(scaler.transform([next_values]))[0]
        pred_values.append(next_unemp)
        last_values = next_values

    df_future = pd.DataFrame({"Year": pred_years, "Predicted_Unemployment": pred_values})
    st.subheader("Prediksi Unemployment 2024–2026")
    st.dataframe(df_future)

    # Visualisasi tren aktual & prediksi
    years_plot = list(df_final["Year"]) + pred_years
    unemp_plot = list(df_final["Unemployment"]) + pred_values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_plot, unemp_plot, marker='o', color='tab:blue')
    ax.axvline(2023, color='red', linestyle='--')
    ax.set_title("Tren Unemployment Indonesia (2010–2026)")
    ax.grid(True)
    st.pyplot(fig)

    # ==============================
# Feature Importance Random Forest - Unemployment (2017–2023)
# ==============================
    from sklearn.ensemble import RandomForestRegressor

    st.subheader("Feature Importance (Random Forest) | 2017–2023")

# Ambil model Random Forest untuk Unemployment (latih dengan seluruh data 2010–2023)
    X_full = df_model[["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]]
    y_full = df_model["Unemployment"]

    rf_unemp = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf_unemp.fit(X_full, y_full)

# Ambil subset 2017–2023
    df_focus = df_final[df_final["Year"].between(2017, 2023)]
    X_focus = df_focus[["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]]
    y_focus = df_focus["Unemployment"]

# Feature importance
    rf_feat_imp = rf_unemp.feature_importances_
    features = X_focus.columns

    df_importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf_feat_imp
    }).sort_values(by="Importance", ascending=False)

# Tampilkan tabel
    st.dataframe(df_importance)

# Visualisasi bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=df_importance, palette="magma", ax=ax)
    ax.set_title("Feature Importance Unemployment / Faktor Berpengaruh (2017–2023)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Fitur")
    st.pyplot(fig)

