# Proyek_Data_dan_Analisis_streamlit.py
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

st.set_page_config(page_title="Analisis GDP & Unemployment", layout="wide")

DATA_PATH = "API_IDN_DS2_en_csv_v2_893274.csv"  # ganti jika path berbeda

# ---------------------
# Utility: Load data
# ---------------------
@st.cache_data
def load_raw(path=DATA_PATH):
    try:
        df = pd.read_csv(path, skiprows=4, on_bad_lines="skip", low_memory=False, encoding="utf-8")
        df = df.dropna(axis=1, how="all")
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        st.error(f"File '{path}' tidak ditemukan. Upload lewat sidebar atau letakkan file di folder app.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None

# ---------------------
# Fungsi umum plotting
# ---------------------
def line_plot(x, ys_dict, title, xlabel="Year", ylabel="Value"):
    fig, ax = plt.subplots(figsize=(10,5))
    for name, y in ys_dict.items():
        ax.plot(x, y, marker='o', label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def heatmap_plot(df_num, title="Heatmap korelasi"):
    if df_num.shape[1] < 2:
        st.warning("Tidak cukup kolom numerik untuk heatmap.")
        return
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# ---------------------
# GDP pipeline (sesuai Colab)
# ---------------------
def prepare_gdp(df_raw):
    indicators = [
        "SL.UEM.TOTL.ZS",    # Unemployment (% of labor force)
        "NY.GDP.MKTP.KD.ZG", # GDP Growth (annual %)
        "FP.CPI.TOTL.ZG",    # Inflation (annual %)
        "NE.GDI.TOTL.ZS",    # Investment (% of GDP)
        "SL.TLF.CACT.ZS",    # Labor Participation Rate (%)
        "NE.TRD.GNFS.ZS",    # Trade Openness (% of GDP)
        "NE.CON.GOVT.ZS",    # Government Expenditure (% of GDP)
        "SP.POP.GROW"        # Population Growth (annual %)
    ]
    df = df_raw[(df_raw["Country Code"] == "IDN") & (df_raw["Indicator Code"].isin(indicators))]
    df_long = df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
                      var_name="Year", value_name="Value")
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
    return df_final

def run_gdp_analysis(df_final, train_year_from=2010, train_year_to=2023):
    df_model = df_final[df_final["Year"].between(train_year_from, train_year_to)].copy()
    features = ["Unemployment", "Inflation", "Investment",
                "Labor_Participation", "Trade_Openness",
                "Government_Expenditure", "Population_Growth"]
    X = df_model[features]
    y = df_model["GDP_Growth"]
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # model
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    # predictions
    y_pred = rf.predict(X_test)
    y_pred_full = rf.predict(X)
    # evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # future forecasting based on recent growth rates (as in Colab)
    recent_years = df_final[df_final["Year"].between(2021, 2023)]
    indicators = features
    growth_rates = {}
    for col in indicators:
        # pct_change mean over recent period
        growth_rates[col] = recent_years[col].pct_change().mean()
    future_years = [2024, 2025, 2026]
    last_values = recent_years.iloc[-1][indicators]
    future_X = []
    last_arr = last_values.values.astype(float)
    for i in range(len(future_years)):
        row = last_arr * np.array([ (1 + (growth_rates[col] if pd.notna(growth_rates[col]) else 0))**(i+1) for col in indicators ])
        future_X.append(row)
    future_X_df = pd.DataFrame(future_X, columns=indicators)
    future_pred = rf.predict(future_X_df)
    # feature importance (permutation on recent window)
    try:
        df_focus = df_final[df_final["Year"].between(2017, 2023)]
        X_focus = df_focus[features]
        y_focus = df_focus["GDP_Growth"]
        perm = permutation_importance(rf, X_focus, y_focus, n_repeats=20, random_state=42)
        importance_df = pd.DataFrame({"Feature": X_focus.columns, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
    except Exception:
        importance_df = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    return {
        "model": rf,
        "X": X, "y": y,
        "y_pred": y_pred, "y_test": y_test, "y_pred_full": y_pred_full,
        "mae": mae, "rmse": rmse, "r2": r2,
        "future_years": future_years, "future_pred": future_pred,
        "importance_df": importance_df, "features": features
    }

# ---------------------
# Unemployment pipeline (sesuai Colab)
# ---------------------
def prepare_unemp(df_raw):
    new_indicators = [
        "SL.TLF.CACT.ZS",      # Labor participation %
        "SL.TLF.TOTL.IN",      # Labor force total (count)
        "SL.IND.EMPL.ZS",      # Employment industry %
        "NY.GDP.MKTP.KD.ZG",   # GDP growth
        "NE.GDI.TOTL.ZS",      # Investment
        "NE.TRD.GNFS.ZS",      # Trade openness
        "SE.TER.ENRR",         # Enrollment tertiary
        "SP.POP.1564.TO.ZS"    # Population productive %
    ]
    target_indicator = ["SL.UEM.TOTL.ZS"]
    df = df_raw[(df_raw["Country Code"] == "IDN") & (df_raw["Indicator Code"].isin(new_indicators + target_indicator))]
    df_long = df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
                      var_name="Year", value_name="Value")
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
    return df_final

def run_unemp_analysis(df_final_unemp, train_year_from=2010, train_year_to=2023):
    df_model = df_final_unemp[df_final_unemp["Year"].between(train_year_from, train_year_to)].copy()
    features = ["Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]
    X = df_model[features].astype(float)
    y = df_model["Unemployment"].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)
    # RandomForest for comparison
    rf_unemp = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf_unemp.fit(X_train, y_train)
    y_pred_rf = rf_unemp.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    # Forecast 2024-2026 using recent growth rates (2022->2023)
    df_last2 = df_final_unemp[df_final_unemp["Year"].isin([2022, 2023])][features + ["Year"]].set_index("Year")
    growth_rates = (df_last2.loc[2023] - df_last2.loc[2022]) / df_last2.loc[2022]
    pred_years = [2024, 2025, 2026]
    pred_values = []
    last_values = df_final_unemp[df_final_unemp["Year"] == 2023][features].iloc[0].values.astype(float)
    for i in range(len(pred_years)):
        next_values = last_values * (1 + growth_rates.values)
        next_unemp = ridge.predict(scaler.transform([next_values]))[0]
        pred_values.append(next_unemp)
        last_values = next_values
    # Feature importance (permutation on 2017-2023)
    try:
        df_focus = df_final_unemp[df_final_unemp["Year"].between(2017, 2023)]
        X_focus = df_focus[features].astype(float)
        y_focus = df_focus["Unemployment"].astype(float)
        perm = permutation_importance(rf_unemp, X_focus, y_focus, n_repeats=20, random_state=42)
        importance_df = pd.DataFrame({"Feature": X_focus.columns, "Importance": perm.importances_mean}).sort_values("Importance", ascending=False)
    except Exception:
        importance_df = pd.DataFrame({"Feature": features, "Importance": rf_unemp.feature_importances_}).sort_values("Importance", ascending=False)
    return {
        "ridge": ridge, "rf": rf_unemp, "scaler": scaler,
        "y_test": y_test, "y_pred_ridge": y_pred_ridge, "y_pred_rf": y_pred_rf,
        "mae_ridge": mae_ridge, "rmse_ridge": rmse_ridge, "r2_ridge": r2_ridge,
        "mae_rf": mae_rf, "rmse_rf": rmse_rf, "r2_rf": r2_rf,
        "pred_years": pred_years, "pred_values": pred_values,
        "importance_df": importance_df, "features": features
    }

# ---------------------
# Streamlit UI
# ---------------------
st.sidebar.title("Pengaturan")
choice = st.sidebar.selectbox("Pilih analisis / variabel:", ["GDP_Growth (RandomForest)", "Unemployment (Ridge)"])
uploaded = st.sidebar.file_uploader("Upload CSV World Bank (opsional)", type=["csv"])
st.sidebar.markdown("Tips: gunakan file `API_IDN_DS2_en_csv_v2_893274.csv` dari World Bank untuk hasil yang sesuai Colab.")

# Load dataset
if uploaded is not None:
    with st.spinner("Memuat file upload..."):
        try:
            df_raw = pd.read_csv(uploaded, skiprows=4, on_bad_lines="skip", low_memory=False)
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            df_raw = None
else:
    with st.spinner("Memuat dataset..."):
        df_raw = load_raw()

if df_raw is None:
    st.stop()

# Jalankan pipeline sesuai pilihan
if "GDP_Growth" in choice:
    st.title("Analisis: GDP Growth (Random Forest)")
    with st.spinner("Menyiapkan data..."):
        df_gdp = prepare_gdp(df_raw)
    st.write("Data akhir (baris, kolom):", df_gdp.shape)
    st.dataframe(df_gdp.tail(10))
    # Viz 2017-2023
    df_viz = df_gdp[df_gdp["Year"].between(2017, 2023)]
    st.subheader("Tren 2017–2023 (beberapa indikator)")
    ys = {col: df_viz[col].values for col in ["GDP_Growth", "Unemployment", "Inflation", "Investment"]}
    line_plot(df_viz["Year"].values, ys, "Tren indikator (2017–2023)", ylabel="Persentase (%)")
    st.subheader("Heatmap Korelasi (2017–2023)")
    heatmap_plot(df_viz[["GDP_Growth","Unemployment","Inflation","Investment","Labor_Participation","Trade_Openness","Government_Expenditure","Population_Growth"]])
    # Modeling
    with st.spinner("Melatih Random Forest..."):
        res = run_gdp_analysis(df_gdp)
    st.success("Pelatihan selesai")
    st.metric("R² (test)", f"{res['r2']:.3f}")
    st.metric("RMSE (test)", f"{res['rmse']:.3f}")
    st.subheader("Perbandingan Aktual vs Prediksi (Test set)")
    comp = pd.DataFrame({"Year": res["y_test"].index, "Actual": res["y_test"].values, "Predicted": res["y_pred"]})
    st.dataframe(comp.reset_index(drop=True).head(20))
    # full series plot
    st.subheader("Prediksi vs Aktual (2010–2023)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(res["X"].index, res["y"].values, marker='o', label="Aktual")
    ax.plot(res["X"].index, res["y_pred_full"], marker='x', linestyle='--', label="Prediksi")
    ax.set_title("Prediksi vs Aktual (2010–2023)")
    ax.set_xlabel("Index (tahun)")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend()
    st.pyplot(fig)
    # Forecast
    st.subheader("Forecast 2024–2026 (estimasi sederhana berdasarkan growth rates)")
    df_future = pd.DataFrame({"Year": res["future_years"], "GDP_Growth_Predicted": res["future_pred"]})
    st.dataframe(df_future)
    # Importance
    st.subheader("Feature Importance (permutation / model)")
    st.dataframe(res["importance_df"].reset_index(drop=True))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.barh(res["importance_df"]["Feature"], res["importance_df"]["Importance"])
    ax2.invert_yaxis()
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

elif "Unemployment" in choice:
    st.title("Analisis: Unemployment (Ridge & RandomForest)")
    with st.spinner("Menyiapkan data..."):
        df_unemp = prepare_unemp(df_raw)
    st.write("Data akhir (baris, kolom):", df_unemp.shape)
    st.dataframe(df_unemp.tail(10))
    # Viz 2017-2023 subplots
    st.subheader("Tren 2017–2023 (variabel relevan)")
    df_viz_u = df_unemp[df_unemp["Year"].between(2017, 2023)]
    percent_cols = ["Unemployment", "Labor_Participation", "Employment_Industry", "GDP_Growth", "Investment", "Trade_Openness"]
    fig, axes = plt.subplots(len(percent_cols), 1, figsize=(10, 3*len(percent_cols)), sharex=True)
    for i, col in enumerate(percent_cols):
        axes[i].plot(df_viz_u["Year"], df_viz_u[col], marker='o')
        axes[i].set_title(col)
        axes[i].grid(True)
    st.pyplot(fig)
    st.subheader("Korelasi (2017–2023)")
    heatmap_plot(df_viz_u[percent_cols])
    # Modeling
    with st.spinner("Melatih Ridge & RandomForest..."):
        resu = run_unemp_analysis(df_unemp)
    st.success("Pelatihan selesai")
    st.write("Ridge - R² (test): {:.3f} | MAE: {:.3f} | RMSE: {:.3f}".format(resu["r2_ridge"], resu["mae_ridge"], resu["rmse_ridge"]))
    st.write("RandomForest - R² (test): {:.3f} | MAE: {:.3f} | RMSE: {:.3f}".format(resu["r2_rf"], resu["mae_rf"], resu["rmse_rf"]))
    # Prediksi test set (Ridge)
    st.subheader("Perbandingan Prediksi (Test set) - Ridge")
    df_comp_ridge = pd.DataFrame({"Actual": resu["y_test"].values, "Predicted_Ridge": resu["y_pred_ridge"]})
    st.dataframe(df_comp_ridge.head(20))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_comp_ridge["Actual"].values, label="Actual", marker='o')
    ax.plot(df_comp_ridge["Predicted_Ridge"].values, label="Predicted (Ridge)", marker='x')
    ax.legend(); st.pyplot(fig)
    # Forecast Unemployment
    st.subheader("Forecast Unemployment 2024–2026 (sederhana)")
    df_future_u = pd.DataFrame({"Year": resu["pred_years"], "Unemployment_Predicted": resu["pred_values"]})
    st.dataframe(df_future_u)
    # Feature importance
    st.subheader("Feature Importance (RandomForest permutation)")
    st.dataframe(resu["importance_df"].reset_index(drop=True))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.barh(resu["importance_df"]["Feature"], resu["importance_df"]["Importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)

# Footer note
st.write("---")
st.info("Catatan: Forecasting pada aplikasi ini menggunakan pendekatan sederhana (pertumbuhan rata-rata + model supervised). "
        "Untuk forecasting time-series yang lebih akurat, pertimbangkan ARIMA/Prophet/SARIMAX/LSTM.")
