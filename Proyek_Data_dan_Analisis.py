# Proyek_Data_dan_Analisis_streamlit.py
# Aplikasi Streamlit untuk analisis GDP Growth & Unemployment (World Bank dataset)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Analisis GDP & Unemployment", layout="wide")

DATA_PATH = "API_IDN_DS2_en_csv_v2_893274.csv"

# --- Fungsi Load Data ---
@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, skiprows=4, on_bad_lines="skip", low_memory=False, encoding="utf-8")
        df = df.dropna(axis=1, how="all")
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        st.error(f"❌ File '{path}' tidak ditemukan. Upload lewat sidebar atau letakkan di folder yang sama dengan app.")
        return None
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        return None

# --- Fungsi Membuat Fitur Lag ---
def create_lag_features(df, col="Target", lags=3):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

# --- Fungsi Pelatihan & Prediksi ---
def train_and_predict(df, target_col="Target", model_type="rf", test_size=0.2, random_state=42):
    df = df.copy().dropna(subset=[target_col])
    df = create_lag_features(df, target_col, lags=3)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    scaler = None
    if model_type == "ridge":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=random_state) if model_type == "rf" else Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Forecast sederhana
    last_row = df.iloc[-1:].copy()
    preds_future = []
    for _ in range(6):
        X_cur = last_row[feature_cols]
        if model_type == "ridge" and scaler is not None:
            X_cur = scaler.transform(X_cur)
        pred = model.predict(X_cur)[0]
        preds_future.append(pred)
        for lag in range(3, 1, -1):
            last_row[f"{target_col}_lag{lag}"] = last_row[f"{target_col}_lag{lag-1}"]
        last_row[f"{target_col}_lag1"] = pred

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
        "preds_future": preds_future,
        "feature_cols": feature_cols,
    }

# --- Visualisasi Tren ---
def plot_trend(df_long, label="Target"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_long["Year"], df_long["Target"], marker="o", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.set_title(f"Tren {label}")
    st.pyplot(fig)

# --- Visualisasi Heatmap ---
def plot_heatmap(df_long):
    num = df_long.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.warning("Tidak cukup kolom numerik untuk heatmap.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Visualisasi Feature Importance ---
def show_feature_importance(model, feature_cols, model_type):
    if model_type == "rf" and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif model_type == "ridge" and hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        st.warning("Model tidak memiliki feature importance.")
        return

    dfimp = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
    st.write("### Feature Importance")
    st.dataframe(dfimp)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dfimp["feature"], dfimp["importance"])
    ax.set_xticklabels(dfimp["feature"], rotation=45, ha="right")
    st.pyplot(fig)

# ======================
# === BAGIAN UTAMA ====
# ======================

st.sidebar.title("⚙️ Pengaturan")
indicator_choice = st.sidebar.selectbox(
    "Pilih indikator:",
    ["GDP growth (annual %)", "Unemployment, total (% of total labor force)"]
)
uploaded = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])
st.sidebar.markdown("---")

# Load data
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded, skiprows=4, on_bad_lines="skip", low_memory=False)
    except Exception as e:
        st.error(f"Gagal membaca file upload: {e}")
        df_raw = None
else:
    df_raw = load_data()

if df_raw is None:
    st.stop()

# Filter & transformasi
df_filtered = df_raw[df_raw["Indicator Name"] == indicator_choice]
if df_filtered.empty:
    st.error(f"Indikator '{indicator_choice}' tidak ditemukan dalam dataset.")
    st.stop()

df_long = df_filtered.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    var_name="Year",
    value_name="Value"
)
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long = df_long.dropna(subset=["Value"]).reset_index(drop=True)
df_long = df_long.rename(columns={"Value": "Target"})

# Tampilan utama
st.title("📊 Dashboard Analisis: GDP Growth & Unemployment")
st.write(f"Indikator: {indicator_choice}")
st.dataframe(df_long.head())

plot_trend(df_long, label=indicator_choice)
plot_heatmap(df_long)

with st.spinner("Melatih model dan membuat prediksi..."):
    df_feat = df_long[["Target"]].copy()
    result = train_and_predict(df_feat, target_col="Target", model_type="rf" if "GDP" in indicator_choice else "ridge")

st.success("✅ Pelatihan selesai!")
st.write(f"**RMSE:** {result['rmse']:.4f}, **R²:** {result['r2']:.4f}")

st.write("#### Perbandingan Prediksi (Test Set)")
st.dataframe(pd.DataFrame({"Actual": result["y_test"], "Predicted": result["y_pred"]}).head(20))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result["y_test"].values, label="Actual", marker="o")
ax.plot(result["y_pred"].values, label="Predicted", marker="x")
ax.legend()
st.pyplot(fig)

st.write("#### Forecast (6 Langkah ke Depan)")
st.dataframe(pd.DataFrame({"Step": range(1, len(result["preds_future"]) + 1), "Predicted": result["preds_future"]}))

show_feature_importance(result["model"], result["feature_cols"], "rf" if "GDP" in indicator_choice else "ridge")

st.info("Catatan: Forecasting menggunakan fitur lag sederhana dan model non-sequence. "
        "Untuk hasil lebih akurat, gunakan model time-series seperti ARIMA, Prophet, atau LSTM.")
