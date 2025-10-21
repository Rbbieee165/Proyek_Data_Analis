# Proyek_Data_dan_Analisis.py
# Versi final - fix indikator World Bank & tampilkan analisis penuh

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

# --- Nama file dataset ---
DATA_PATH = "API_IDN_DS2_en_csv_v2_893274.csv"

# --- Fungsi untuk memuat dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            DATA_PATH,
            skiprows=4,           # lewati metadata World Bank
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8"
        )
        df = df.dropna(axis=1, how="all")
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        st.error(f"❌ File '{DATA_PATH}' tidak ditemukan. Pastikan file CSV ada di folder yang sama dengan aplikasi.")
        return None
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        return None

# --- Buat fitur lag sederhana ---
def create_lag_features(df, col, lags=3):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

# --- Pelatihan model dan prediksi ---
def train_and_predict(df, target_col, model_type="rf", test_size=0.2, random_state=42):
    df = df.copy().dropna(subset=[target_col])
    df = create_lag_features(df, target_col, lags=3)
    feature_cols = [c for c in df.columns if c not in [target_col]]
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

    # Forecasting sederhana
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

# --- Visualisasi ---
def plot_trend(df, col, label):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Year"], df[col], marker="o", linewidth=1, label=label)
    ax.set_title(f"Tren Historis - {label}")
    ax.set_xlabel("Tahun")
    ax.set_ylabel(label)
    ax.legend()
    st.pyplot(fig)

def plot_heatmap(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.warning("Tidak cukup kolom numerik untuk membuat heatmap.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi (Variabel Numerik)")
    st.pyplot(fig)

def show_feature_importance(model, feature_cols, model_type):
    if model_type == "rf" and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif model_type == "ridge" and hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        st.warning("Model tidak memiliki informasi feature importance.")
        return
    df_imp = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
    st.write("### Feature Importance")
    st.dataframe(df_imp)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_imp["feature"], df_imp["importance"])
    ax.set_xticklabels(df_imp["feature"], rotation=45, ha="right")
    st.pyplot(fig)

# --- UI Aplikasi ---
st.sidebar.title("Pengaturan")

indicator_choice = st.sidebar.selectbox(
    "Pilih indikator untuk analisis:",
    ["GDP growth (annual %)", "Unemployment, total (% of total labor force)"]
)

uploaded = st.sidebar.file_uploader("Upload file CSV (opsional)", type=["csv"])
st.sidebar.markdown("---")

# --- Load data ---
if uploaded is not None:
    df = pd.read_csv(uploaded, skiprows=4, on_bad_lines="skip", low_memory=False)
else:
    with st.spinner("📊 Sedang memuat data..."):
        df = load_data()

if df is None:
    st.stop()

# --- Filter data berdasarkan indikator ---
df_filtered = df[df["Indicator Name"] == indicator_choice]
if df_filtered.empty:
    st.error(f"Tidak menemukan indikator '{indicator_choice}' di dataset.")
    st.stop()

# --- Ubah dari wide ke long ---
df_long = df_filtered.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    var_name="Year",
    value_name="Value"
)
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long = df_long.dropna(subset=["Value"])
df_long = df_long.reset_index(drop=True)
df_long.rename(columns={"Value": "Target"}, inplace=True)

# --- Tampilan utama ---
st.title("📈 Dashboard Analisis: GDP Growth & Unemployment")
st.write(f"### Indikator: {indicator_choice}")
st.dataframe(df_long.head())

# --- Visualisasi tren dan korelasi ---
plot_trend(df_long, "Target", indicator_choice)
plot_heatmap(df_long)

# --- Analisis & Forecasting ---
with st.spinner("🔮 Melatih model dan membuat prediksi..."):
    df_feat = create_lag_features(df_long[["Target"]], "Target", lags=3)
    model_type = "rf" if "GDP" in indicator_choice else "ridge"
    result = train_and_predict(df_feat, "Target", model_type=model_type)

st.success("✅ Analisis selesai!")
st.write(f"**Model:** {'Random Forest' if model_type == 'rf' else 'Ridge Regression'}")
st.write(f"**RMSE:** {result['rmse']:.4f} | **R²:** {result['r2']:.4f}")

# --- Hasil prediksi ---
comp = pd.DataFrame({"Actual": result["y_test"], "Predicted": result["y_pred"]})
st.write("##### Perbandingan Hasil (Test Set)")
st.dataframe(comp.head(20))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(comp["Actual"].values, label="Actual", marker="o")
ax.plot(comp["Predicted"].values, label="Predicted", marker="x")
ax.legend()
ax.set_title("Perbandingan Aktual vs Prediksi")
st.pyplot(fig)

# --- Forecast ke depan ---
st.write("#### Prediksi ke Depan (Forecasting Sederhana)")
future_df = pd.DataFrame({"Step": range(1, len(result["preds_future"]) + 1), "Predicted": result["preds_future"]})
st.dataframe(future_df)

show_feature_importance(result["model"], result["feature_cols"], model_type)

st.write("---")
st.info(
    "Catatan: Metode forecasting ini menggunakan fitur lag dari target dan model ML non-sequence. "
    "Untuk hasil yang lebih akurat, gunakan model khusus time-series seperti ARIMA, Prophet, atau LSTM."
)
