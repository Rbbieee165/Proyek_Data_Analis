
# Proyek_Data_dan_Analisis.py
# Aplikasi Streamlit hasil konversi otomatis dari Proyek_Data_dan_Analisis.ipynb
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
import os

st.set_page_config(page_title="Analisis GDP & Unemployment", layout="wide")

DATA_PATH = None

def load_data():
    if DATA_PATH and os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    for p in os.listdir("/mnt/data"):
        if p.lower().endswith(".csv"):
            try:
                return pd.read_csv(os.path.join("/mnt/data", p))
            except Exception:
                pass
    st.error("Tidak menemukan file CSV otomatis. Silakan upload file CSV di bagian sidebar atau letakkan file CSV di folder aplikasi.")
    return None

def create_lag_features(df, col, lags=3):
    df = df.copy()
    for lag in range(1, lags+1):
        df[f"{{col}}_lag{{lag}}"] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def train_and_predict(df, target_col, model_type="rf", test_size=0.2, random_state=42):
    df2 = df.copy()
    time_cols = [c for c in df2.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
    if time_cols:
        time_col = time_cols[0]
    else:
        time_col = None
    df_lag = create_lag_features(df2, target_col, lags=3)
    feature_cols = [c for c in df_lag.columns if c not in [target_col]]
    X = df_lag[feature_cols]
    y = df_lag[target_col]
    scaler = None
    if model_type == "ridge":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    else:
        model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    last_row = df_lag.iloc[-1:].copy()
    preds = []
    n_steps = 6
    cur = last_row.copy()
    for i in range(n_steps):
        X_cur = cur[feature_cols]
        if model_type == "ridge" and scaler is not None:
            X_cur = scaler.transform(X_cur)
        p = model.predict(X_cur)[0]
        preds.append(p)
        for lag in range(3, 1, -1):
            cur[f"{target_col}_lag{{lag}}"] = cur[f"{target_col}_lag{{lag-1}}"]
        cur[f"{target_col}_lag1"] = p
    results = {
        "model": model,
        "rmse": rmse,
        "r2": r2,
        "y_test": y_test.values,
        "y_pred": y_pred,
        "preds_future": preds,
        "feature_cols": feature_cols
    }
    return results

def plot_trend(df, col):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df[col], marker='o', linewidth=1)
    ax.set_title(f"Tren Historis - {{col}}")
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    st.pyplot(fig)

def plot_heatmap(df):
    num = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi (variabel numerik)")
    st.pyplot(fig)

def show_feature_importance(model, feature_cols, model_type):
    if model_type == "rf":
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            dfimp = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
        else:
            st.write("Model tidak punya feature_importances_")
            return
    else:
        if hasattr(model, "coef_"):
            imp = model.coef_
            dfimp = pd.DataFrame({"feature": feature_cols, "importance": np.abs(imp)}).sort_values("importance", ascending=False)
        else:
            st.write("Model tidak punya coef_")
            return
    st.write("### Feature importance")
    st.dataframe(dfimp)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(dfimp["feature"], dfimp["importance"])
    ax.set_xticklabels(dfimp["feature"], rotation=45, ha="right")
    st.pyplot(fig)

st.sidebar.title("Pengaturan")
st.sidebar.markdown("Pilih variabel untuk analisis:")
var_choice = st.sidebar.selectbox("Analisis Variabel", ["GDP_Growth", "Unemployment"])
st.sidebar.markdown("---")
if st.sidebar.button("Reload dataset"):
    st.experimental_rerun()
uploaded = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_data()
if df is None:
    st.stop()
st.title("Dashboard Analisis: GDP Growth & Unemployment")
st.markdown("Aplikasi ini menampilkan analisis tren, korelasi, prediksi, dan feature importance untuk variabel GDP_Growth dan Unemployment.")
st.write("### Preview data")
st.dataframe(df.head())
if var_choice not in df.columns:
    st.error(f"Kolom target '{{var_choice}}' tidak ditemukan dalam dataset. Kolom tersedia: {{', '.join(df.columns)}}")
else:
    st.subheader(f"Analisis untuk: {{var_choice}}")
    st.write("#### Tren historis")
    plot_trend(df, var_choice)
    st.write("#### Heatmap korelasi")
    plot_heatmap(df)
    st.write("#### Pelatihan model & Prediksi")
    model_type = "rf" if var_choice == "GDP_Growth" else "ridge"
    with st.spinner("Melatih model..."):
        res = train_and_predict(df, var_choice, model_type=model_type)
    st.write(f"**Model**: {{'RandomForestRegressor' if model_type=='rf' else 'Ridge'}}")
    st.write(f"RMSE: {{res['rmse']:.4f}}  |  R2: {{res['r2']:.4f}}")
    comp = pd.DataFrame({{'actual': res['y_test'], 'predicted': res['y_pred']}})
    st.write("##### Perbandingan (Test set)")
    st.dataframe(comp.head(20))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(comp['actual'], label='actual', marker='o')
    ax.plot(comp['predicted'], label='predicted', marker='x')
    ax.legend()
    ax.set_title("Actual vs Predicted (Test set)")
    st.pyplot(fig)
    st.write("#### Prediksi masa depan (multi-step, pendekatan lag-based)")
    future = res['preds_future']
    st.write(pd.DataFrame({{'step': list(range(1, len(future)+1)), 'predicted': future}}))
    show_feature_importance(res['model'], res['feature_cols'], model_type)
st.write("---")
st.write("Catatan: Metode forecasting sederhana ini menggunakan fitur lag dari target dan model ML non-sequence (RandomForest atau Ridge). Untuk forecasting time-series yang lebih baik, gunakan model khusus time-series (ARIMA, Prophet, SARIMAX, LSTM, dsb.).")
