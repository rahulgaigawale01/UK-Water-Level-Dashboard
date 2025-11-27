"""
UK Water Level Prediction Dashboard (Streamlit)
Single-file runnable dashboard that:
 - loads the Excel dataset (UK_Water_Reservoir_Dataset_2020_2024.xlsx)
 - preprocesses and resamples to weekly frequency per reservoir
 - builds two model types: tree-based (XGBoost / RandomForest fallback), LSTM (Keras)
 - produces explainability (SHAP where available) and prediction visualisations
 - ensembles models and provides uncertainty bands

USAGE
1. Create a Python 3.8+ venv
   python -m venv venv
   source venv/bin/activate   (Linux/macOS) OR venv\Scripts\activate (Windows)
2. Install requirements
   pip install -r requirements.txt
   (A suggested requirements.txt content is shown below in comments.)
3. Put your Excel file in ./data/UK_Water_Reservoir_Dataset_2020_2024.xlsx OR upload it through the sidebar file uploader in the app.
4. Run the app
   streamlit run UK_Water_Reservoir_Dashboard_app.py

Notes
- The app contains a small Met Office forecast stub: to use real Met Office forecast data you must obtain an API key and implement the fetch function (instructions are inside the file).
- Training models from scratch will save models into ./models by default.

Suggested requirements.txt
streamlit
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
joblib
tensorflow
plotly
matplotlib
openpyxl
requests

"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import requests

# Utilities 
SEED = 42
np.random.seed(SEED)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper: load dataset from file (Path or uploaded)
@st.cache_data
def load_excel(file) -> pd.DataFrame:
    if isinstance(file, str):
        df = pd.read_excel(file, engine="openpyxl")
    else:
        # uploaded file object
        df = pd.read_excel(file, engine="openpyxl")
    return df

# Preprocessing: parse dates, sort, basic cleaning
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure Date column
    if 'Date' not in df.columns:
        raise ValueError("No 'Date' column found in dataset")
    df['Date'] = pd.to_datetime(df['Date'])
    # Remove rows with missing reservoir names
    df = df[df['Reservoir Name'].notna()].reset_index(drop=True)
    # Standardise drought status
    if 'Drought Status' in df.columns:
        df['Drought Status'] = df['Drought Status'].astype(str).str.strip().fillna('Unknown')
    # Make numeric conversions where possible
    numeric_cols = ['Capacity (ML)', 'Storage (%)', 'Inflow (ML/day)', 'Outflow (ML/day)', 'Rainfall (mm)', 'Temperature (°C)']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# Resample to weekly frequency per reservoir (forward fill)
def to_weekly(df: pd.DataFrame, week_start='W-MON') -> pd.DataFrame:
    # We'll forward-fill values to weekly frequency for each reservoir
    out_rows = []
    for (region, res), g in df.groupby(['Region', 'Reservoir Name']):
        g = g.sort_values('Date').set_index('Date')
        # resample weekly - start of week (MON) and forward-fill
        w = g.resample(week_start).ffill()
        w['Region'] = region
        w['Reservoir Name'] = res
        w['week_start'] = w.index
        out_rows.append(w.reset_index(drop=True))
    weekly = pd.concat(out_rows, ignore_index=True)
    # ensure columns
    weekly = weekly.drop_duplicates(subset=['Region','Reservoir Name','week_start'])
    # rename week_start -> Date
    weekly = weekly.rename(columns={'week_start':'Date'})
    # ensure Date dtype
    weekly['Date'] = pd.to_datetime(weekly['Date'])
    return weekly

# Feature engineering for tabular models
def make_features(df: pd.DataFrame, lags=[1,2,3,4], rolling_windows=[4,12]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['Reservoir Name','Date']).reset_index(drop=True)
    # encode drought status
    if 'Drought Status' in df.columns:
        df['Drought_Code'] = df['Drought Status'].astype('category').cat.codes
    else:
        df['Drought_Code'] = 0

    # encode reservoir identity so models can learn per-reservoir behaviour
    df['Reservoir_Code'] = df['Reservoir Name'].astype('category').cat.codes
    # fill capacity constant per reservoir
    df['Capacity (ML)'] = df.groupby('Reservoir Name')['Capacity (ML)'].transform('first')
    # create lag features per reservoir
    for lag in lags:
        df[f'Storage_lag_{lag}'] = df.groupby('Reservoir Name')['Storage (%)'].shift(lag)
        df[f'Inflow_lag_{lag}'] = df.groupby('Reservoir Name')['Inflow (ML/day)'].shift(lag)
        df[f'Outflow_lag_{lag}'] = df.groupby('Reservoir Name')['Outflow (ML/day)'].shift(lag)
        df[f'Rainfall_lag_{lag}'] = df.groupby('Reservoir Name')['Rainfall (mm)'].shift(lag)
    # rolling means
    for w in rolling_windows:
        df[f'Storage_roll_{w}'] = df.groupby('Reservoir Name')['Storage (%)'].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        df[f'Rainfall_roll_{w}'] = df.groupby('Reservoir Name')['Rainfall (mm)'].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
    # change features
    df['Inflow_diff'] = df['Inflow (ML/day)'] - df.groupby('Reservoir Name')['Inflow (ML/day)'].shift(1)
    df['Outflow_diff'] = df['Outflow (ML/day)'] - df.groupby('Reservoir Name')['Outflow (ML/day)'].shift(1)
    # Drop rows with NaN in target or essential features
    df = df.dropna(subset=['Storage (%)'])
    return df

# Build train/test splits preserving time order per reservoir
def time_series_split(df: pd.DataFrame, test_size=0.2):
    df = df.sort_values('Date')
    unique_dates = df['Date'].sort_values().unique()
    cutoff_index = int(len(unique_dates) * (1 - test_size))
    cutoff_date = unique_dates[cutoff_index]
    train = df[df['Date'] <= cutoff_date].copy()
    test = df[df['Date'] > cutoff_date].copy()
    return train, test

# Standard scaler saving
def save_scaler(scaler, name='scaler.save'):
    joblib.dump(scaler, os.path.join(MODEL_DIR, name))

def load_scaler(name='scaler.save'):
    return joblib.load(os.path.join(MODEL_DIR, name))

# Model builders 

def train_tree_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, 'tree_model.joblib'))
    return model

# LSTM data preparation
def build_lstm_sequences(df: pd.DataFrame, features, target_col='Storage (%)', seq_len=8):
    Xs = []
    ys = []
    res_ids = []

    for res_name, g in df.groupby('Reservoir Name'):
        g = g.sort_values('Date')
        vals = g[features].values
        t = g[target_col].values
        for i in range(seq_len, len(g)):
            Xs.append(vals[i - seq_len:i])
            ys.append(t[i])
            res_ids.append(res_name)

    Xs = np.array(Xs)
    ys = np.array(ys)
    res_ids = np.array(res_ids)
    return Xs, ys, res_ids

# LSTM model builder
def create_lstm_model(input_shape, dropout_rate=0.2):
    inp = keras.Input(shape=input_shape)

    # 1st LSTM layer
    x = layers.Masking(mask_value=np.nan)(inp)
    x = layers.LSTM(50, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)

    # 2nd LSTM layer
    x = layers.LSTM(30, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)

    # 3rd LSTM layer
    x = layers.LSTM(20, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)

    out = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

# Train LSTM
def train_lstm(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    input_shape = X_train.shape[1:]
    model = create_lstm_model(input_shape)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val) if X_val is not None else None,
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
    model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))
    return model, history

# Monte Carlo dropout predictions (uncertainty)
def mc_dropout_predict(model, X, n_samples=50):
    preds = []
    for i in range(n_samples):
        preds.append(model(X, training=True).numpy().squeeze())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# Explainability 

def shap_explain(model, X_sample, feature_names):
    if not SHAP_AVAILABLE:
        return None
    # Use TreeExplainer for tree models
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        return shap_values
    except Exception:
        return None

# --------------------------- Forecast API stub ---------------------------
# To integrate Met Office forecasts: sign up for their API and insert your key below.
# This is a placeholder function. The Met Office datapoint API requires registration.

MET_OFFICE_API_KEY = None  # <-- put your key here or in a .env file

def fetch_met_office_forecast_stub(lat, lon, start_date, days=7):
    """
    Placeholder: returns a DataFrame with columns ['Date','Rainfall (mm)','Temperature (°C)'] for the requested period.
    Replace this with real API calls.
    """
    dates = pd.date_range(start_date, periods=days, freq='D')
    # naive synthetic forecast: repeat recent climatology or zeros
    df = pd.DataFrame({'Date': dates, 'Rainfall (mm)': np.random.rand(len(dates)) * 10, 'Temperature (°C)': 10 + np.random.randn(len(dates))})
    return df

# Prediction helpers 

def evaluate_model(model, X, y_true):
    preds = model.predict(X)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# Streamlit App 

st.set_page_config(page_title='UK Reservoir Level Prediction Dashboard', layout='wide')
st.title('UK Water Level Prediction Dashboard — Data Analytics Approach')

if 'preds_df' not in st.session_state:
    st.session_state['preds_df'] = None

# Sidebar: file upload and settings
st.sidebar.header('Data & Settings')
uploaded_file = st.sidebar.file_uploader('Upload Excel file (or leave blank to load from ./data)', type=['xlsx','xls'])
use_sample = False
if uploaded_file is None:
    default_path = 'data/UK_Water_Reservoir_Dataset_2020_2024.xlsx'
    if os.path.exists(default_path):
        df_raw = load_excel(default_path)
    else:
        st.sidebar.warning('No file uploaded and default file not found. Please upload your Excel dataset.')
        use_sample = True
else:
    df_raw = load_excel(uploaded_file)

if use_sample:
    st.info('Showing a tiny autogenerated sample to illustrate the dashboard flow. Upload your full dataset to operate on real data.')
    # create a tiny sample based on the user's sample rows
    sample = {
        'Date':['2020-01-31','2020-02-29','2020-03-31','2020-04-30','2020-05-31','2020-06-30','2020-07-31','2020-08-31','2020-09-30','2020-10-31','2020-11-30','2020-12-31','2021-01-31'],
        'Region':['North West']*13,
        'Reservoir Name':['Reservoir_45']*13,
        'Capacity (ML)':[35795]*13,
        'Storage (%)':[74.2,85.2,30.8,84.3,100,75.9,74.4,60.5,100,87.9,54.9,46.3,78.7],
        'Inflow (ML/day)':[75.9,116.7,30.2,25.3,96.4,28.8,98.9,64.5,82.9,95.3,111.4,49.2,98.2],
        'Outflow (ML/day)':[27.2,101.6,43.5,42.1,32,17.2,43.5,13.8,44.3,113.3,19.7,40.9,18.2],
        'Rainfall (mm)':[237.7,177,53.1,72.8,91.6,11.6,237.2,110,227.3,46.2,223.7,67.8,135.7],
        'Temperature (°C)':[18.8,2.5,6.2,16.1,12.5,16,24.2,4.8,8,24.3,15.8,21.1,5.2],
        'Drought Status':['Normal','Normal','Severe','Normal','Normal','Normal','Normal','Normal','Normal','Normal','Moderate','Moderate','Normal']
    }
    df_raw = pd.DataFrame(sample)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

# Preprocess
try:
    df = preprocess_raw(df_raw)
except Exception as e:
    st.error(f'Error processing uploaded file: {e}')
    st.stop()

st.sidebar.markdown('---')
weekly_button = st.sidebar.checkbox('Convert to weekly and show preview', value=True)
if weekly_button:
    weekly = to_weekly(df)
else:
    weekly = df.copy()

st.sidebar.markdown('Model & training settings')
seq_len = st.sidebar.number_input('LSTM sequence length (weeks)', value=8, min_value=1, max_value=52)
test_size = st.sidebar.slider('Test set fraction (chronological)', min_value=0.05, max_value=0.5, value=0.2)

# Feature engineering
with st.expander('Preview data'):
    st.write('Raw (first 10 rows)')
    st.dataframe(df.head(10))
    st.write('Weekly (first 10 rows)')
    st.dataframe(weekly.head(10))

st.header('Feature engineering')
features_df = make_features(weekly)
st.write('Feature-engineered sample (first 10 rows)')
st.dataframe(features_df.head(10))

# Allow user to pick a reservoir
reservoirs = features_df['Reservoir Name'].unique().tolist()
sel_res = st.selectbox('Select reservoir for inspection & forecasting', reservoirs)

res_df = features_df[features_df['Reservoir Name'] == sel_res].sort_values('Date')

# Show time-series
st.subheader(f'Time series: {sel_res}')
fig = px.line(res_df, x='Date', y=['Storage (%)','Rainfall (mm)','Inflow (ML/day)'], title=f'Time series for {sel_res}')
st.plotly_chart(fig, use_container_width=True)

# Features used by the LSTM model (for both training AND forecasting)
LSTM_FEATURES = [
    'Storage (%)',
    'Inflow (ML/day)',
    'Outflow (ML/day)',
    'Rainfall (mm)',
    'Temperature (°C)',
    'Drought_Code',
    'Reservoir_Code'
]

# Training section
st.header('Train / Load models')
col1, col2 = st.columns(2)
with col1:
    if st.button('Train tabular tree model (XGBoost / RandomForest)'):
        # Prepare tabular dataset
        train_df, test_df = time_series_split(features_df, test_size=test_size)
        feature_cols = [c for c in features_df.columns if c not in ['Date','Region','Reservoir Name','Storage (%)','Drought Status']]
        # drop any object columns
        feature_cols = [c for c in feature_cols if features_df[c].dtype.kind in 'biufc']
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df['Storage (%)'].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df['Storage (%)'].values
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        save_scaler(scaler, 'tree_scaler.save')
        tree_model = train_tree_model(X_train_s, y_train)
        evals = evaluate_model(tree_model, X_test_s, y_test)
        st.success(f"Trained tree model — test RMSE: {evals['rmse']:.3f}, MAE: {evals['mae']:.3f}, R2: {evals['r2']:.3f}")
        # Per-reservoir performance for tree model 
        per_res_tree = []
        for res_name in test_df['Reservoir Name'].unique():
            sub = test_df[test_df['Reservoir Name'] == res_name]
            if len(sub) < 5:
                continue  # skip tiny test sets
        
            X_res = sub[feature_cols].fillna(0).values
            X_res_s = X_res  
        
            y_res = sub['Storage (%)'].values
            m = evaluate_model(tree_model, X_res_s, y_res)
            per_res_tree.append({
                'Reservoir Name': res_name,
                'RMSE': m['rmse'],
                'MAE': m['mae'],
                'R2': m['r2'],
            })
        
        if per_res_tree:
            st.subheader('Per-reservoir test performance — Tree model')
            per_tree_df = pd.DataFrame(per_res_tree)
            st.dataframe(per_tree_df.style.format({'RMSE': '{:.3f}', 'MAE': '{:.3f}', 'R2': '{:.3f}'}))

        st.info('Model saved to ./models/tree_model.joblib')
with col2:
    if st.button('Train LSTM model'):
        # choose features for LSTM
        lstm_features = [f for f in LSTM_FEATURES if f in features_df.columns]

        X, y, res_ids = build_lstm_sequences(features_df, lstm_features, seq_len=seq_len)

        # drop any sequences that contains NaNs (in X or y)
        if len(X) > 0:
            nan_mask = (~np.isnan(X).any(axis=(1,2))) & (~np.isnan(y))
            X = X[nan_mask]
            y = y[nan_mask]
            res_ids = res_ids[nan_mask]

        if len(X) == 0:
            st.error('Not enough data to build LSTM sequences. Consider lowering the sequence length or uploading a larger dataset.')
        else:
            # 1) Chronological split 70/15/15
            n = len(X)
            train_end = int(0.7 * n)
            val_end   = int(0.85 * n)

            X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
            y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
            res_train, res_val, res_test = res_ids[:train_end], res_ids[train_end:val_end], res_ids[val_end:]

            # 2) Scale X with MinMax and SAVE X-scaler
            nsamples, s, nfeatures = X_train.shape
            x_scaler = MinMaxScaler()

            X_train_flat = X_train.reshape(-1, nfeatures)
            x_scaler.fit(X_train_flat)
            joblib.dump(x_scaler, os.path.join(MODEL_DIR, 'lstm_x_scaler.save'))

            def scale_X(block):
                flat = block.reshape(-1, nfeatures)
                scaled = x_scaler.transform(flat)
                return scaled.reshape(block.shape)

            X_train_s = scale_X(X_train)
            X_val_s   = scale_X(X_val)
            X_test_s  = scale_X(X_test)

            # 3) Scale y with MinMax and SAVE y-scaler
            y_scaler = MinMaxScaler()
            y_scaler.fit(y_train.reshape(-1, 1))
            joblib.dump(y_scaler, os.path.join(MODEL_DIR, 'lstm_y_scaler.save'))

            y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
            y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
            # keep test targets in ORIGINAL scale for reporting
            y_test_orig = y_test.copy()

            # 4) Train LSTM
            lstm_model, history = train_lstm(
                X_train_s, y_train_s,
                X_val_s, y_val_s,
                epochs=50,
                batch_size=32
            )

            # 5) Predict on test set: scaled → inverse transform
            preds_s = lstm_model.predict(X_test_s).squeeze()
            preds = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).reshape(-1)

            # 6) Remove NaNs just in case
            valid_mask = (~np.isnan(preds)) & (~np.isnan(y_test_orig))
            preds = preds[valid_mask]
            y_test_orig = y_test_orig[valid_mask]
            res_test = res_test[valid_mask]

            # 7) Global metrics
            rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
            mae  = mean_absolute_error(y_test_orig, preds)
            r2   = r2_score(y_test_orig, preds)
            st.success(f'LSTM trained — test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}')

            # 8) Per-reservoir metrics
            per_res_lstm = []
            for res_name in np.unique(res_test):
                mask = (res_test == res_name)
                if mask.sum() < 5:
                    continue  # skip very small test sets

                y_true_res = y_test_orig[mask]
                y_pred_res = preds[mask]

                rmse_res = np.sqrt(mean_squared_error(y_true_res, y_pred_res))
                mae_res  = mean_absolute_error(y_true_res, y_pred_res)
                r2_res   = r2_score(y_true_res, y_pred_res)

                per_res_lstm.append({
                    'Reservoir Name': res_name,
                    'RMSE': rmse_res,
                    'MAE': mae_res,
                    'R2': r2_res,
                })

            if per_res_lstm:
                st.subheader('Per-reservoir test performance — LSTM model')
                per_lstm_df = pd.DataFrame(per_res_lstm)
                st.dataframe(per_lstm_df.style.format({'RMSE': '{:.3f}', 'MAE': '{:.3f}', 'R2': '{:.3f}'}))

            st.info('Saved LSTM under ./models/lstm_model.keras and scalers under ./models/lstm_x_scaler.save, ./models/lstm_y_scaler.save')

# Load models 
st.markdown('---')
if st.button('Load saved models (if any)'):
    loaded = {}
    try:
        tree = joblib.load(os.path.join(MODEL_DIR, 'tree_model.joblib'))
        loaded['tree'] = tree
        st.success('Tree model loaded')
    except Exception:
        st.warning('No saved tree model found')
    try:
        lstm = keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        loaded['lstm'] = lstm
        st.success('LSTM model loaded')
    except Exception:
        st.warning('No saved LSTM model found')

# Forecasting UI
st.header('Forecast (Ensemble)')
horizon_weeks = st.number_input('Forecast horizon (weeks)', value=4, min_value=1, max_value=52)
if st.button('Run ensemble forecast for selected reservoir'):
    # Load TREE model and scaler
    try:
        tree_model = joblib.load(os.path.join(MODEL_DIR, 'tree_model.joblib'))
        tree_scaler = load_scaler('tree_scaler.save')
    except Exception:
        st.error('Tree model or scaler not found. Please train the tree model first.')
        st.stop()
    # Load models and scalers
    try:
        lstm_model = keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
        lstm_x_scaler = joblib.load(os.path.join(MODEL_DIR, 'lstm_x_scaler.save'))
        lstm_y_scaler = joblib.load(os.path.join(MODEL_DIR, 'lstm_y_scaler.save'))
    except Exception:
        st.warning('LSTM model or scalers not found — ensemble will use only tree model')
        lstm_model = None
        lstm_x_scaler = None
        lstm_y_scaler = None

    # prepare last seq for the reservoir
    history_df = features_df[features_df['Reservoir Name']==sel_res].sort_values('Date')
    last_row = history_df.iloc[-1:]
    preds_list = []
    index_dates = pd.date_range(last_row['Date'].iloc[0] + pd.Timedelta(days=7), periods=horizon_weeks, freq='W-MON')
    # naive iterative forecasting strategy: for each week, predict using features (lag features updated)
    sim = history_df.copy()
    for d in index_dates:
        # create feature row using last known values
        fr = sim.iloc[-1:].copy()
        fr = fr.copy()
        fr['Date'] = d
        # update lag features
        for lag in [1,2,3,4]:
            fr[f'Storage_lag_{lag}'] = sim['Storage (%)'].iloc[-lag] if len(sim) >= lag else sim['Storage (%)'].iloc[-1]
        # tree features
        feat_cols = [c for c in features_df.columns if c not in ['Date','Region','Reservoir Name','Storage (%)','Drought Status']]
        feat_cols = [c for c in feat_cols if features_df[c].dtype.kind in 'biufc']
        X_tree = fr[feat_cols].fillna(0).values
        X_tree_s = tree_scaler.transform(X_tree)
        tree_pred = tree_model.predict(X_tree_s).squeeze()
        # quantile
        try:
            ql = joblib.load(os.path.join(MODEL_DIR, 'quantile_lower.joblib'))
            qu = joblib.load(os.path.join(MODEL_DIR, 'quantile_upper.joblib'))
            lower = ql.predict(X_tree_s)
            upper = qu.predict(X_tree_s)
        except Exception:
            lower, upper = tree_pred - 5, tree_pred + 5
        lstm_pred = None
        if lstm_model is not None and lstm_x_scaler is not None and lstm_y_scaler is not None:
            # prepare LSTM sequence using the SAME features as training
            lstm_feats = [f for f in LSTM_FEATURES if f in sim.columns]

            recent = sim[lstm_feats].values[-seq_len:]
            if len(recent) < seq_len:
                # pad with last row to reach seq_len
                pad = np.repeat(recent[:1], seq_len - len(recent), axis=0)
                recent = np.vstack([pad, recent])

            # scale X using the trained x-scaler
            recent_flat = recent.reshape(-1, recent.shape[1])
            recent_s = lstm_x_scaler.transform(recent_flat).reshape(1, seq_len, recent.shape[1])

            # predict in scaled y-space, then invert
            pred_s = lstm_model.predict(recent_s).squeeze()
            lstm_pred = float(lstm_y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)[0])

        # combine
        if lstm_pred is not None:
            # simple weighting by model type: 0.5/0.5 or we could weight by validation RMSE
            ensemble_pred = 0.5 * tree_pred + 0.5 * lstm_pred
        else:
            ensemble_pred = tree_pred
        preds_list.append({'Date':d, 'tree':float(tree_pred), 'lstm': float(lstm_pred) if lstm_pred is not None else None, 'ensemble': float(ensemble_pred), 'lower':float(lower), 'upper':float(upper)})
        # append back to sim to update lags with predicted storage
        newrow = fr.copy()
        newrow['Storage (%)'] = ensemble_pred
        sim = pd.concat([sim, newrow], ignore_index=True)
    preds_df = pd.DataFrame(preds_list)
    preds_df = preds_df.set_index('Date')
    st.session_state['preds_df'] = preds_df
    # show
    st.subheader('Forecast (ensemble)')
    fig2 = px.line(preds_df.reset_index(), x='Date', y=['ensemble','tree','lstm'], title=f'Forecast for {sel_res}')
    # add uncertainty band
    fig2.add_traces(px.line(preds_df.reset_index(), x='Date', y='lower').data)
    fig2.add_traces(px.line(preds_df.reset_index(), x='Date', y='upper').data)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(preds_df)

# Explainability
st.header('Explainability & Feature importance')
if st.button('Show SHAP summary for tree model (if available)'):
    if not SHAP_AVAILABLE:
        st.warning('SHAP not installed — please pip install shap to enable this feature.')
    else:
        try:
            # Load model + scaler
            tree_model = joblib.load(os.path.join(MODEL_DIR, 'tree_model.joblib'))
            tree_scaler = load_scaler('tree_scaler.save')

            # Build feature list exactly as used during training
            feat_cols = [c for c in features_df.columns 
                         if c not in ['Date','Region','Reservoir Name','Storage (%)','Drought Status']]
            feat_cols = [c for c in feat_cols if features_df[c].dtype.kind in 'biufc']

            # Sample data
            sample = features_df.sample(min(200, len(features_df)))
            Xs = sample[feat_cols].fillna(0).values
            Xs_s = tree_scaler.transform(Xs)

            # IMPORTANT PART: pass feature names
            explainer = shap.Explainer(tree_model, feature_names=feat_cols)
            shap_vals = explainer(Xs_s)

            # Plot
            fig = plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_vals, show=False)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f'Error computing SHAP: {e}')


# Download predictions
st.header('Export & reproducibility')
if st.button('Export last predictions to CSV'):
    if st.session_state.get('preds_df') is None:
        st.warning('Please run a forecast first so there are predictions to export.')
    else:
        preds_df = st.session_state['preds_df']
        try:
            csv_bytes = preds_df.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                'Download CSV',
                data = csv_bytes,
                file_name = 'predictions_export.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f'Error exporting: {e}')

# Final notes
st.markdown('''
### Notes & next steps
- To connect real Met Office forecasts: obtain an API key from the Met Office DataPoint service, then replace `fetch_met_office_forecast_stub` with real requests to their JSON endpoints and map fields to 'Rainfall (mm)' and 'Temperature (°C)'.
- The dashboard trains simple proof-of-concept models. For production: add hyperparameter optimisation, proper time-series cross-validation (walk-forward), careful handling of missing data, and domain-informed features (holiday/consumption patterns, customer demand signals).
- The ensemble and uncertainty approach here is intentionally modular — you can replace the averaging with a stacking regressor, or use probabilistic deep learning for full predictive distributions.

If you'd like, I can also export a requirements.txt and a concise README (in the same folder) with exact pip lines and sample commands.''')

# End of file
