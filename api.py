import json, logging, os
from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from typing import List, Dict, Any, Optional
import joblib, pandas as pd

from preprocess import preprocess_raw_df  # shared

log = logging.getLogger("fraud-api")
logging.basicConfig(level=logging.INFO)

CSV_PATH = os.getenv("CSV_PATH", "fraud_oracle.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
TARGET_COL = os.getenv("TARGET_COL", "FraudFound_P")
FEATURE_SCHEMA_PATH = os.getenv("FEATURE_SCHEMA_PATH", "feature_cols.json")  # optional

app = FastAPI(
    title="Insurance Fraud Model API",
    description="Loads a pre-trained model.pkl and serves predictions.",
    version="4.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

model: Optional[any] = None
feature_cols: Optional[list] = None
startup_error: Optional[str] = None  # remember why we degraded

def _load_feature_cols_from_csv() -> list:
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"CSV not found at {CSV_PATH} to infer feature columns. "
                           f"Ship {CSV_PATH} or set FEATURE_SCHEMA_PATH to a JSON with feature list.")
    df_raw = pd.read_csv(CSV_PATH)
    df_prep = preprocess_raw_df(df_raw)
    X = df_prep.drop(columns=[TARGET_COL]) if TARGET_COL in df_prep.columns else df_prep
    return list(X.columns)

def _load_feature_cols() -> list:
    # Prefer explicit schema if present (no need to ship CSV to prod)
    if os.path.exists(FEATURE_SCHEMA_PATH):
        with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise RuntimeError(f"Bad schema in {FEATURE_SCHEMA_PATH}: expected JSON list[str].")
        return cols
    # Fallback: infer from CSV
    return _load_feature_cols_from_csv()

def load_model_and_schema():
    global model, feature_cols
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}.")
    m = joblib.load(MODEL_PATH)
    if not hasattr(m, "predict"):
        raise RuntimeError("Loaded object is not a valid model (no predict()).")
    # ok
    cols = _load_feature_cols()
    model = m
    feature_cols = cols
    log.info("Model and feature schema loaded. num_features=%d", len(feature_cols))

@app.on_event("startup")
def _startup():
    global startup_error
    try:
        load_model_and_schema()
        startup_error = None
    except Exception as e:
        import logging
        startup_error = f"{type(e).__name__}: {e}"
        logging.exception("Startup load failed: %s", startup_error)


class RawRow(RootModel[Dict[str, Any]]): 
    pass

class NumericRow(RootModel[Dict[str, Any]]): 
    pass

@app.get("/")
def root():
    return {"status": "ok", "message": "Use /predict-raw (strings) or /predict (numeric)."}

@app.get("/health")
def health():
    ok = (model is not None) and (feature_cols is not None)
    return {
        "status": "healthy" if ok else "degraded",
        "num_features": len(feature_cols) if feature_cols else 0,
        "error": startup_error
    }

@app.post("/predict-raw")
def predict_raw(rows: List[RawRow]):
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {startup_error}")
    df_input_raw = pd.DataFrame([r.root for r in rows])
    df_proc = preprocess_raw_df(df_input_raw)
    if TARGET_COL in df_proc.columns:
        df_proc = df_proc.drop(columns=[TARGET_COL])
    for c in feature_cols:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[feature_cols]
    preds = model.predict(df_proc)
    try:
        probs = model.predict_proba(df_proc)[:, 1].tolist()
    except Exception:
        probs = None
    return {"predictions": preds.tolist(), "probabilities_for_class_1": probs}

@app.post("/predict")
def predict_numeric(rows: List[NumericRow]):
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {startup_error}")
    df = pd.DataFrame([r.root for r in rows])
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed:"):
            df = df.drop(columns=[c])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols]
    preds = model.predict(df)
    try:
        probs = model.predict_proba(df)[:, 1].tolist()
    except Exception:
        probs = None
    return {"predictions": preds.tolist(), "probabilities_for_class_1": probs}

