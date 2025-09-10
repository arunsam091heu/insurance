import os
import joblib
import pandas as pd
from fastapi import FastAPI
from typing import Dict, Any, List, Optional
from pydantic import RootModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----- Config -----
CSV_PATH = os.getenv("CSV_PATH", "fraud_oracle.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
TARGET_COL = "FraudFound_P"

# How to load the model:
#   MODEL_SOURCE = "local"  -> load joblib from MODEL_PATH (default)
#   MODEL_SOURCE = "mlflow" -> load from MLflow model URI (set MLFLOW_MODEL_URI)
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local").lower()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # optional, e.g. http://127.0.0.1:5000
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")        # e.g. runs:/<RUN_ID>/model or models:/fraud-detection/Production

# ----- App -----
app = FastAPI(
    title="Insurance Fraud Model API (MLflow-enabled)",
    description="Serves predictions from local or MLflow model with the same preprocessing.",
    version="4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ---------- Encoders / preprocessing ----------
day_map = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
bool_map = {"No":0, "Yes":1}
agent_map = {"External":0, "Internal":1}
veh_map = {"Sport":0, "Sedan":1, "Utility":2}
marital_map = {"Widow":0, "Single":1, "Married":2, "Divorced":3}
sup_map = {"none":0, "1 to 2":1.5, "3 to 5":4, "more than 5":6}
base_policy_map = {"Liability":0, "Collision":1, "All Perils":2}
addr_change_map = {"1 year":1, "no change":0, "4 to 8 years":6, "2 to 3 years":2.5, "under 6 months":0.3}

def sex_to_int(x): return 1 if str(x).strip().lower()=="male" else 0
def area_to_int(x): return 1 if str(x).strip().lower()=="urban" else 0
def fault_to_int(x): return 1 if str(x).strip().lower()=="policy holder" else 0

DROP_COLS_ANYWAY = ["Combined Name","PolicyType","PolicyNumber","Unnamed: 33","Unnamed: 34","Statement"]

# global Make encoder
make_map: Dict[str, int] = {}
def _build_make_map(series: pd.Series) -> Dict[str, int]:
    labels = sorted(set(str(v) for v in series.dropna().tolist()))
    return {lab: i for i, lab in enumerate(labels)}
def _encode_make(col: pd.Series) -> pd.Series:
    return col.apply(lambda v: make_map.get(str(v), -1)).astype(int)

def parse_range(val: str) -> float:
    if pd.isna(val):
        return 0.0
    val = str(val).strip().lower()
    if "less than" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] - 2 if nums else 0
    if "more than" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] + 2 if nums else 0
    if "to" in val:
        parts = [float(s) for s in val.replace("years","").replace("vehicle","").split() if s.replace('.','',1).isdigit()]
        if len(parts) == 2:
            return sum(parts)/2
    if val.replace('.','',1).isdigit():
        return float(val)
    if "vehicle" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return float(nums[0]) if nums else 1.0
    if "new" in val:
        return 0.5
    return 0.0

def engineer_policy_relation(row: pd.Series) -> int:
    policy_type = str(row.get("PolicyType", "")).strip()
    combined = f"{row.get('VehicleCategory','')} - {row.get('BasePolicy','')}"
    return 1 if policy_type == combined else 0

def preprocess_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Policy realtion with Base" not in df.columns:
        df["Combined Name"] = df.get("VehicleCategory","").astype(str) + " - " + df.get("BasePolicy","").astype(str)
        df["Policy realtion with Base"] = df.apply(engineer_policy_relation, axis=1)

    if "Sex" in df.columns: df["Sex"] = df["Sex"].apply(sex_to_int).astype(int)
    if "AccidentArea" in df.columns: df["AccidentArea"] = df["AccidentArea"].apply(area_to_int).astype(int)
    if "Fault" in df.columns: df["Fault"] = df["Fault"].apply(fault_to_int).astype(int)
    if "PoliceReportFiled" in df.columns: df["PoliceReportFiled"] = df["PoliceReportFiled"].map(bool_map).astype(int)
    if "WitnessPresent" in df.columns: df["WitnessPresent"] = df["WitnessPresent"].map(bool_map).astype(int)
    if "AgentType" in df.columns: df["AgentType"] = df["AgentType"].map(agent_map).astype(int)
    if "VehicleCategory" in df.columns: df["VehicleCategory"] = df["VehicleCategory"].map(veh_map).astype(int)
    if "MaritalStatus" in df.columns: df["MaritalStatus"] = df["MaritalStatus"].map(marital_map).astype(int)
    if "NumberOfSuppliments" in df.columns: df["NumberOfSuppliments"] = df["NumberOfSuppliments"].map(sup_map).astype(float)
    if "BasePolicy" in df.columns: df["BasePolicy"] = df["BasePolicy"].map(base_policy_map).astype(int)

    for col in ["DayOfWeek","DayOfWeekClaimed"]:
        if col in df.columns:
            df[col] = df[col].replace(day_map).replace("0", 0).astype(int)
    for col in ["Month","MonthClaimed"]:
        if col in df.columns:
            df[col] = df[col].replace(month_map).astype(int)
    if "AddressChange_Claim" in df.columns:
        df["AddressChange_Claim"] = df["AddressChange_Claim"].replace(addr_change_map).astype(float)

    if "Make" in df.columns:
        df["Make"] = _encode_make(df["Make"].astype(str))

    if "VehiclePrice" in df.columns: df["VehiclePrice"] = df["VehiclePrice"].apply(parse_range).astype(float)
    if "AgeOfVehicle" in df.columns: df["AgeOfVehicle"] = df["AgeOfVehicle"].apply(parse_range).astype(float)
    if "NumberOfCars" in df.columns: df["NumberOfCars"] = df["NumberOfCars"].apply(parse_range).astype(float)
    if "AgeOfPolicyHolder" in df.columns: df["AgeOfPolicyHolder"] = df["AgeOfPolicyHolder"].apply(parse_range).astype(float)

    for c in DROP_COLS_ANYWAY:
        if c in df.columns: df = df.drop(columns=[c])
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed:"): df = df.drop(columns=[c])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(parse_range).astype(float)

    return df

# ---------- Training (fallback) ----------
model: Optional[RandomForestClassifier] = None
feature_cols: Optional[List[str]] = None

def _train_from_csv() -> None:
    global model, feature_cols, make_map
    df_raw = pd.read_csv(CSV_PATH)
    if "Make" in df_raw.columns:
        make_map = _build_make_map(df_raw["Make"])
    df_prep = preprocess_raw_df(df_raw)
    if TARGET_COL not in df_prep.columns:
        raise RuntimeError(f"Target column '{TARGET_COL}' not found.")
    X = df_prep.drop(columns=[TARGET_COL])
    y = df_prep[TARGET_COL].astype(int)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
    clf = RandomForestClassifier(random_state=0, class_weight="balanced")
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    model = clf
    feature_cols = list(X.columns)

def _load_local_or_train():
    global model, feature_cols, make_map
    if os.path.exists(MODEL_PATH):
        try:
            model_obj = joblib.load(MODEL_PATH)
            if not hasattr(model_obj, "predict"):
                raise ValueError("Invalid model file")
            model = model_obj
        except Exception as e:
            print(f"[WARN] Failed to load local model: {e}. Retraining...")
            _train_from_csv()
    else:
        print("[INFO] model.pkl not found; training from CSV...")
        _train_from_csv()

    df_raw = pd.read_csv(CSV_PATH)
    if "Make" in df_raw.columns:
        make_map = _build_make_map(df_raw["Make"])
    df_prep = preprocess_raw_df(df_raw)
    X = df_prep.drop(columns=[TARGET_COL]) if TARGET_COL in df_prep.columns else df_prep
    feature_cols = list(X.columns)

def _load_from_mlflow():
    global model, feature_cols, make_map
    import mlflow
    import mlflow.sklearn

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not MLFLOW_MODEL_URI:
        raise RuntimeError("MODEL_SOURCE=mlflow but MLFLOW_MODEL_URI is not set.")

    print(f"[INFO] Loading model from MLflow URI: {MLFLOW_MODEL_URI}")
    model_obj = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    if not hasattr(model_obj, "predict"):
        raise RuntimeError("Loaded MLflow object is not a sklearn model.")
    model = model_obj

    df_raw = pd.read_csv(CSV_PATH)
    if "Make" in df_raw.columns:
        make_map = _build_make_map(df_raw["Make"])
    df_prep = preprocess_raw_df(df_raw)
    X = df_prep.drop(columns=[TARGET_COL]) if TARGET_COL in df_prep.columns else df_prep
    feature_cols = list(X.columns)

@app.on_event("startup")
def _startup():
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"CSV not found at {CSV_PATH}")
    if MODEL_SOURCE == "mlflow":
        _load_from_mlflow()
    else:
        _load_local_or_train()

# ---------- Request models ----------
class RawRow(RootModel[Dict[str, Any]]): pass
class NumericRow(RootModel[Dict[str, Any]]): pass

# ---------- Endpoints ----------
@app.get("/")
def root():
    src = "MLflow" if MODEL_SOURCE == "mlflow" else "Local model.pkl (auto-train fallback)"
    return {"status": "ok", "model_source": src}

@app.get("/model-info")
def model_info():
    return {
        "model_source": MODEL_SOURCE,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_model_uri": MLFLOW_MODEL_URI,
        "features": feature_cols
    }

@app.post("/retrain")
def retrain():
    _train_from_csv()
    return {"status": "retrained", "num_features": len(feature_cols), "features": feature_cols}

@app.post("/predict-raw")
def predict_raw(rows: List[RawRow]):
    assert model is not None and feature_cols is not None
    df_input_raw = pd.DataFrame([r.root for r in rows])
    df_proc = preprocess_raw_df(df_input_raw)
    if TARGET_COL in df_proc.columns:
        df_proc = df_proc.drop(columns=[TARGET_COL])
    for c in feature_cols:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[feature_cols]
    preds = model.predict(df_proc).tolist()
    probs = None
    try:
        probs = model.predict_proba(df_proc)[:, 1].tolist()
    except Exception:
        pass
    return {"predictions": preds, "probabilities_for_class_1": probs}

@app.post("/predict")
def predict_numeric(rows: List[NumericRow]):
    assert model is not None and feature_cols is not None
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
    preds = model.predict(df).tolist()
    probs = None
    try:
        probs = model.predict_proba(df)[:, 1].tolist()
    except Exception:
        pass

    if preds[0] ==1:
        value= "Fraud"
    else:
        value="Not Fraud"
    return {"predictions": value, "probabilities_for_Fraud": probs}
