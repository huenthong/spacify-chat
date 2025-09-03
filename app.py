import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# -------------------------
# App & Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("belive-alps")
app = Flask(__name__)
CORS(app)

# -------------------------
# Globals
# -------------------------
model = None
feature_names = None
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.dirname(__file__))

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_rf_model.pkl")
FEATURES_FILENAME = os.environ.get("FEATURES_FILENAME", "feature_names.pkl")

# -------------------------
# Load artifacts on import
# -------------------------
def load_model_artifacts():
    global model, feature_names
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    features_path = os.path.join(ARTIFACT_DIR, FEATURES_FILENAME)
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded model: {type(model).__name__} from {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        model = None

    try:
        feature_names = joblib.load(features_path)
        if isinstance(feature_names, (list, np.ndarray, pd.Index)):
            feature_names = list(feature_names)
        logger.info(f"✅ Loaded feature_names: {len(feature_names)} from {features_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load feature_names from {features_path}: {e}")
        feature_names = None

load_model_artifacts()

# -------------------------
# Feature prep
# -------------------------
# ---------- Normalizers ----------
def _to_float(x, default=0.0):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if not s: return default
        return float(s)
    except Exception:
        return default

def normalize_pax(v) -> int:
    """Accepts 1/2/3 numeric OR '1 person'/'2 people'/'More than 2' strings."""
    if isinstance(v, (int, float)):
        n = int(v)
        return 1 if n <= 1 else 2 if n == 2 else 3
    if isinstance(v, str):
        t = v.strip().lower()
        if t.startswith("1"): return 1
        if t.startswith("2"): return 2
        if "more" in t or "3" in t: return 3
    return 1

_LEAD_MAP = {
    "facebook":"Facebook",
    "instagram":"Instagram",
    "whatsapp":"WhatsApp",
    "google":"Google_Search",
    "google_search":"Google_Search",
    "google_ads":"Google_Ads",
    "website":"Website",
    "portal":"Website",     # iProperty/PropertyGuru -> treat as Website
    "referral":"Referral",
    "walk_in":"Walk_In",
    "walkin":"Walk_In",
    "email":"Email",
    "phone":"Phone_Call",
    "phone_call":"Phone_Call",
    "other":"Unknown",
    "unknown":"Unknown",
}

def normalize_lead_source(s: str) -> str:
    if not s: return "Website"
    key = str(s).strip().lower()
    return _LEAD_MAP.get(key, "Unknown")

def normalize_nationality(nationality: str = None,
                          is_malaysian: bool = None,
                          detail: str = "") -> str:
    """
    Returns a country label suitable for one-hots used in training.
    Harmonizes 'Malaysia'/'Malaysian' and accepts an 'Others' detail.
    """
    if is_malaysian is True:
        return "Malaysia"
    if nationality:
        n = nationality.strip()
        if n.lower() in ("malaysia", "malaysian"):
            return "Malaysia"
        if n == "Others" and detail:
            return detail.title()
        return n
    return "Other"
                              
def prepare_features(form_data):
    """
    Build a single-row DataFrame aligned to your training feature space.

    Accepts payloads from index_latest.html buildFeaturePayload(...) directly:
      area, property, budget_num, pax, days_to_move_in, tenancy_months,
      user_type, room_type, bedroom_type, lead_source, has_car, need_parking,
      is_malaysian, has_contact, workplace, workplace_hot, is_business_hours,
      is_weekend, raw_contact

    Also accepts the older names:
      budget, pax("1 person"/"2 people"/"More than 2"), movein (YYYY-MM-DD),
      car("Yes"/"No"), parking("Yes"/"No"/"Unknown"), nationality, gender, etc.
    """
    current_date = datetime.now()
    feats = {}

    # ------- Core numeric -------
    # Budget: prefer budget_num from new UI, else fallback to budget
    budget = _to_float(form_data.get("budget_num",
                    form_data.get("budget", 800.0)), 800.0)
    feats["Budget"] = budget
    feats["Rental Proposed"] = budget

    # Pax: accept number 1/2/3 or string "1 person" etc.
    feats["No of Pax"] = normalize_pax(form_data.get("pax", 1))

    # Contact time
    feats["contact_hour"]  = current_date.hour
    feats["contact_month"] = current_date.month

    # ------- Engineered: Budget buckets -------
    if budget == 0:
        feats["Budget_Category_Encoded"] = 0
    elif budget < 500:
        feats["Budget_Category_Encoded"] = 1
    elif budget < 1000:
        feats["Budget_Category_Encoded"] = 2
    elif budget < 1500:
        feats["Budget_Category_Encoded"] = 3
    else:
        feats["Budget_Category_Encoded"] = 4

    # ------- Engineered: Move urgency (from days_to_move_in OR dates) -------
    feats["Move_Urgency_Encoded"] = 0
    dd = None

    # (a) if client already sent computed days_to_move_in
    if form_data.get("days_to_move_in") is not None:
        try:
            dd = int(float(form_data["days_to_move_in"]))
        except Exception:
            dd = None

    # (b) else parse date from new key move_in_date or old key movein
    if dd is None:
        date_str = form_data.get("move_in_date") or form_data.get("movein")
        if date_str:
            try:
                move_dt = datetime.strptime(date_str, "%Y-%m-%d")
                dd = (move_dt - current_date).days
            except Exception:
                dd = None

    if dd is not None:
        if dd <= 30:  feats["Move_Urgency_Encoded"] = 1
        elif dd <= 90: feats["Move_Urgency_Encoded"] = 2
        else:          feats["Move_Urgency_Encoded"] = 3

    # ------- Temporal flags: allow client overrides -------
    is_biz = form_data.get("is_business_hours")
    is_wend = form_data.get("is_weekend")

    if isinstance(is_biz, bool):
        feats["Is_Business_Hours"] = 1 if is_biz else 0
    else:
        feats["Is_Business_Hours"] = 1 if 9 <= current_date.hour <= 17 else 0

    if isinstance(is_wend, bool):
        feats["Is_Weekend"] = 1 if is_wend else 0
    else:
        feats["Is_Weekend"] = 1 if current_date.weekday() >= 5 else 0

    # ------- Lead Source (one-hots your RF used) -------
    lead_categories = [
        "Facebook","Google_Search","Google_Ads","Instagram","WhatsApp",
        "Website","Referral","Walk_In","Email","Phone_Call","Unknown"
    ]
    incoming_lead = form_data.get("lead_source") or form_data.get("leadSource") or "Website"
    cur_source = normalize_lead_source(incoming_lead)
    for c in lead_categories:
        feats[f"Lead_Source_Standard_{c}"] = 1 if c == cur_source else 0

    # ------- Gender (optional in UI; safe default) -------
    gender_categories = ["Male","Female","Mixed","Unknown"]
    cur_gender = form_data.get("gender", "Unknown")
    if cur_gender not in gender_categories:
        cur_gender = "Unknown"
    for c in gender_categories:
        feats[f"Gender_Clean_{c}"] = 1 if c == cur_gender else 0

    # ------- Optional: binary flags the model may have seen -------
    # These are safe; if feature_names doesn't include them they’ll be dropped later.
    # Car / Parking
    has_car = form_data.get("has_car")
    if isinstance(has_car, bool):
        feats["Transportation_Car"] = 1 if has_car else 0
    else:
        car_str = str(form_data.get("car", "")).strip().lower()
        feats["Transportation_Car"] = 1 if car_str == "yes" else 0

    need_parking = form_data.get("need_parking")
    if isinstance(need_parking, bool):
        feats["Parking_Needed"] = 1 if need_parking else 0
    else:
        park_str = str(form_data.get("parking", "")).strip().lower()
        feats["Parking_Needed"] = 1 if park_str == "yes" else 0

    # Nationality (normalize Malaysia/Malaysian/Others+detail)
    nat = normalize_nationality(
        form_data.get("nationality"),
        form_data.get("is_malaysian"),
        form_data.get("nationality_detail", "")
    )
    feats["Is_Malaysian"] = 1 if nat == "Malaysia" else 0

    # Contact present
    if "has_contact" in form_data:
        feats["Has_Contact"] = 1 if bool(form_data.get("has_contact")) else 0

    # Workplace hot spot
    if "workplace_hot" in form_data:
        feats["Workplace_Hot"] = 1 if bool(form_data.get("workplace_hot")) else 0

    # (Optional) tenancy months could be useful if in training
    if "tenancy_months" in form_data:
        feats["Tenancy_Months"] = _to_float(form_data.get("tenancy_months"), 0)

    # -------- Build DataFrame and align to training feature order --------
    df = pd.DataFrame([feats])

    if feature_names:
        # add any missing columns with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        # drop any unexpected columns
        extra = [c for c in df.columns if c not in feature_names]
        if extra:
            df = df.drop(columns=extra)
        # reorder
        df = df[feature_names]

    # Ensure numeric dtype (one-hots will become 0/1)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

# -------------------------
# Fallback score
# -------------------------
def calculate_fallback_score(form_data):
    score = 50

    # Budget: budget_num (new) or budget (old)
    b = _to_float(form_data.get("budget_num", form_data.get("budget", 0)), 0)
    if b >= 1200: score += 20
    elif b >= 800: score += 15
    elif b >= 600: score += 10

    # Nationality: accept is_malaysian True or nationality ~ 'Malaysia/Malaysian'
    nat = normalize_nationality(
        form_data.get("nationality"),
        form_data.get("is_malaysian"),
        form_data.get("nationality_detail", "")
    )
    if nat == "Malaysia":
        score += 15

    # Move-in: prefer provided days_to_move_in, else parse date
    dd = None
    if form_data.get("days_to_move_in") is not None:
        try:
            dd = int(float(form_data["days_to_move_in"]))
        except Exception:
            dd = None
    if dd is None:
        mv = form_data.get("move_in_date") or form_data.get("movein")
        if mv:
            try:
                dd = (datetime.strptime(mv, "%Y-%m-%d") - datetime.now()).days
            except Exception:
                dd = None
    if dd is not None:
        if dd <= 30: score += 15
        elif dd <= 90: score += 10

    # Business hours: accept client flag, else use server time
    is_biz = form_data.get("is_business_hours")
    if isinstance(is_biz, bool):
        if is_biz: score += 10
    else:
        if 9 <= datetime.now().hour <= 17:
            score += 10

    # Completeness
    filled = sum(1 for v in form_data.values() if v not in (None, "", []))
    score += min(15, filled * 2)

    return max(0, min(100, score))

# -------------------------
# Routes
# -------------------------
@app.route("/")
def root():
    return jsonify({
        "status": "BeLive ALPS API running",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else None,
        "artifact_dir": ARTIFACT_DIR,
        "model_filename": MODEL_FILENAME,
        "features_filename": FEATURES_FILENAME,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/score", methods=["POST"])
def score():
    if not request.is_json:
        return jsonify({"error": "Send JSON", "score": 50}), 400
    payload = request.get_json(silent=True) or {}
    logger.info(f"Incoming keys: {list(payload.keys())}")

    # If model not loaded, return fallback (200 so the UI can proceed)
    if model is None:
        fb = calculate_fallback_score(payload)
        logger.error("Model not loaded — returning fallback")
        return jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "model_not_loaded"
        }), 200

    try:
        X = prepare_features(payload)
        logger.info(f"Prepared features shape: {X.shape}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # assume binary with class 1 = success
            p1 = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            score = max(0.0, min(100.0, p1 * 100.0))
        else:
            pred = model.predict(X)
            # if classify 0/1, treat as prob
            p1 = float(pred[0])
            score = max(0.0, min(100.0, p1 * 100.0))

        logger.info(f"Model OK — p1={p1:.4f} score={score:.2f}")
        return jsonify({
            "score": round(score, 2),
            "success_probability": round(p1, 4),
            "timestamp": datetime.now().isoformat(),
            "model_used": True,
            "model_type": type(model).__name__
        })
    except Exception as e:
        logger.exception(f"Scoring failed, using fallback: {e}")
        fb = calculate_fallback_score(payload)
        return jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "exception",
            "error": str(e)
        }), 200

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else None,
        "timestamp": datetime.now().isoformat()
    })

# -------------------------
# Entrypoint (dev)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")

