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
    Build features that exactly match your trained Random Forest model
    """
    current_date = datetime.now()
    feats = {}

    # Core numerical features from your model
    budget = _to_float(form_data.get("budget"), 0.0)
    feats["budget"] = budget
    feats["rental_proposed"] = budget  # Same as budget in your model
    feats["no_of_pax"] = int(form_data.get("pax", 1)) if form_data.get("pax") != "3+" else 3
    feats["contact_hour"] = current_date.hour
    feats["contact_month"] = current_date.month
    feats["frequency"] = 1  # New lead
    feats["recencydays"] = 0  # New lead

    # Budget Category Encoded (exactly as in your model)
    if budget == 0:
        feats["Budget_Category_Encoded"] = 0  # Unknown
    elif budget < 500:
        feats["Budget_Category_Encoded"] = 1  # Low
    elif budget < 1000:
        feats["Budget_Category_Encoded"] = 2  # Medium
    elif budget < 1500:
        feats["Budget_Category_Encoded"] = 3  # High
    else:
        feats["Budget_Category_Encoded"] = 4  # Premium

    # Move Urgency Encoded (exactly as in your model)
    move_date = form_data.get("move_in_date")
    if move_date:
        try:
            move_dt = datetime.strptime(move_date, "%Y-%m-%d")
            days_diff = (move_dt - current_date).days
            if days_diff <= 30:
                feats["Move_Urgency_Encoded"] = 1  # Urgent
            elif days_diff <= 90:
                feats["Move_Urgency_Encoded"] = 2  # Soon
            else:
                feats["Move_Urgency_Encoded"] = 3  # Future
        except:
            feats["Move_Urgency_Encoded"] = 0  # Unknown
    else:
        feats["Move_Urgency_Encoded"] = 0  # Unknown

    # Time-based features
    feats["Is_Business_Hours"] = 1 if 9 <= current_date.hour <= 17 else 0
    feats["Is_Weekend"] = 1 if current_date.weekday() >= 5 else 0

    # Location area (grouped as in your model)
    area = form_data.get("area", "Other")
    feats["location_area"] = area

    # Property group (grouped as in your model)
    property_name = form_data.get("property", "Other")
    feats["property_group"] = property_name

    # Nationality group (exactly as in your model)
    nationality = form_data.get("nationality", "Other")
    if nationality == "Malaysia":
        feats["nationality_group"] = "Malaysia"
        feats["Is_Malaysian"] = 1
    else:
        feats["nationality_group"] = nationality
        feats["Is_Malaysian"] = 0

    # Lead Source Standard (one-hot encoded as in your model)
    lead_source = form_data.get("lead_source", "Unknown")
    lead_sources = ["Facebook", "Google_Search", "Google_Ads", "Instagram", 
                   "WhatsApp", "Website", "Referral", "Walk_In", "Email", 
                   "Phone_Call", "Unknown"]
    
    # Map common variations
    source_mapping = {
        "Google": "Google_Search",
        "Walk-in": "Walk_In", 
        "Other": "Unknown"
    }
    mapped_source = source_mapping.get(lead_source, lead_source)
    
    for source in lead_sources:
        feats[f"Lead_Source_Standard_{source}"] = 1 if source == mapped_source else 0

    # Customer Journey Clean (one-hot encoded as in your model)
    user_type = form_data.get("user_type", "Unknown")
    journey_mapping = {
        "Working Professional": "Property_Inquiry",
        "Student": "Information_Collection", 
        "Intern/Trainee": "Information_Collection",
        "Other": "Unknown"
    }
    journey = journey_mapping.get(user_type, "Unknown")
    
    journeys = ["Information_Collection", "Property_Inquiry", "Viewing_Arrangement",
               "Room_Selection", "Property_Viewing", "Booking_Process", "Unknown"]
    
    for j in journeys:
        feats[f"Customer_Journey_Clean_{j}"] = 1 if j == journey else 0

    # Gender Clean (one-hot encoded as in your model)
    gender = form_data.get("gender", "Unknown")
    genders = ["Male", "Female", "Mixed", "Unknown"]
    
    for g in genders:
        feats[f"Gender_Clean_{g}"] = 1 if g == gender else 0

    # Binary features
    feats["Transportation_Car"] = 1 if form_data.get("has_car") == "Yes" else 0
    feats["Parking_Needed"] = 1 if form_data.get("need_parking") == "Yes" else 0
    
    # Workplace hot spot
    workplace = form_data.get("workplace", "").lower()
    hot_spots = ["klcc", "kl sentral", "cyberjaya", "bangsar", "mont kiara"]
    feats["Workplace_Hot"] = 1 if any(spot in workplace for spot in hot_spots) else 0
    
    # Contact info
    feats["Has_Contact"] = 1 if form_data.get("contact") else 0
    
    # Tenancy months
    feats["Tenancy_Months"] = _to_float(form_data.get("tenancy_months"), 12)

    # Build DataFrame
    df = pd.DataFrame([feats])

    # Align to training features if available
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        extra = [c for c in df.columns if c not in feature_names]
        if extra:
            df = df.drop(columns=extra)
        df = df[feature_names]

    # Ensure numeric types
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


