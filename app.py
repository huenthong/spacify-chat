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
def prepare_features(form_data):
    """
    Build a single-row DataFrame with the engineered / one-hot features
    your RandomForest was trained on.
    """
    current_date = datetime.now()
    feats = {}

    # Core numeric
    try:
        budget = float(form_data.get("budget", 800) or 800)
    except Exception:
        budget = 800.0
    feats["Budget"] = budget
    feats["Rental Proposed"] = budget

    pax_map = {"1 person": 1, "2 people": 2, "More than 2": 3}
    feats["No of Pax"] = pax_map.get(form_data.get("pax", "1 person"), 1)

    feats["contact_hour"] = current_date.hour
    feats["contact_month"] = current_date.month

    # Engineered
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

    feats["Move_Urgency_Encoded"] = 0
    movein = form_data.get("movein")
    if movein:
        try:
            move_dt = datetime.strptime(movein, "%Y-%m-%d")
            dd = (move_dt - current_date).days
            if dd <= 30:
                feats["Move_Urgency_Encoded"] = 1
            elif dd <= 90:
                feats["Move_Urgency_Encoded"] = 2
            else:
                feats["Move_Urgency_Encoded"] = 3
        except Exception:
            feats["Move_Urgency_Encoded"] = 0

    feats["Is_Weekend"] = 1 if current_date.weekday() >= 5 else 0
    feats["Is_Business_Hours"] = 1 if 9 <= current_date.hour <= 17 else 0

    # One-hot categories (prefixes used in your training)
    # Customer Journey
    journey_categories = [
        "Information_Collection","Property_Inquiry","Viewing_Arrangement",
        "Room_Selection","Property_Viewing","Booking_Process","Other","Unknown"
    ]
    cur_journey = "Unknown"
    for c in journey_categories:
        feats[f"Customer_Journey_Clean_{c}"] = 1 if c == cur_journey else 0

    # Gender
    gender_categories = ["Male","Female","Mixed","Unknown"]
    cur_gender = form_data.get("gender", "Unknown")
    if cur_gender not in gender_categories:
        cur_gender = "Unknown"
    for c in gender_categories:
        feats[f"Gender_Clean_{c}"] = 1 if c == cur_gender else 0

    # Lead Source
    lead_categories = [
        "Facebook","Google_Search","Google_Ads","Instagram","WhatsApp",
        "Website","Referral","Walk_In","Email","Phone_Call","Unknown"
    ]
    cur_source = "Website"
    for c in lead_categories:
        feats[f"Lead_Source_Standard_{c}"] = 1 if c == cur_source else 0

    # Room Type
    room_categories = ["Studio","1_Bedroom","2_Bedroom","3_Bedroom","Other","Unknown"]
    rt = str(form_data.get("room_type", "Unknown") or "Unknown")
    rtl = rt.lower()
    if "studio" in rtl:
        cur_room = "Studio"
    elif "1" in rt and "bed" in rtl:
        cur_room = "1_Bedroom"
    elif "2" in rt and "bed" in rtl:
        cur_room = "2_Bedroom"
    elif "3" in rt and "bed" in rtl:
        cur_room = "3_Bedroom"
    else:
        cur_room = "Unknown"
    for c in room_categories:
        feats[f"Room_Type_Standard_{c}"] = 1 if c == cur_room else 0

    # Transportation
    transport_categories = ["Car","Public Transport","Both","Unknown"]
    cur_transport = "Car" if form_data.get("car") == "Yes" else "Unknown"
    for c in transport_categories:
        feats[f"Transportation_{c}"] = 1 if c == cur_transport else 0

    # Parking
    parking_categories = ["Yes","No","Unknown"]
    cur_park = form_data.get("parking", "Unknown")
    for c in parking_categories:
        feats[f"Parking_{c}"] = 1 if c == cur_park else 0

    # Day of week
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    cur_wd = weekdays[current_date.weekday()]
    for d in weekdays:
        feats[f"contact_dayofweek_{d}"] = 1 if d == cur_wd else 0

    # Nationality (grouped)
    nationality_countries = [
        "Malaysia","Indonesia","India","Sudan","Zimbabwe","China",
        "Thailand","Myanmar","Pakistan","Yemen","Other"
    ]
    nat = form_data.get("nationality", "Other")
    if nat == "Others" and form_data.get("nationality_detail"):
        detail = str(form_data["nationality_detail"]).title()
        nat = detail if detail in nationality_countries else "Other"
    for c in nationality_countries:
        feats[f"Nationality_Standard_Grouped_{c}"] = 1 if c == nat else 0

    df = pd.DataFrame([feats])

    # Align to training feature order (very important!)
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

    # Ensure numeric dtypes
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

# -------------------------
# Fallback score
# -------------------------
def calculate_fallback_score(form_data):
    score = 50
    try:
        b = float(form_data.get("budget", 0) or 0)
        if b >= 1200: score += 20
        elif b >= 800: score += 15
        elif b >= 600: score += 10
    except Exception:
        pass

    if form_data.get("nationality") == "Malaysian":
        score += 15

    mv = form_data.get("movein")
    if mv:
        try:
            dd = (datetime.strptime(mv, "%Y-%m-%d") - datetime.now()).days
            if dd <= 30: score += 15
            elif dd <= 90: score += 10
        except Exception:
            pass

    if 9 <= datetime.now().hour <= 17:
        score += 10

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

