import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("belive-alps-api")

# -----------------------------------------------------------------------------
# Flask
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Globals / Artifacts
# -----------------------------------------------------------------------------
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.dirname(__file__))
MODEL_FILENAME = "best_rf_model.pkl"
FEATURES_FILENAME = "alps_feature_names.pkl"

model = None
feature_names = None


def load_model_artifacts():
    """
    Load joblib-saved model and optional feature names.
    Executed at import so it works when run by gunicorn (Railway).
    """
    global model, feature_names
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    feat_path = os.path.join(ARTIFACT_DIR, FEATURES_FILENAME)

    try:
        model = joblib.load(model_path)
        logger.info(f"[MODEL] Loaded {type(model).__name__} from {model_path}")
    except Exception as e:
        logger.error(f"[MODEL] Failed to load model from {model_path}: {e}")
        model = None

    try:
        feature_names = joblib.load(feat_path)
        logger.info(f"[FEATURES] Loaded feature list ({len(feature_names)} cols) from {feat_path}")
    except FileNotFoundError:
        feature_names = None
        logger.warning(f"[FEATURES] {feat_path} not found (continuing without schema)")
    except Exception as e:
        feature_names = None
        logger.warning(f"[FEATURES] Failed to load feature list: {e}")


# Load on import (works with gunicorn)
load_model_artifacts()

# -----------------------------------------------------------------------------
# Feature Prep
# -----------------------------------------------------------------------------
def prepare_features(form_data: dict) -> pd.DataFrame:
    """
    Convert request JSON into the exact feature frame expected by the trained model.
    Fills any missing trained columns with 0 and reorders columns to training order.
    """
    try:
        features = {}
        now = datetime.now()

        # ---------------- Core numerical ----------------
        budget = 800.0
        if form_data.get("budget"):
            try:
                budget = float(form_data["budget"])
            except Exception:
                budget = 800.0

        features["Budget"] = budget
        features["Rental Proposed"] = budget

        pax_mapping = {"1 person": 1, "2 people": 2, "More than 2": 3}
        features["No of Pax"] = pax_mapping.get(form_data.get("pax", "1 person"), 1)

        features["contact_hour"] = now.hour
        features["contact_month"] = now.month

        # ---------------- Engineered ----------------
        if budget == 0:
            features["Budget_Category_Encoded"] = 0
        elif budget < 500:
            features["Budget_Category_Encoded"] = 1
        elif budget < 1000:
            features["Budget_Category_Encoded"] = 2
        elif budget < 1500:
            features["Budget_Category_Encoded"] = 3
        else:
            features["Budget_Category_Encoded"] = 4

        features["Move_Urgency_Encoded"] = 0
        if form_data.get("movein"):
            try:
                move_date = datetime.strptime(form_data["movein"], "%Y-%m-%d")
                days_diff = (move_date - now).days
                if days_diff <= 30:
                    features["Move_Urgency_Encoded"] = 1
                elif days_diff <= 90:
                    features["Move_Urgency_Encoded"] = 2
                else:
                    features["Move_Urgency_Encoded"] = 3
            except Exception:
                features["Move_Urgency_Encoded"] = 0

        features["Is_Weekend"] = 1 if now.weekday() >= 5 else 0
        features["Is_Business_Hours"] = 1 if 9 <= now.hour <= 17 else 0

        # ---------------- One-hot: Journey ----------------
        journey_categories = [
            "Information_Collection", "Property_Inquiry", "Viewing_Arrangement",
            "Room_Selection", "Property_Viewing", "Booking_Process", "Other", "Unknown"
        ]
        current_journey = "Unknown"
        for c in journey_categories:
            features[f"Customer_Journey_Clean_{c}"] = 1 if c == current_journey else 0

        # ---------------- One-hot: Gender ----------------
        gender_categories = ["Male", "Female", "Mixed", "Unknown"]
        current_gender = form_data.get("gender", "Unknown")
        if current_gender not in gender_categories:
            current_gender = "Unknown"
        for c in gender_categories:
            features[f"Gender_Clean_{c}"] = 1 if c == current_gender else 0

        # ---------------- One-hot: Lead Source ----------------
        lead_categories = [
            "Facebook", "Google_Search", "Google_Ads", "Instagram", "WhatsApp",
            "Website", "Referral", "Walk_In", "Email", "Phone_Call", "Unknown"
        ]
        current_source = "Website"
        for c in lead_categories:
            features[f"Lead_Source_Standard_{c}"] = 1 if c == current_source else 0

        # ---------------- One-hot: Room Type ----------------
        room_categories = ["Studio", "1_Bedroom", "2_Bedroom", "3_Bedroom", "Other", "Unknown"]
        current_room_raw = str(form_data.get("room_type", "Unknown") or "Unknown")
        low = current_room_raw.lower()
        if "studio" in low:
            current_room = "Studio"
        elif "1" in current_room_raw and "bed" in low:
            current_room = "1_Bedroom"
        elif "2" in current_room_raw and "bed" in low:
            current_room = "2_Bedroom"
        elif "3" in current_room_raw and "bed" in low:
            current_room = "3_Bedroom"
        else:
            current_room = "Unknown"
        for c in room_categories:
            features[f"Room_Type_Standard_{c}"] = 1 if c == current_room else 0

        # ---------------- One-hot: Transportation ----------------
        transport_categories = ["Car", "Public Transport", "Both", "Unknown"]
        current_transport = "Car" if form_data.get("car") == "Yes" else "Unknown"
        for c in transport_categories:
            features[f"Transportation_{c}"] = 1 if c == current_transport else 0

        # ---------------- One-hot: Parking ----------------
        parking_categories = ["Yes", "No", "Unknown"]
        current_parking = form_data.get("parking", "Unknown")
        for c in parking_categories:
            features[f"Parking_{c}"] = 1 if c == current_parking else 0

        # ---------------- One-hot: Day of week ----------------
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_weekday = weekdays[now.weekday()]
        for dname in weekdays:
            features[f"contact_dayofweek_{dname}"] = 1 if dname == current_weekday else 0

        # ---------------- One-hot: Nationality (grouped) ----------------
        nationality_countries = [
            "Malaysia", "Indonesia", "India", "Sudan", "Zimbabwe", "China",
            "Thailand", "Myanmar", "Pakistan", "Yemen", "Other"
        ]
        current_nationality = form_data.get("nationality", "Other")
        if current_nationality == "Others" and form_data.get("nationality_detail"):
            detail = str(form_data["nationality_detail"]).title()
            current_nationality = detail if detail in nationality_countries else "Other"
        for c in nationality_countries:
            features[f"Nationality_Standard_Grouped_{c}"] = 1 if c == current_nationality else 0

        # ---------------- DataFrame + Align to training schema ----------------
        df = pd.DataFrame([features])

        if feature_names:
            # add any missing columns with 0
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            # drop any extra columns and order exactly as training
            df = df[feature_names]

        # Coerce to numeric to avoid dtype issues
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        logger.info(f"[FEATURES] Prepared DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception(f"[FEATURES] Preparation failed: {e}")
        raise


def calculate_fallback_score(form_data: dict) -> float:
    """Simple heuristic fallback (kept for debugging; not used if model works)."""
    score = 50
    try:
        if form_data.get("budget"):
            b = float(form_data["budget"])
            if b >= 1200: score += 20
            elif b >= 800: score += 15
            elif b >= 600: score += 10
    except Exception:
        pass

    if form_data.get("nationality") == "Malaysian":
        score += 15

    try:
        if form_data.get("movein"):
            md = datetime.strptime(form_data["movein"], "%Y-%m-%d")
            dd = (md - datetime.now()).days
            if dd <= 30: score += 15
            elif dd <= 90: score += 10
    except Exception:
        pass

    if 9 <= datetime.now().hour <= 17:
        score += 10

    filled = sum(1 for v in form_data.values() if v not in (None, "", []))
    score += min(15, filled * 2)

    return float(max(0, min(100, score)))

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    mp = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    fp = os.path.join(ARTIFACT_DIR, FEATURES_FILENAME)
    return jsonify({
        "status": "BeLive ALPS API running",
        "artifact_dir": ARTIFACT_DIR,
        "model_file": mp,
        "model_file_exists": os.path.exists(mp),
        "feature_file": fp,
        "feature_file_exists": os.path.exists(fp),
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else "Unknown",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/health")
def health():
    mp = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    fp = os.path.join(ARTIFACT_DIR, FEATURES_FILENAME)
    info = {
        "status": "ok" if model is not None else "model_not_loaded",
        "artifact_dir": ARTIFACT_DIR,
        "model_file_exists": os.path.exists(mp),
        "feature_file_exists": os.path.exists(fp),
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "feature_count": len(feature_names) if feature_names else "Unknown",
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(info)


@app.route("/api/model-info")
def model_info():
    try:
        return jsonify({
            "model_loaded": model is not None,
            "model_type": type(model).__name__ if model else None,
            "feature_names_available": feature_names is not None,
            "feature_count": len(feature_names) if feature_names else "Unknown"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/score", methods=["POST"])
def score():
    """
    Predict with the trained model.
    (If you want to force a 500 when model is missing, we keep that behavior.)
    """
    if not request.is_json:
        return jsonify({"error": "No JSON data provided"}), 400

    form_data = request.get_json(silent=True) or {}
    logger.info(f"[REQUEST] /api/score keys: {list(form_data.keys())}")

    # If you prefer to 500 when model isn't loaded (as you said 500/200 isn't important to change):
    if model is None:
        logger.error("[MODEL] Not loaded; refusing to score")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        X = prepare_features(form_data)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # binary: class 1 probability; multi-class: choose second column if applicable
            if proba.shape[1] > 1:
                success_probability = float(proba[0, 1])
            else:
                success_probability = float(proba[0, 0])
        else:
            pred = model.predict(X)
            success_probability = float(pred[0])

        score_val = max(0.0, min(100.0, success_probability * 100.0))

        logger.info(f"[PREDICT] model_used=True prob={success_probability:.4f} score={score_val:.2f}")
        return jsonify({
            "score": round(score_val, 2),
            "success_probability": round(success_probability, 4),
            "timestamp": datetime.now().isoformat(),
            "model_used": True,
            "model_type": type(model).__name__
        })

    except Exception as e:
        # If the model runs but errors (e.g., shape mismatch), log and return 500 so you can see it.
        logger.exception(f"[PREDICT] Error: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# Local dev entrypoint (Railway/Gunicorn won't hit this)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting BeLive ALPS API (dev server)")
    # Load again in dev just in case
    load_model_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")
