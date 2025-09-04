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
    """Accepts 1/2/3 numeric OR '1'/'2'/'3+' strings from HTML."""
    if isinstance(v, (int, float)):
        n = int(v)
        return 1 if n <= 1 else 2 if n == 2 else 3
    if isinstance(v, str):
        v = v.strip()
        if v == "1": return 1
        if v == "2": return 2  
        if v == "3+" or "3" in v: return 3
    return 1

# Lead source mapping based on HTML options
_LEAD_MAP = {
    "facebook": "Facebook",
    "instagram": "Instagram", 
    "portal (iproperty/propertyguru)": "Website",
    "portal": "Website",
    "iproperty": "Website",
    "propertyguru": "Website",
    "google": "Google_Search",
    "google_search": "Google_Search",
    "referral": "Referral",
    "walk-in": "Walk_In",
    "walkin": "Walk_In",
    "whatsapp": "WhatsApp",
    "other": "Unknown",
    "unknown": "Unknown",
}

def normalize_lead_source(s: str) -> str:
    if not s: return "Website"
    key = str(s).strip().lower()
    return _LEAD_MAP.get(key, "Unknown")

def normalize_nationality(nationality: str = None, is_malaysian: bool = None) -> str:
    """Returns standardized nationality for modeling."""
    if is_malaysian is True or (nationality and nationality.lower() in ["malaysian", "malaysia"]):
        return "Malaysia"
    return "Other"

def normalize_user_type(user_type: str = None) -> str:
    """Map HTML user types to model categories."""
    if not user_type:
        return "Other"
    
    user_type = user_type.lower()
    if "working" in user_type or "professional" in user_type:
        return "Working"
    elif "student" in user_type:
        return "Student" 
    elif "intern" in user_type or "trainee" in user_type:
        return "Intern"
    else:
        return "Other"

def extract_location_features(area=None, property_name=None, workplace=None):
    """Extract location-based features for the model."""
    features = {}
    
    # Location search - map area to standardized locations
    if area:
        area_lower = area.lower()
        if "kl" in area_lower or "kuala lumpur" in area_lower:
            features["location_search"] = "KL_City"
        elif "selangor" in area_lower:
            features["location_search"] = "Selangor"
        elif "mont kiara" in area_lower:
            features["location_search"] = "Mont_Kiara"
        else:
            features["location_search"] = "Other"
    else:
        features["location_search"] = "Unknown"
    
    # Selected property - group common properties
    if property_name:
        prop_lower = property_name.lower()
        # Group common properties to reduce cardinality
        if any(term in prop_lower for term in ["residence", "suites", "regency"]):
            features["selected_property"] = "Residential_Complex"
        elif "mont kiara" in prop_lower:
            features["selected_property"] = "Mont_Kiara_Property"
        elif any(term in prop_lower for term in ["city", "urban", "kl"]):
            features["selected_property"] = "City_Property"
        else:
            features["selected_property"] = "Other_Property"
    else:
        features["selected_property"] = "Unknown"
    
    # Workplace hot spot
    workplace_hot = False
    if workplace:
        workplace_lower = workplace.lower()
        hot_spots = ["klcc", "kl sentral", "cyberjaya", "bangsar", "mont kiara", 
                    "petaling jaya", "subang", "damansara"]
        workplace_hot = any(spot in workplace_lower for spot in hot_spots)
    
    features["workplace_hot"] = workplace_hot
    
    return features

def prepare_features(form_data):
    """
    Build a single-row DataFrame aligned to your training feature space.
    Based on the actual model features from alps_rfm_machine_learning_1.py
    """
    current_date = datetime.now()
    feats = {}

    # ------- Core numeric features -------
    budget = _to_float(form_data.get("budget"), 0.0)
    feats["budget"] = budget
    feats["rental_proposed"] = budget  # Assuming these are the same

    # Pax: expect 1, 2, or 3+ from HTML
    feats["no_of_pax"] = normalize_pax(form_data.get("pax", 1))

    # Contact time features
    feats["contact_hour"] = current_date.hour
    feats["contact_month"] = current_date.month

    # RFM features (if your model uses them, otherwise set defaults)
    feats["frequency"] = 1  # New lead
    feats["recencydays"] = 0  # Today's lead

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

    # ------- Move urgency (from move_in_date) -------
    feats["Move_Urgency_Encoded"] = 0
    
    move_in_date = form_data.get("move_in_date")
    if move_in_date:
        try:
            move_dt = datetime.strptime(move_in_date, "%Y-%m-%d")
            days_diff = (move_dt - current_date).days
            if days_diff <= 30:
                feats["Move_Urgency_Encoded"] = 1  # Urgent
            elif days_diff <= 90:
                feats["Move_Urgency_Encoded"] = 2  # Soon
            else:
                feats["Move_Urgency_Encoded"] = 3  # Future
        except Exception:
            feats["Move_Urgency_Encoded"] = 0

    # ------- Temporal flags -------
    feats["Is_Business_Hours"] = 1 if 9 <= current_date.hour <= 17 else 0
    feats["Is_Weekend"] = 1 if current_date.weekday() >= 5 else 0

    # ------- Location features -------
    location_features = extract_location_features(
        area=form_data.get("area"),
        property_name=form_data.get("property"), 
        workplace=form_data.get("workplace")
    )
    feats.update(location_features)

    # ------- Lead Source (one-hots) -------
    lead_categories = [
        "Facebook", "Google_Search", "Google_Ads", "Instagram", "WhatsApp",
        "Website", "Referral", "Walk_In", "Email", "Phone_Call", "Unknown"
    ]
    
    lead_source = normalize_lead_source(form_data.get("lead_source", ""))
    for category in lead_categories:
        feats[f"Lead_Source_Standard_{category}"] = 1 if category == lead_source else 0

    # ------- Customer Journey (from HTML user_type) -------
    user_type = normalize_user_type(form_data.get("user_type"))
    journey_categories = ["Information_Collection", "Property_Inquiry", "Viewing_Arrangement", 
                         "Room_Selection", "Property_Viewing", "Booking_Process", "Unknown"]
    
    # Map user type to likely customer journey stage
    if user_type == "Student":
        primary_journey = "Information_Collection"
    elif user_type == "Working":
        primary_journey = "Property_Inquiry" 
    else:
        primary_journey = "Unknown"
    
    for category in journey_categories:
        feats[f"Customer_Journey_Clean_{category}"] = 1 if category == primary_journey else 0

    # ------- Gender (optional, default to Unknown) -------
    gender_categories = ["Male", "Female", "Mixed", "Unknown"]
    gender = form_data.get("gender", "Unknown")
    for category in gender_categories:
        feats[f"Gender_Clean_{category}"] = 1 if category == gender else 0

    # ------- Nationality -------
    nationality = normalize_nationality(form_data.get("nationality"), 
                                      form_data.get("nationality") == "Malaysian")
    feats["Is_Malaysian"] = 1 if nationality == "Malaysia" else 0

    # ------- Transportation & Parking (binary features) -------
    has_car = form_data.get("has_car", "No") == "Yes"
    feats["Transportation_Car"] = 1 if has_car else 0
    
    need_parking = form_data.get("need_parking", "No") == "Yes"  
    feats["Parking_Needed"] = 1 if need_parking else 0

    # ------- Contact & workplace flags -------
    contact = form_data.get("contact", "").strip()
    feats["Has_Contact"] = 1 if len(contact) >= 5 else 0
    
    feats["Workplace_Hot"] = 1 if location_features.get("workplace_hot", False) else 0

    # ------- Tenancy months -------
    tenancy_months = _to_float(form_data.get("tenancy_months"), 12)
    feats["Tenancy_Months"] = tenancy_months

    # -------- Build DataFrame and align to training feature order --------
    df = pd.DataFrame([feats])

    if feature_names:
        # Add any missing columns with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Drop any unexpected columns  
        extra = [c for c in df.columns if c not in feature_names]
        if extra:
            logger.info(f"Dropping extra columns: {extra}")
            df = df.drop(columns=extra)
        
        # Reorder to match training
        df = df[feature_names]

    # Ensure numeric dtype
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

# -------------------------
# Fallback score (enhanced for new features)
# -------------------------
def calculate_fallback_score(form_data):
    score = 50

    # Budget scoring
    budget = _to_float(form_data.get("budget"), 0)
    if budget >= 1500: score += 25
    elif budget >= 1200: score += 20
    elif budget >= 900: score += 15
    elif budget >= 600: score += 10
    elif budget > 0: score += 5

    # Nationality bonus
    if form_data.get("nationality") == "Malaysian":
        score += 15

    # Move-in urgency
    move_in_date = form_data.get("move_in_date")
    if move_in_date:
        try:
            days_diff = (datetime.strptime(move_in_date, "%Y-%m-%d") - datetime.now()).days
            if days_diff <= 30: score += 15
            elif days_diff <= 90: score += 10
        except Exception:
            pass

    # Business hours bonus
    if 9 <= datetime.now().hour <= 17:
        score += 10

    # Contact information bonus
    if form_data.get("contact", "").strip():
        score += 10

    # Location preference (KL/Selangor areas tend to be higher value)
    area = form_data.get("area", "").lower()
    if "kl" in area or "mont kiara" in area:
        score += 10

    # User type scoring
    user_type = form_data.get("user_type", "").lower()
    if "working" in user_type or "professional" in user_type:
        score += 8
    elif "student" in user_type:
        score += 5

    # Completeness bonus
    filled_fields = sum(1 for v in form_data.values() if v and str(v).strip())
    score += min(20, filled_fields * 2)

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
    logger.info(f"Incoming payload keys: {list(payload.keys())}")
    logger.info(f"Sample payload: {dict(list(payload.items())[:5])}")  # Log first 5 items

    # If model not loaded, return fallback
    if model is None:
        fb = calculate_fallback_score(payload)
        logger.error("Model not loaded – returning fallback")
        return jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "model_not_loaded"
        }), 200

    try:
        X = prepare_features(payload)
        logger.info(f"Prepared features shape: {X.shape}")
        logger.info(f"Feature columns: {list(X.columns)}")
        logger.info(f"Sample feature values: {X.iloc[0].to_dict()}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Assume binary classification with class 1 = success
            if proba.shape[1] > 1:
                p1 = float(proba[0][1])
            else:
                p1 = float(proba[0][0])
            score = max(0.0, min(100.0, p1 * 100.0))
        else:
            pred = model.predict(X)
            p1 = float(pred[0])
            score = max(0.0, min(100.0, p1 * 100.0))

        logger.info(f"Model prediction – p1={p1:.4f} score={score:.2f}")
        return jsonify({
            "score": round(score, 2),
            "success_probability": round(p1, 4),
            "timestamp": datetime.now().isoformat(),
            "model_used": True,
            "model_type": type(model).__name__,
            "features_used": len(X.columns)
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

@app.route("/api/debug", methods=["POST"])
def debug():
    """Debug endpoint to see what features are being generated"""
    if not request.is_json:
        return jsonify({"error": "Send JSON"}), 400
    
    payload = request.get_json(silent=True) or {}
    
    try:
        X = prepare_features(payload)
        return jsonify({
            "input_payload": payload,
            "generated_features": X.to_dict('records')[0],
            "feature_count": len(X.columns),
            "expected_features": feature_names[:10] if feature_names else None,
            "feature_names_count": len(feature_names) if feature_names else 0
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "input_payload": payload
        }), 500

# -------------------------
# Entrypoint (dev)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")
