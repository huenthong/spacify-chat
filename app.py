import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
from pathlib import Path

# Import your LeadPreprocessor class
from preprocess import LeadPreprocessor

# -------------------------
# App & Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("belive-alps")

app = Flask(__name__)

# -------------------------
# VERY PERMISSIVE CORS CONFIGURATION
# -------------------------
CORS(app, origins=["https://huenthong.github.io"])

# -------------------------
# Globals
# -------------------------
preprocessor = None
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.dirname(__file__))
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "rf_model.pkl")

# -------------------------
# Load preprocessor on import
# -------------------------
def load_preprocessor():
    global preprocessor
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    
    try:
        preprocessor = LeadPreprocessor.load_rf_model(model_path)
        logger.info(f"✅ Loaded preprocessor with model: {type(preprocessor.best_model).__name__} from {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load preprocessor from {model_path}: {e}")
        preprocessor = None

load_preprocessor()

# -------------------------
# Feature mapping functions
# -------------------------
def map_form_to_model_data(form_data):
    """
    Map form data to match the expected columns in your model.
    This creates a DataFrame row that matches what your LeadPreprocessor expects.
    """
    current_date = datetime.now()
    
    # Create a dictionary with the expected columns from your model
    model_data = {
        # Core identification (won't be used in prediction but needed for pipeline)
        'customer_id': f'web_lead_{current_date.strftime("%Y%m%d_%H%M%S")}',
        
        # Date fields
        'initial_contact_date': current_date,
        'last_action_date': current_date,
        'move_in_date': None,  # Will be set below if provided
        
        # Customer journey - map from user_type
        'customer_journey': map_user_type_to_journey(form_data.get('user_type', 'Unknown')),
        
        # Location
        'location_search': form_data.get('area', 'Unknown'),
        
        # Property
        'selected_property': form_data.get('property', 'Unknown'),
        
        # Lead source
        'lead_source': normalize_lead_source(form_data.get('lead_source', 'Unknown')),
        'combined_lead_source': normalize_lead_source(form_data.get('lead_source', 'Unknown')),
        'source_from': normalize_lead_source(form_data.get('lead_source', 'Unknown')),
        
        # Budget and rental
        'budget': float(form_data.get('budget', 0)),
        'rental_proposed': float(form_data.get('budget', 0)),  # Same as budget
        
        # Personal details
        'gender': form_data.get('gender', 'Unknown'),
        'nationality': normalize_nationality_input(form_data),
        'no_of_pax': normalize_pax(form_data.get('pax', 1)),
        
        # Transportation
        'transportation': 'Car' if form_data.get('has_car') == 'Yes' else 'Unknown',
        'parking': 'Yes' if form_data.get('need_parking') == 'Yes' else 'No',
        
        # Room and tenancy
        'room_type': 'Unknown',  # Not provided in form
        'tenancy_period': f"{form_data.get('tenancy_months', 12)} months",
        
        # Contact timing
        'contact_hour': current_date.hour,
        'contact_month': current_date.month,
        'contact_dayofweek': current_date.strftime('%A'),
        
        # Engagement metrics
        'frequency': 1,  # New lead
        'recencydays': 0,  # New lead
        
        # Status (for preprocessing, won't affect prediction)
        'viewing_status': None,  # Unknown for new leads
        
        # Fields that will be removed by preprocessor but might be expected
        'lead_score': 0,
        'inserted_at': current_date,
        'clean_phone': form_data.get('contact', ''),
    }
    
    # Handle move-in date
    if form_data.get('move_in_date'):
        try:
            model_data['move_in_date'] = datetime.strptime(form_data['move_in_date'], '%Y-%m-%d')
        except:
            model_data['move_in_date'] = None
    
    return model_data

def map_user_type_to_journey(user_type):
    """Map user type to customer journey"""
    if not user_type:
        return 'Unknown'
    
    mapping = {
        'Working Professional': 'Property_Inquiry',
        'Student': 'Information_Collection',
        'Intern/Trainee': 'Information_Collection',
        'Other': 'Unknown'
    }
    return mapping.get(user_type, 'Unknown')

def normalize_lead_source(source):
    """Normalize lead source to match model expectations"""
    if not source:
        return 'Unknown'
    
    source_lower = str(source).lower().strip()
    mapping = {
        'facebook': 'Facebook',
        'instagram': 'Instagram', 
        'whatsapp': 'WhatsApp',
        'google': 'Google',
        'google_search': 'Google',
        'google_ads': 'Google Ads',
        'website': 'Website',
        'portal': 'Website',
        'referral': 'Referral',
        'walk_in': 'Walk-In',
        'walkin': 'Walk-In',
        'email': 'Email',
        'phone': 'Phone',
        'phone_call': 'Phone',
        'other': 'Unknown'
    }
    return mapping.get(source_lower, source.title())

def normalize_nationality_input(form_data):
    """Normalize nationality from form data"""
    nationality = form_data.get('nationality', 'Unknown')
    is_malaysian = form_data.get('is_malaysian', False)
    
    if is_malaysian or nationality.lower() in ['malaysia', 'malaysian']:
        return 'Malaysia'
    
    return nationality if nationality else 'Unknown'

def normalize_pax(pax_value):
    """Normalize pax to integer"""
    if isinstance(pax_value, (int, float)):
        return int(pax_value)
    
    if isinstance(pax_value, str):
        if pax_value.strip() == '3+':
            return 3
        try:
            return int(float(pax_value.strip()))
        except:
            return 1
    
    return 1

# -------------------------
# Fallback score
# -------------------------
def calculate_fallback_score(form_data):
    score = 50
    
    # Budget
    budget = float(form_data.get('budget', 0))
    if budget >= 1200:
        score += 20
    elif budget >= 800:
        score += 15
    elif budget >= 600:
        score += 10
    
    # Nationality
    nationality = normalize_nationality_input(form_data)
    if nationality == 'Malaysia':
        score += 15
    
    # Move-in urgency
    move_date = form_data.get('move_in_date')
    if move_date:
        try:
            move_dt = datetime.strptime(move_date, '%Y-%m-%d')
            days_diff = (move_dt - datetime.now()).days
            if days_diff <= 30:
                score += 15
            elif days_diff <= 90:
                score += 10
        except:
            pass
    
    # Business hours
    if 9 <= datetime.now().hour <= 17:
        score += 10
    
    # Completeness
    filled = sum(1 for v in form_data.values() if v not in (None, '', []))
    score += min(15, filled * 2)
    
    return max(0, min(100, score))

# -------------------------
# Manual CORS headers for all responses
# -------------------------
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Access-Control-Allow-Credentials'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# -------------------------
# Routes
# -------------------------
@app.route("/")
@cross_origin()
def root():
    response = jsonify({
        "status": "BeLive ALPS API running",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": preprocessor and preprocessor.best_model is not None,
        "model_type": type(preprocessor.best_model).__name__ if preprocessor and preprocessor.best_model else None,
        "feature_count": len(preprocessor.feature_names) if preprocessor and preprocessor.feature_names else None,
        "artifact_dir": ARTIFACT_DIR,
        "model_filename": MODEL_FILENAME,
        "timestamp": datetime.now().isoformat(),
        "cors_enabled": True
    })
    return add_cors_headers(response)

@app.route("/api/score", methods=["POST", "OPTIONS"])
@cross_origin()
def score():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        return add_cors_headers(response)
        
    if not request.is_json:
        response = jsonify({"error": "Send JSON", "score": 50})
        return add_cors_headers(response), 400
    
    payload = request.get_json(silent=True) or {}
    logger.info(f"Incoming keys: {list(payload.keys())}")
    
    # If preprocessor not loaded, return fallback
    if preprocessor is None or preprocessor.best_model is None:
        fb = calculate_fallback_score(payload)
        logger.error("Preprocessor/model not loaded – returning fallback")
        response = jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "preprocessor_not_loaded"
        })
        return add_cors_headers(response)
    
    try:
        # Map form data to model format
        model_data = map_form_to_model_data(payload)
        
        # Create DataFrame
        df = pd.DataFrame([model_data])
        
        logger.info(f"Created DataFrame with shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Use preprocessor to predict
        predictions, probabilities = preprocessor.predict(df)
        
        # Get the probability and score
        probability = float(probabilities[0])
        score_value = max(0.0, min(100.0, probability * 100.0))
        
        logger.info(f"Model prediction - probability={probability:.4f}, score={score_value:.2f}")
        
        response = jsonify({
            "score": round(score_value, 2),
            "success_probability": round(probability, 4),
            "timestamp": datetime.now().isoformat(),
            "model_used": True,
            "model_type": type(preprocessor.best_model).__name__
        })
        return add_cors_headers(response)
        
    except Exception as e:
        logger.exception(f"Scoring failed, using fallback: {e}")
        fb = calculate_fallback_score(payload)
        response = jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "prediction_exception",
            "error": str(e)
        })
        return add_cors_headers(response)

@app.route("/api/health")
@cross_origin()
def health():
    response = jsonify({
        "status": "ok",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": preprocessor and preprocessor.best_model is not None,
        "model_type": type(preprocessor.best_model).__name__ if preprocessor and preprocessor.best_model else None,
        "feature_count": len(preprocessor.feature_names) if preprocessor and preprocessor.feature_names else None,
        "timestamp": datetime.now().isoformat(),
        "cors_enabled": True
    })
    return add_cors_headers(response)

# -------------------------
# Apply CORS headers to all responses
# -------------------------
@app.after_request
def after_request(response):
    return add_cors_headers(response)

# -------------------------
# Entrypoint (dev)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")
