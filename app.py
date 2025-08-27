import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variable for model
model = None

def load_model():
    """Load the pickle model"""
    global model
    try:
        with open('best_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return True
    except FileNotFoundError:
        logger.error("Model file 'best_rf_model.pkl' not found")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def prepare_features(form_data):
    """
    Convert form data to model features based on your Random Forest model
    Adjust these mappings based on your actual training data
    """
    try:
        # Initialize feature dictionary
        features = {}
        
        # Date-based features
        current_date = datetime.now()
        
        # Initial Contact Date (most important feature - 38.03%)
        features['Initial Contact Date'] = current_date.timestamp()
        
        # Move in Date (16.57% importance)
        if form_data.get('movein'):
            move_date = datetime.strptime(form_data['movein'], '%Y-%m-%d')
            features['Move in Date'] = move_date.timestamp()
            
            # Move Urgency (1.46% importance)
            days_diff = (move_date - current_date).days
            if days_diff <= 7:
                features['Move_Urgency_Encoded'] = 3  # Urgent
            elif days_diff <= 30:
                features['Move_Urgency_Encoded'] = 2  # Soon
            else:
                features['Move_Urgency_Encoded'] = 1  # Flexible
        else:
            features['Move in Date'] = (current_date + timedelta(days=30)).timestamp()
            features['Move_Urgency_Encoded'] = 1
        
        # Contact hour (10.56% importance)
        features['contact_hour'] = current_date.hour
        
        # Business hours (3.07% importance)
        features['Is_Business_Hours'] = 1 if 9 <= current_date.hour <= 17 else 0
        
        # Weekend check (1.03% importance)
        features['Is_Weekend'] = 1 if current_date.weekday() >= 5 else 0
        
        # Day of week features
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_weekday = weekdays[current_date.weekday()]
        
        # Wednesday contact (3.04% importance)
        features['contact_dayofweek_Wednesday'] = 1 if current_weekday == 'Wednesday' else 0
        
        # Monday contact (1.78% importance)
        features['contact_dayofweek_Monday'] = 1 if current_weekday == 'Monday' else 0
        
        # Contact month (0.79% importance)
        features['contact_month'] = current_date.month
        
        # Budget features (7.46% + 1.87% + 1.66% importance combined)
        budget = 0
        if form_data.get('budget'):
            try:
                budget = float(form_data['budget'])
            except:
                budget = 800  # default
        
        features['Rental Proposed'] = budget
        features['Budget'] = budget
        
        # Budget category encoding
        if budget < 600:
            features['Budget_Category_Encoded'] = 1
        elif budget <= 1000:
            features['Budget_Category_Encoded'] = 2
        elif budget <= 1500:
            features['Budget_Category_Encoded'] = 3
        else:
            features['Budget_Category_Encoded'] = 4
        
        # Number of Pax (3.07% importance)
        pax_mapping = {'1 person': 1, '2 people': 2, 'More than 2': 3}
        features['No of Pax'] = pax_mapping.get(form_data.get('pax', '1 person'), 1)
        
        # Nationality (3.77% importance)
        nationality_mapping = {
            'Malaysian': 1,
            'Singaporean': 2,
            'Chinese': 3,
            'Indian': 4,
            'Indonesian': 5,
            'Filipino': 6,
            'Others': 7
        }
        
        nationality = form_data.get('nationality', 'Others')
        if nationality == 'Others' and form_data.get('nationality_detail'):
            # You might want to map specific nationalities here
            nationality = form_data['nationality_detail']
        
        features['Nationality_Standard_Grouped'] = nationality_mapping.get(nationality, 7)
        
        # Customer Journey - Unknown fields (3.87% importance)
        # Count how many fields are filled vs total expected
        expected_fields = ['area', 'property', 'budget', 'pax', 'movein', 'nationality', 'workplace', 'car', 'parking']
        filled_fields = sum(1 for field in expected_fields if form_data.get(field))
        features['Customer_Journey_Clean_Unknown'] = 1 - (filled_fields / len(expected_fields))
        
        # Additional features that might be in your model
        # Area encoding
        area_mapping = {
            'KL/Selangor': 1,
            'Perak': 2,
            'JB/Penang': 3,
            'Negeri Sembilan': 4,
            'Genting Highlands': 5
        }
        features['Area_Encoded'] = area_mapping.get(form_data.get('area', 'KL/Selangor'), 1)
        
        # Property type (if your model uses it)
        features['Property_Type'] = 1  # Default value
        
        # Car and parking
        features['Has_Car'] = 1 if form_data.get('car') == 'Yes' else 0
        features['Needs_Parking'] = 1 if form_data.get('parking') == 'Yes' else 0
        
        # Gender
        features['Gender'] = 1 if form_data.get('gender') == 'Male' else 0
        
        # Tenancy period
        features['Tenancy_Months'] = 12 if form_data.get('tenancy') == '12 months' else 6
        
        # Convert to DataFrame (required for most sklearn models)
        feature_df = pd.DataFrame([features])
        
        # Ensure all expected features are present (fill missing with defaults)
        # You might need to adjust this based on your model's exact feature list
        expected_model_features = [
            'Initial Contact Date', 'Move in Date', 'contact_hour', 'Rental Proposed',
            'Customer_Journey_Clean_Unknown', 'Nationality_Standard_Grouped', 'No of Pax',
            'Is_Business_Hours', 'contact_dayofweek_Wednesday', 'Budget_Category_Encoded',
            'contact_dayofweek_Monday', 'Budget', 'Move_Urgency_Encoded', 'Is_Weekend',
            'contact_month'
        ]
        
        for feature in expected_model_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        # Reorder columns to match training data (important for some models)
        feature_df = feature_df.reindex(columns=expected_model_features, fill_value=0)
        
        logger.info(f"Prepared features: {feature_df.columns.tolist()}")
        return feature_df
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        # Return default features
        default_features = pd.DataFrame([{
            'Initial Contact Date': datetime.now().timestamp(),
            'Move in Date': (datetime.now() + timedelta(days=30)).timestamp(),
            'contact_hour': 12,
            'Rental Proposed': 800,
            'Customer_Journey_Clean_Unknown': 0.5,
            'Nationality_Standard_Grouped': 7,
            'No of Pax': 1,
            'Is_Business_Hours': 1,
            'contact_dayofweek_Wednesday': 0,
            'Budget_Category_Encoded': 2,
            'contact_dayofweek_Monday': 0,
            'Budget': 800,
            'Move_Urgency_Encoded': 2,
            'Is_Weekend': 0,
            'contact_month': datetime.now().month
        }])
        return default_features

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'BeLive ALPS API is running',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/score', methods=['POST'])
def calculate_score():
    """Calculate ALPS score using the pickle model"""
    try:
        # Get JSON data from request
        if not request.json:
            return jsonify({'error': 'No JSON data provided', 'score': 50}), 400
        
        form_data = request.json
        logger.info(f"Received data: {form_data}")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded', 'score': 50}), 500
        
        # Prepare features
        features_df = prepare_features(form_data)
        logger.info(f"Features prepared: {features_df.shape}")
        
        # Make prediction
        try:
            # For most sklearn models
            if hasattr(model, 'predict_proba'):
                # If it's a classification model, get probability of positive class
                probabilities = model.predict_proba(features_df)
                if probabilities.shape[1] > 1:
                    score = probabilities[0][1] * 100  # Probability of positive class
                else:
                    score = probabilities[0][0] * 100
            else:
                # If it's a regression model or direct scoring
                prediction = model.predict(features_df)
                score = float(prediction[0])
        
        except Exception as model_error:
            logger.error(f"Model prediction error: {str(model_error)}")
            # Fallback to a simple heuristic
            score = calculate_fallback_score(form_data)
        
        # Ensure score is in 0-100 range
        score = max(0, min(100, float(score)))
        
        logger.info(f"Calculated score: {score}")
        
        return jsonify({
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'model_used': True
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        # Return fallback score instead of error
        fallback_score = calculate_fallback_score(request.json if request.json else {})
        return jsonify({
            'score': fallback_score,
            'timestamp': datetime.now().isoformat(),
            'model_used': False,
            'fallback': True
        }), 200

def calculate_fallback_score(form_data):
    """Fallback scoring when model fails"""
    score = 50  # Base score
    
    if form_data.get('budget'):
        try:
            budget = float(form_data['budget'])
            if budget >= 800:
                score += 20
            elif budget >= 600:
                score += 10
        except:
            pass
    
    if form_data.get('nationality') == 'Malaysian':
        score += 15
    
    if form_data.get('movein'):
        try:
            move_date = datetime.strptime(form_data['movein'], '%Y-%m-%d')
            days_diff = (move_date - datetime.now()).days
            if days_diff <= 7:
                score += 15  # Urgent
            elif days_diff <= 30:
                score += 10  # Soon
        except:
            pass
    
    # Business hours bonus
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 17:
        score += 10
    
    return max(0, min(100, score))

@app.route('/api/health')
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'timestamp': datetime.now().isoformat(),
        'python_version': os.sys.version
    })

# Initialize model on startup
if __name__ == '__main__':
    logger.info("Starting BeLive ALPS API...")
    
    # Load model
    if not load_model():
        logger.warning("Model not loaded, using fallback scoring")
    
    # Get port from environment variable or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
