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

# Global variables
model = None
feature_names = None

def load_model_artifacts():
    """Load the pickle model and feature information"""
    global model, feature_names
    try:
        # Load the main model
        with open('best_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        
        # Try to load feature names if available
        try:
            with open('alps_feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            logger.info(f"Feature names loaded: {len(feature_names)} features")
        except FileNotFoundError:
            logger.warning("alps_feature_names.pkl not found - will use default feature order")
            feature_names = None
            
        return True
    except FileNotFoundError:
        logger.error("Model file 'best_rf_model.pkl' not found")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def prepare_features(form_data):
    """
    Convert form data to model features based on actual Random Forest training
    """
    try:
        # Initialize all features with default values
        features = {}
        
        # Get current datetime
        current_date = datetime.now()
        
        # =============================================================================
        # CORE NUMERICAL FEATURES (from your ML code)
        # =============================================================================
        
        # Budget (direct numerical)
        budget = 800.0  # default
        if form_data.get('budget'):
            try:
                budget = float(form_data['budget'])
            except:
                budget = 800.0
        features['Budget'] = budget
        features['Rental Proposed'] = budget  # These were the same in your data
        
        # Number of Pax
        pax_mapping = {'1 person': 1, '2 people': 2, 'More than 2': 3}
        features['No of Pax'] = pax_mapping.get(form_data.get('pax', '1 person'), 1)
        
        # Contact hour and month (current time)
        features['contact_hour'] = current_date.hour
        features['contact_month'] = current_date.month
        
        # =============================================================================
        # ENGINEERED FEATURES (from your ML code)
        # =============================================================================
        
        # Budget Category (Label encoded as per your code)
        if budget == 0:
            features['Budget_Category_Encoded'] = 0  # Unknown
        elif budget < 500:
            features['Budget_Category_Encoded'] = 1  # Low
        elif budget < 1000:
            features['Budget_Category_Encoded'] = 2  # Medium
        elif budget < 1500:
            features['Budget_Category_Encoded'] = 3  # High
        else:
            features['Budget_Category_Encoded'] = 4  # Premium
        
        # Move Urgency (based on move-in date)
        features['Move_Urgency_Encoded'] = 0  # Unknown default
        if form_data.get('movein'):
            try:
                move_date = datetime.strptime(form_data['movein'], '%Y-%m-%d')
                days_diff = (move_date - current_date).days
                if days_diff <= 30:
                    features['Move_Urgency_Encoded'] = 1  # Urgent
                elif days_diff <= 90:
                    features['Move_Urgency_Encoded'] = 2  # Soon
                else:
                    features['Move_Urgency_Encoded'] = 3  # Future
            except:
                features['Move_Urgency_Encoded'] = 0
        
        # Time-based binary features
        features['Is_Weekend'] = 1 if current_date.weekday() >= 5 else 0
        features['Is_Business_Hours'] = 1 if 9 <= current_date.hour <= 17 else 0
        
        # =============================================================================
        # ONE-HOT ENCODED FEATURES (this is crucial for your model)
        # =============================================================================
        
        # Customer Journey - standardized categories from your ML code
        journey_categories = [
            'Information_Collection', 'Property_Inquiry', 'Viewing_Arrangement',
            'Room_Selection', 'Property_Viewing', 'Booking_Process', 'Other', 'Unknown'
        ]
        # Default to 'Unknown' - you'd need to map form data to these categories
        current_journey = 'Unknown'  # You can enhance this based on form completeness
        for category in journey_categories:
            features[f'Customer_Journey_Clean_{category}'] = 1 if category == current_journey else 0
        
        # Gender - standardized from your ML code
        gender_categories = ['Male', 'Female', 'Mixed', 'Unknown']
        current_gender = form_data.get('gender', 'Unknown')
        if current_gender not in gender_categories:
            current_gender = 'Unknown'
        for category in gender_categories:
            features[f'Gender_Clean_{category}'] = 1 if category == current_gender else 0
        
        # Lead Source - you'd need to determine this based on how the user arrived
        lead_categories = [
            'Facebook', 'Google_Search', 'Google_Ads', 'Instagram', 'WhatsApp',
            'Website', 'Referral', 'Walk_In', 'Email', 'Phone_Call', 'Unknown'
        ]
        current_source = 'Website'  # Default for web form
        for category in lead_categories:
            features[f'Lead_Source_Standard_{category}'] = 1 if category == current_source else 0
        
        # Room Type - standardized
        room_categories = ['Studio', '1_Bedroom', '2_Bedroom', '3_Bedroom', 'Other', 'Unknown']
        current_room = form_data.get('room_type', 'Unknown')
        # Map common variations
        if 'studio' in current_room.lower():
            current_room = 'Studio'
        elif '1' in current_room and 'bed' in current_room.lower():
            current_room = '1_Bedroom'
        elif '2' in current_room and 'bed' in current_room.lower():
            current_room = '2_Bedroom'
        elif '3' in current_room and 'bed' in current_room.lower():
            current_room = '3_Bedroom'
        else:
            current_room = 'Unknown'
            
        for category in room_categories:
            features[f'Room_Type_Standard_{category}'] = 1 if category == current_room else 0
        
        # Transportation
        transport_categories = ['Car', 'Public Transport', 'Both', 'Unknown']
        current_transport = 'Car' if form_data.get('car') == 'Yes' else 'Unknown'
        for category in transport_categories:
            features[f'Transportation_{category}'] = 1 if category == current_transport else 0
        
        # Parking
        parking_categories = ['Yes', 'No', 'Unknown']
        current_parking = form_data.get('parking', 'Unknown')
        for category in parking_categories:
            features[f'Parking_{category}'] = 1 if category == current_parking else 0
        
        # Day of week
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_weekday = weekdays[current_date.weekday()]
        for day in weekdays:
            features[f'contact_dayofweek_{day}'] = 1 if day == current_weekday else 0
        
        # Nationality (grouped as per your ML code)
        nationality_countries = [
            'Malaysia', 'Indonesia', 'India', 'Sudan', 'Zimbabwe', 'China',
            'Thailand', 'Myanmar', 'Pakistan', 'Yemen', 'Other'
        ]
        current_nationality = form_data.get('nationality', 'Other')
        # Handle "Others" case
        if current_nationality == 'Others' and form_data.get('nationality_detail'):
            detail = form_data['nationality_detail'].title()
            if detail in nationality_countries:
                current_nationality = detail
            else:
                current_nationality = 'Other'
        
        for country in nationality_countries:
            features[f'Nationality_Standard_Grouped_{country}'] = 1 if country == current_nationality else 0
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        logger.info(f"Prepared {len(features)} features for model")
        return feature_df
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise e

def get_feature_importance_info():
    """Return information about feature importance based on your ML results"""
    return {
        'top_features': [
            'Initial Contact Date (38.03%)',
            'Move in Date (16.57%)', 
            'contact_hour (10.56%)',
            'Rental Proposed (7.46%)',
            'Customer_Journey_Clean_Unknown (3.87%)',
            'Nationality_Standard_Grouped (3.77%)',
            'No of Pax (3.07%)',
            'Is_Business_Hours (3.07%)',
            'contact_dayofweek_Wednesday (3.04%)',
            'Budget_Category_Encoded (1.87%)'
        ],
        'model_type': 'Random Forest Classifier (Original - Default Parameters)',
        'performance': {
            'roc_auc': 'Check your model_metadata.pkl for exact values',
            'f1_score': 'Check your model_metadata.pkl for exact values'
        }
    }

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'BeLive ALPS API is running',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'features_loaded': feature_names is not None,
        'feature_count': len(feature_names) if feature_names else 'Unknown',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info')
def model_info():
    """Get information about the loaded model"""
    try:
        info = get_feature_importance_info()
        info.update({
            'model_loaded': model is not None,
            'model_type_actual': type(model).__name__ if model else None,
            'feature_names_available': feature_names is not None,
            'feature_count': len(feature_names) if feature_names else 'Unknown'
        })
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/score', methods=['POST'])
def calculate_score():
    """Calculate ALPS score using the trained Random Forest model"""
    try:
        # Get JSON data from request
        if not request.json:
            return jsonify({'error': 'No JSON data provided', 'score': 50}), 400
        
        form_data = request.json
        logger.info(f"Received data: {list(form_data.keys())}")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded', 'score': 50}), 500
        
        # Prepare features
        features_df = prepare_features(form_data)
        logger.info(f"Features prepared: {features_df.shape}")
        
        # Make prediction
        try:
            # Get probability of success (positive class)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_df)
                # Assuming binary classification where class 1 is success
                if probabilities.shape[1] > 1:
                    success_probability = probabilities[0][1]  # Probability of class 1 (success)
                else:
                    success_probability = probabilities[0][0]
            else:
                # Fallback if no predict_proba
                prediction = model.predict(features_df)
                success_probability = float(prediction[0])
            
            # Convert to 0-100 scale
            score = success_probability * 100
            
            # Ensure score is in valid range
            score = max(0, min(100, float(score)))
            
            logger.info(f"Model prediction - Success probability: {success_probability:.4f}, Score: {score:.2f}")
            
            return jsonify({
                'score': round(score, 2),
                'success_probability': round(success_probability, 4),
                'timestamp': datetime.now().isoformat(),
                'model_used': True,
                'model_type': type(model).__name__
            })
        
        except Exception as model_error:
            logger.error(f"Model prediction error: {str(model_error)}")
            # Return fallback score
            fallback_score = calculate_fallback_score(form_data)
            return jsonify({
                'score': fallback_score,
                'timestamp': datetime.now().isoformat(),
                'model_used': False,
                'error': str(model_error),
                'fallback': True
            }), 200
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        # Return fallback score instead of error
        fallback_score = calculate_fallback_score(request.json if request.json else {})
        return jsonify({
            'score': fallback_score,
            'timestamp': datetime.now().isoformat(),
            'model_used': False,
            'error': str(e),
            'fallback': True
        }), 200

def calculate_fallback_score(form_data):
    """Fallback scoring based on business logic when model fails"""
    score = 50  # Base score
    
    # Budget factor
    if form_data.get('budget'):
        try:
            budget = float(form_data['budget'])
            if budget >= 1200:
                score += 20
            elif budget >= 800:
                score += 15
            elif budget >= 600:
                score += 10
        except:
            pass
    
    # Nationality factor (Malaysian preference based on your data)
    if form_data.get('nationality') == 'Malaysian':
        score += 15
    
    # Move-in urgency
    if form_data.get('movein'):
        try:
            move_date = datetime.strptime(form_data['movein'], '%Y-%m-%d')
            days_diff = (move_date - datetime.now()).days
            if days_diff <= 30:
                score += 15  # Urgent
            elif days_diff <= 90:
                score += 10  # Soon
        except:
            pass
    
    # Business hours bonus
    current_hour = datetime.now().hour
    if 9 <= current_hour <= 17:
        score += 10
    
    # Completeness bonus
    filled_fields = sum(1 for value in form_data.values() if value and value != '')
    completeness_bonus = min(15, filled_fields * 2)
    score += completeness_bonus
    
    return max(0, min(100, score))

@app.route('/api/health')
def health_check():
    """Detailed health check"""
    health_info = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'features_available': feature_names is not None,
        'feature_count': len(feature_names) if feature_names else 'Unknown',
        'timestamp': datetime.now().isoformat(),
        'python_version': os.sys.version.split()[0]
    }
    
    # Try to get model info if available
    if model is not None:
        try:
            health_info['model_info'] = {
                'n_estimators': getattr(model, 'n_estimators', 'N/A'),
                'max_depth': getattr(model, 'max_depth', 'N/A'),
                'n_features_in_': getattr(model, 'n_features_in_', 'N/A')
            }
        except:
            pass
    
    return jsonify(health_info)

# Initialize model on startup
if __name__ == '__main__':
    logger.info("Starting BeLive ALPS API...")
    
    # Load model and artifacts
    if not load_model_artifacts():
        logger.warning("Model not loaded properly, using fallback scoring")
    else:
        logger.info("Model and artifacts loaded successfully")
    
    # Get port from environment variable or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
