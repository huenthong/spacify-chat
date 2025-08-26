import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load pickle model
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "BeLive ALPS API is running"

@app.route('/api/score', methods=['POST'])
def calculate_score():
    try:
        data = request.json
        
        # Convert your form data to model features
        # Adjust this based on how you trained your model
        features = prepare_features(data)
        
        # Get prediction
        score = model.predict([features])[0]
        score = max(0, min(100, float(score)))
        
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'score': 50}), 200  # fallback score

def prepare_features(form_data):
    # Map your form data to model features
    # Example - adjust to match your model training:
    feature_vector = []
    
    # Area mapping
    areas = {'KL/Selangor': 1, 'Perak': 2, 'JB/Penang': 3}
    feature_vector.append(areas.get(form_data.get('area'), 0))
    
    # Budget
    budget = int(form_data.get('budget', 0)) if form_data.get('budget') else 0
    feature_vector.append(budget)
    
    # Add other features based on your model...
    
    return feature_vector

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
