from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Add cache-busting headers
@app.after_request
def add_cache_busting_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Load trained model
MODEL_PATH = 'model/model.pkl'

def load_model():
    """Load the trained model from pickle file"""
    if not os.path.exists(MODEL_PATH):
        return None
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

# Load model at startup
model = load_model()

@app.route('/')
def home():
    """Render the home page with input form"""
    if model is None:
        return '''
            <h1>Error</h1>
            <p>Model not found! Please run train_model.py first to train and save the model.</p>
            <pre>python train_model.py</pre>
        ''', 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get input data from request
        data = request.get_json()
        print(f"[{datetime.now()}] Prediction request received: {data}")
        
        temperature = float(data.get('temperature'))
        rainfall = float(data.get('rainfall', 0.5))  # Default rainfall value
        
        # Validate input ranges (based on dataset)
        if temperature < 40 or temperature > 85:
            return jsonify({'error': 'Temperature should be between 40°C and 85°C', 'success': False}), 400
        
        if rainfall < 0 or rainfall > 2:
            return jsonify({'error': 'Rainfall should be between 0 and 2 inches', 'success': False}), 400
        
        # Make prediction using DataFrame with proper feature names
        input_df = pd.DataFrame({
            'Temperature': [temperature],
            'Rainfall': [rainfall]
        })
        
        prediction = model.predict(input_df)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, round(float(prediction), 2))
        
        response_data = {
            'temperature': float(temperature),
            'rainfall': float(rainfall),
            'predicted_sales': float(prediction),
            'success': True
        }
        
        print(f"[{datetime.now()}] Prediction result: {response_data}")
        
        return jsonify(response_data)
    
    except ValueError as e:
        error_msg = f'Invalid input. Please enter numeric values. Error: {str(e)}'
        print(f"[{datetime.now()}] ValueError: {error_msg}")
        return jsonify({'error': error_msg, 'success': False}), 400
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"[{datetime.now()}] Exception: {error_msg}")
        return jsonify({'error': error_msg, 'success': False}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
