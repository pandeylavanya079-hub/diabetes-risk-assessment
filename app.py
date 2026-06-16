import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load models and scalers
try:
    with open('diabetes_model.pkl', 'rb') as f:
        diabetes_model = pickle.load(f)
    with open('diabetes_scaler.pkl', 'rb') as f:
        diabetes_scaler = pickle.load(f)
    print("Diabetes model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()
        
        # Features in order:
        # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        features = [
            float(data.get('pregnancies')),
            float(data.get('glucose')),
            float(data.get('blood_pressure')),
            float(data.get('skin_thickness')),
            float(data.get('insulin')),
            float(data.get('bmi')),
            float(data.get('pedigree_function')),
            float(data.get('age'))
        ]
        
        # Scale and predict
        features_scaled = diabetes_scaler.transform([features])
        prediction = int(diabetes_model.predict(features_scaled)[0])
        probabilities = diabetes_model.predict_proba(features_scaled)[0]
        confidence = float(probabilities[prediction])
        
        result = {
            'outcome': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'prediction': prediction,
            'confidence': confidence,
            'probability_diabetic': float(probabilities[1])
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Running on port 5000 by default
    app.run(debug=True, port=5000)
