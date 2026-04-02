"""
DigiGuard - Flask Backend
==========================
Prediction API + Page serving.
Supabase handles auth, Flask handles ML prediction.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'digiguard-secret-key-change-me')
CORS(app)

# Supabase config (passed to frontend)
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

# Load ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully")
else:
    print("⚠️  model.pkl not found. Run 'python train_model.py' first.")

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# =============================================
# Page Routes
# =============================================

@app.route('/')
def index():
    return render_template('index.html',
                         supabase_url=SUPABASE_URL,
                         supabase_key=SUPABASE_KEY)

@app.route('/login')
def login():
    return render_template('login.html',
                         supabase_url=SUPABASE_URL,
                         supabase_key=SUPABASE_KEY)

@app.route('/register')
def register():
    return render_template('register.html',
                         supabase_url=SUPABASE_URL,
                         supabase_key=SUPABASE_KEY)

@app.route('/input')
def input_page():
    return render_template('input.html',
                         supabase_url=SUPABASE_URL,
                         supabase_key=SUPABASE_KEY)

@app.route('/result')
def result():
    return render_template('result.html',
                         supabase_url=SUPABASE_URL,
                         supabase_key=SUPABASE_KEY)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                         supabase_url=SUPABASE_URL,
                         supabase_key=SUPABASE_KEY)

# =============================================
# API Routes
# =============================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """Receive usage data, predict risk level."""
    try:
        data = request.get_json()
        
        screen_time = float(data.get('screen_time', 0))
        social_media = float(data.get('social_media', 0))
        gaming = float(data.get('gaming', 0))
        night_usage = float(data.get('night_usage', 0))
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 500
        
        # Predict
        features = np.array([[screen_time, social_media, gaming, night_usage]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)
        
        risk_level = RISK_LABELS[prediction]
        
        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'confidence': round(confidence, 1),
            'screen_time': screen_time,
            'social_media': social_media,
            'gaming': gaming,
            'night_usage': night_usage,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'supabase_configured': bool(SUPABASE_URL and SUPABASE_KEY)
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🛡️  DigiGuard Server Starting")
    print("=" * 50)
    print(f"  Supabase URL: {'✅ Set' if SUPABASE_URL else '❌ Not set'}")
    print(f"  Supabase Key: {'✅ Set' if SUPABASE_KEY else '❌ Not set'}")
    print(f"  ML Model:     {'✅ Loaded' if model else '❌ Not found'}")
    print(f"\n  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
