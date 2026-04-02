"""
DigiGuard - Train ML Model
===========================
Generates synthetic teenager screen usage data
and trains a Random Forest classifier to predict
addiction risk levels: Low, Medium, High.

Run once: python train_model.py
Output:  model.pkl (used by Flask app)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def generate_data(n_samples=1000):
    """Generate synthetic teenager screen usage data."""
    np.random.seed(42)
    
    screen_time = np.random.uniform(1, 16, n_samples)      # hours/day
    social_media = np.random.uniform(0, 12, n_samples)      # hours/day
    gaming = np.random.uniform(0, 10, n_samples)            # hours/day
    night_usage = np.random.uniform(0, 6, n_samples)        # hours/day
    
    # Label based on combined risk factors
    labels = []
    for i in range(n_samples):
        total_score = (
            screen_time[i] * 0.35 +
            social_media[i] * 0.25 +
            gaming[i] * 0.20 +
            night_usage[i] * 0.20
        )
        
        if total_score < 3.5:
            labels.append(0)  # Low Risk
        elif total_score < 6.0:
            labels.append(1)  # Medium Risk
        else:
            labels.append(2)  # High Risk
    
    df = pd.DataFrame({
        'screen_time': screen_time,
        'social_media': social_media,
        'gaming': gaming,
        'night_usage': night_usage,
        'risk_level': labels
    })
    
    return df

def train_model():
    """Train Random Forest model and save as model.pkl."""
    print("=" * 50)
    print("DigiGuard ML Model Training")
    print("=" * 50)
    
    # Generate data
    print("\n📊 Generating 1000 synthetic data points...")
    df = generate_data(1000)
    
    risk_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    print(f"\nData distribution:")
    for level, count in df['risk_level'].value_counts().sort_index().items():
        print(f"  {risk_names[level]} Risk: {count} samples")
    
    # Split features and labels
    X = df[['screen_time', 'social_media', 'gaming', 'night_usage']]
    y = df['risk_level']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    print("\n🌳 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Accuracy: {accuracy * 100:.1f}%")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Low Risk', 'Medium Risk', 'High Risk']
    ))
    
    # Feature importance
    print("Feature Importance:")
    for name, importance in zip(X.columns, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    print("=" * 50)

if __name__ == '__main__':
    train_model()
