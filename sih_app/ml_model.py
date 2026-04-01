import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('data/crop_recommendation.csv')

# Feature engineering
def feature_engineer(df):
    df['NPK'] = (df['N'] + df['P'] + df['K']) / 3
    df['THI'] = df['temperature'] * df['humidity'] / 100
    df['rainfall_level'] = pd.cut(df['rainfall'],
                                  bins=[0, 50, 100, 200, 300],
                                  labels=[0,1,2,3])
    df['ph_category'] = df['ph'].apply(lambda p: 0 if p < 5.5 else 1 if p <= 7.5 else 2)
    df['temp_rain_interaction'] = df['temperature'] * df['rainfall']
    df['ph_rain_interaction'] = df['ph'] * df['rainfall']
    return df

data_fe = feature_engineer(data)

# Encode target
le = LabelEncoder()
data_fe['label'] = le.fit_transform(data_fe['label'])

# Split features & target
X = data_fe.drop('label', axis=1)
y = data_fe['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_scaled, y)

# Save model, scaler, encoder, feature columns
joblib.dump(rfc, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")


import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define the list of features your model expects
model_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                  'NPK', 'THI', 'rainfall_level', 'ph_category', 
                  'temp_rain_interaction', 'ph_rain_interaction']

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Feature engineering
    NPK = (N + P + K) / 3
    THI = temperature * humidity / 100
    temp_rain_interaction = temperature * rainfall
    ph_rain_interaction = ph * rainfall

    # Encode categorical features
    rainfall_level = 0 if rainfall <= 50 else 1 if rainfall <= 100 else 2 if rainfall <= 200 else 3
    ph_category = 0 if ph < 5.5 else 1 if ph <= 7.5 else 2

    # Create DataFrame in correct feature order
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall,
                              NPK, THI, rainfall_level, ph_category,
                              temp_rain_interaction, ph_rain_interaction]],
                            columns=model_features)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probabilities
    probs = model.predict_proba(features_scaled)[0]

    # Top 3 crops
    top3_idx = probs.argsort()[-3:][::-1]
    top3_crops = label_encoder.inverse_transform(top3_idx)

    return top3_crops
