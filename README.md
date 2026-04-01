🌱 Kisaan Sahayak – Smart Agriculture Assistant

Kisaan Sahayak is an AI-powered agriculture support system designed to assist farmers with crop recommendation and plant disease detection using machine learning techniques.

The system integrates predictive modeling with practical agricultural parameters to deliver actionable insights.

🚀 Key Features
Crop Recommendation System
  Predicts top 3 suitable crops based on:
      Nitrogen (N), Phosphorus (P), Potassium (K)
      Temperature, Humidity
      Soil pH and Rainfall
  Uses feature engineering + Random Forest model


Plant Disease Detection (In Progress)
  Placeholder for image-based disease classification
  Planned implementation using MobileNet (TensorFlow/Keras)

  
End-to-End ML Pipeline
    Data preprocessing
    Feature engineering
    Model training and persistence
    Prediction interface



**Project Structure**
sih_app/
│
├── app.py                  # Main application entry point
├── ml_model.py             # Crop prediction model (training + inference)
├── disease_model.py        # Disease detection (currently placeholder)
│
├── data/
│   └── Crop_recommendation.csv
│
├── crop_model.pkl          # Trained Random Forest model
├── scaler.pkl              # Feature scaler
├── label_encoder.pkl       # Label encoder
├── model_features.pkl      # Feature schema
│
└── __pycache__/            # Compiled files
