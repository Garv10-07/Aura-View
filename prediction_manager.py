# file: prediction_manager.py (Updated Version)

import pickle
import pandas as pd
import os

class PredictionManager:
    def __init__(self, model_file='prophet_model.pkl'):
        self.model = None
        # Check karo ki model file exist karti hai ya nahin
        if os.path.exists(model_file):
            print("🧠 Trained prediction model load kiya jaa raha hai...")
            # Train kiye hue model ko file se load karo
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Prediction model successfully load ho gaya!")
        else:
            print(f"⚠️ Warning: Model file '{model_file}' nahin mili. Prediction kaam nahin karega.")

    def get_future_prediction(self):
        # Agar model load nahin hua, to 0 return kar do
        if self.model is None:
            return 0
        
        try:
            # Model se agle 1 minute (60 seconds) ka prediction maango
            future = self.model.make_future_dataframe(periods=1, freq='min')
            forecast = self.model.predict(future)
            
            # Aakhri prediction (jo humara future prediction hai) nikaalo
            predicted_value = forecast['yhat'].iloc[-1]
            
            # Prediction kabhi negative nahin ho sakta
            return max(0, round(predicted_value))
        except Exception as e:
            print(f"❌ Prediction karne mein error: {e}")
            return 0