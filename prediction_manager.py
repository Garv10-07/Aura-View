import joblib
import os
import joblib

class PredictionManager:
    def __init__(self, model_path="prophet_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.reload_model()

    def reload_model(self):
        """
        Safe reload - if file missing/corrupt, keep model None.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ Prediction model load ho gaya:", self.model_path)
            else:
                self.model = None
                print("⚠️ Prediction model file missing:", self.model_path)
        except Exception as e:
            self.model = None
            print("❌ Prediction model load error:", e)

    def get_future_prediction(self, minutes_ahead: int = 5) -> int:
        """
        returns next future crowd prediction int
        """
        if self.model is None:
            return 0

        try:
            future = self.model.make_future_dataframe(periods=minutes_ahead, freq="min")
            forecast = self.model.predict(future)
            val = forecast["yhat"].iloc[-1]
            val = int(max(0, round(val)))
            return val
        except Exception as e:
            print("❌ Prediction error:", e)
            return 0
    def reload_model(self):
        try:
            self.model = joblib.load("prophet_model.pkl")
            print("✅ New model loaded into memory")
        except Exception as e:
            print("❌ Model reload failed:", e)
    def hard_reload(self):
        print("🔥 HARD reload prediction model")
        self.model = None
        self.last_prediction = None
        self.last_prediction_time = None
        self.model = joblib.load(self.model_path)
