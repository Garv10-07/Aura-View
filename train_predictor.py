# file: train_predictor.py

import pandas as pd
from prophet import Prophet
import sqlite3
import pickle

print("🚀 Prediction model ki training shuru ho rahi hai...")

# --- 1. Database se Data Load Karo ---
DB_FILE = "crowd_data.db"
conn = sqlite3.connect(DB_FILE)
# SQL query se saara data 'pandas' DataFrame mein padho
df = pd.read_sql_query("SELECT timestamp, person_count FROM crowd_counts", conn)
conn.close()
print(f"✅ Database se {len(df)} entries successfully load ho gayi hain.")

# --- 2. Data ko Prophet ke Format mein Taiyaar Karo ---
# Prophet ko column ke naam 'ds' (timestamp) aur 'y' (value) chahiye hote hain
df.rename(columns={'timestamp': 'ds', 'person_count': 'y'}, inplace=True)
# Timestamp ko sahi format mein convert karo
df['ds'] = pd.to_datetime(df['ds'])

# --- 3. Model ko Train Karo ---
print("🧠 Model ko data par train kiya jaa raha hai... (Ismein thoda time lag sakta hai)")
# Prophet ka model banakar use data par fit karo
model = Prophet()
model.fit(df)
print("✅ Model successfully train ho gaya hai!")

# --- 4. Train kiye hue Model ko Save Karo ---
# Hum 'pickle' ka use karke model ko ek file mein save karenge
MODEL_FILE = 'prophet_model.pkl'
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"🎉 Training poori ho gayi! Model ko '{MODEL_FILE}' file mein save kar diya gaya hai.")