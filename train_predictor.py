import os
from pathlib import Path

import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from prophet import Prophet


# =====================================
# Load ENV (.env must be in same folder)
# =====================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

print("✅ Loaded env from:", ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Debug print (safe)
print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_SERVICE_ROLE_KEY:", "SET" if SUPABASE_SERVICE_ROLE_KEY else "MISSING")
print("Loaded keys:", [k for k in os.environ.keys() if "SUPABASE" in k])

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")


# ==========================
# Supabase Client
# ==========================
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ==========================
# Config
# ==========================
MODEL_PATH = "prophet_model.pkl"
FETCH_LIMIT = 5000
RESAMPLE_FREQ = "1min"   # 1 minute resampling


# ==========================
# Fetch crowd data
# ==========================
def fetch_data():
    all_rows = []
    start = 0
    batch = 1000

    while True:
        res = (
            supabase.table("crowd_data")
            .select("created_at,count")
            .order("created_at", desc=False)
            .range(start, start + batch - 1)
            .execute()
        )

        data = res.data or []
        if not data:
            break

        all_rows.extend(data)
        start += batch

        if len(data) < batch:
            break

    df = pd.DataFrame(all_rows)

    print("Fetched rows:", len(df))
    print("Latest timestamp:", df["created_at"].max())

    return df

    


# ==========================
# Prepare dataset for Prophet
# ==========================
def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning + preparing dataset for Prophet...")

    df = df.sort_values("created_at")

    df_prophet = df.rename(columns={"created_at": "ds", "count": "y"})[["ds", "y"]]
    df_prophet = df_prophet.drop_duplicates(subset=["ds"])

    # ✅ REMOVE TIMEZONE (IMPORTANT for Prophet)
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
    df_prophet = df_prophet.dropna(subset=["ds"])
    df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)

    # resample to regular frequency
    df_prophet = (
        df_prophet.set_index("ds")
        .resample("1min")
        .mean()
        .interpolate()
        .reset_index()
    )

    df_prophet["y"] = df_prophet["y"].clip(lower=0)
    return df_prophet



# ==========================
# Train Prophet Model
# ==========================
def train_prophet(df_prophet: pd.DataFrame) -> Prophet:
    print("🧠 Training Prophet model...")

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="additive",
    )
    

    model.fit(df_prophet)
    return model


# ==========================
# Main
# ==========================
def main():
    df = fetch_data()

    print("✅ Raw rows:", len(df))
    if len(df) < 20:
        raise RuntimeError("❌ Too little data. Collect more records before training.")

    df_prophet = prepare_prophet_df(df)

    print("✅ Prepared rows:", len(df_prophet))
    if len(df_prophet) < 50:
        raise RuntimeError("❌ Not enough data to train. Need at least ~50 points.")

    model = train_prophet(df_prophet)

    print(f"💾 Saving model => {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
