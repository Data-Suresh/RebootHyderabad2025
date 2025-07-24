import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from fastapi import FastAPI, Request

app = FastAPI()

# Load models and scaler
autoencoder = tf.keras.models.load_model("autoencoder_model.keras")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
reconstruction_scaler = joblib.load("reconstruction_scaler.pkl")


# Load expected feature columns
with open("feature_columns.json") as f:
    expected_columns = json.load(f)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    try:
        input_json = await request.json()
        data = pd.DataFrame(input_json["instances"])

        # Ensure input has all required columns
        missing = set(expected_columns) - set(data.columns)
        if missing:
            return {"error": f"Missing columns in input: {missing}"}

        # Reorder columns to match training
        raw_data = data[expected_columns]

        # Scale input
        X_scaled = scaler.transform(raw_data)

        # Autoencoder prediction
        X_pred = autoencoder.predict(X_scaled)
        reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

        # Add engineered features
        data['reconstruction_error'] = reconstruction_error
        data['intent_score'] = data['reversal_flag'] + data['round_number_flag']

        # Calculate final features
        alpha, beta = 0.6, 0.4
        reconstruction_scaled = reconstruction_scaler.transform(data[['reconstruction_error']])
        data['CCR'] = 1 - reconstruction_scaled
        data['IAI'] = alpha * reconstruction_error + beta * data['intent_score']

        # Build final feature set used for training RF
        rf_features = expected_columns + ['reconstruction_error','intent_score', 'CCR', 'IAI']
        missing_rf_features = set(rf_features) - set(data.columns)
        if missing_rf_features:
            return {"error": f"Missing RF features in input: {missing_rf_features}"}
        with open("rf_feature_columns.json") as f:
            rf_feature_columns = json.load(f)

        # Ensure all expected RF features exist
        missing_rf_features = set(rf_feature_columns) - set(data.columns)
        if missing_rf_features:
            return {"error": f"Missing RF features in input: {missing_rf_features}"}

        X_final = data[rf_feature_columns] 
        preds = rf_model.predict(X_final)
        
        return {"predictions": preds.tolist()}

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

# Required for Vertex AI custom container
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
