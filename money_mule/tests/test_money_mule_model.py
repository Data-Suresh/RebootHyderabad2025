import os
import pytest
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf

@pytest.fixture(scope="module")
def run_training_script():
    # Import the script once to execute it (assumes it's idempotent)
    import money_mule_model
    yield
    # Optionally cleanup files if needed
    # for file in ["autoencoder_model.keras", "rf_model.pkl", ...]: os.remove(file)

def test_output_files_exist(run_training_script):
    expected_files = [
        "autoencoder_model.keras",
        "rf_model.pkl",
        "scaler.pkl",
        "reconstruction_scaler.pkl",
        "feature_columns.json",
        "rf_feature_columns.json",
        "classification_report.csv",
        "money_mule_with_IAI_CCR.csv",
    ]
    for file in expected_files:
        assert os.path.exists(file), f"{file} was not created"

def test_feature_columns_json(run_training_script):
    with open("feature_columns.json") as f:
        columns = json.load(f)
    assert isinstance(columns, list)
    assert "incoming_txns" in columns or len(columns) > 0

def test_rf_feature_columns(run_training_script):
    with open("rf_feature_columns.json") as f:
        rf_cols = json.load(f)
    assert isinstance(rf_cols, list)
    assert "IAI" in rf_cols and "CCR" in rf_cols

def test_models_can_be_loaded(run_training_script):
    autoencoder = tf.keras.models.load_model("autoencoder_model.keras")
    assert isinstance(autoencoder, tf.keras.Model)

    rf_model = joblib.load("rf_model.pkl")
    assert hasattr(rf_model, "predict")

    scaler = joblib.load("scaler.pkl")
    assert hasattr(scaler, "transform")

def test_transformed_csv_columns(run_training_script):
    df = pd.read_csv("money_mule_with_IAI_CCR.csv")
    required_cols = ['reconstruction_error', 'intent_score', 'IAI', 'CCR', 'CCR_grade']
    for col in required_cols:
        assert col in df.columns, f"{col} not found in transformed CSV"

def test_classification_report_valid(run_training_script):
    df = pd.read_csv("classification_report.csv")
    assert "precision" in df.columns
    assert "recall" in df.columns
