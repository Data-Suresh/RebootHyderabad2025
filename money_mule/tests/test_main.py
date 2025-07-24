import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import json

from main import app

client = TestClient(app)

# Sample input payload
sample_input = {
    "instances": [
        {
            "account_id": "A123",
            "incoming_txns": 5,
            "outgoing_txns": 3,
            "time_between_txns": 2.5,
            "txn_amount_variance": 10.2,
            "unique_counterparties": 4,
            "txn_frequency": 7,
            "round_number_flag": 1,
            "reversal_flag": 0,
            "hour_of_day": 14,
            "account_age_days": 300,
        }
    ]
}

expected_columns = [
    "account_id", "incoming_txns", "outgoing_txns", "time_between_txns", 
    "txn_amount_variance", "unique_counterparties", "txn_frequency",
    "round_number_flag", "reversal_flag", "hour_of_day", "account_age_days"
]

rf_feature_columns = expected_columns + ['reconstruction_error', 'intent_score', 'CCR', 'IAI']

@pytest.fixture(autouse=True)
def mock_model_loading(monkeypatch):
    # Mock models and scalers
    mock_autoencoder = MagicMock()
    mock_autoencoder.predict.return_value = np.zeros((1, len(expected_columns)))

    mock_rf_model = MagicMock()
    mock_rf_model.predict.return_value = np.array([1])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, len(expected_columns)))

    mock_reconstruction_scaler = MagicMock()
    mock_reconstruction_scaler.transform.return_value = np.array([[0.1]])

    monkeypatch.setattr("main.autoencoder", mock_autoencoder)
    monkeypatch.setattr("main.rf_model", mock_rf_model)
    monkeypatch.setattr("main.scaler", mock_scaler)
    monkeypatch.setattr("main.reconstruction_scaler", mock_reconstruction_scaler)

    # Patch expected columns
    monkeypatch.setattr("main.expected_columns", expected_columns)

    # Patch rf_feature_columns.json file loading
    monkeypatch.setattr("builtins.open", lambda f, *args, **kwargs: 
        MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None, read=lambda: json.dumps(rf_feature_columns)))

    monkeypatch.setattr("json.load", lambda f: rf_feature_columns if "rf_feature_columns.json" in str(f) else expected_columns)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success():
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

def test_missing_column_handling():
    bad_input = {
        "instances": [
            {
                "incoming_txns": 5,
                "outgoing_txns": 3,
                "round_number_flag": 1,
                "reversal_flag": 0
                # other fields missing
            }
        ]
    }
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 200
    assert "error" in response.json()
    assert "Missing columns" in response.json()["error"]

def test_invalid_json():
    response = client.post("/predict", data="not-a-json")
    assert response.status_code == 422  # FastAPI validation error
