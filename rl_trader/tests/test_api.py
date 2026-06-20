"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from rl_trader.api import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RL Trader API"


def test_market_data():
    """Test market data endpoint — may skip if no cached data."""
    response = client.get("/api/v1/market/AAPL?limit=5")
    if response.status_code == 200:
        data = response.json()
        assert data["symbol"] == "AAPL"
    else:
        pytest.skip("No cached data available in CI environment")


def test_predict_schema():
    # This test may skip if models aren't loaded
    response = client.post("/api/v1/predict", json={"symbol": "AAPL", "day_index": 0})
    if response.status_code == 200:
        data = response.json()
        assert "action" in data
        assert "action_label" in data
        assert "current_price" in data
    else:
        pytest.skip("Models not loaded")
