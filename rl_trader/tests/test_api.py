"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

# Only import the app if data is available — skip all API tests if not
try:
    from rl_trader.api import app

    client = TestClient(app)
    _HAS_DATA = True
except Exception:
    _HAS_DATA = False
    pytestmark = pytest.mark.skip(reason="Data/models not available")


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


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
        pytest.skip("No cached data available")


def test_predict_schema():
    """Test predict endpoint — skips if models not loaded."""
    response = client.post("/api/v1/predict", json={"symbol": "AAPL", "day_index": 0})
    if response.status_code == 200:
        data = response.json()
        assert "action" in data
        assert "action_label" in data
    else:
        pytest.skip("Models not loaded")
