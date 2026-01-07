import json
import pytest
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

@pytest.mark.parametrize("payload", [
    # Oversized arrays
    {"user_id": "u", "timestamp": "2026-01-01T00:00:00Z", "request_path": "/x", "request_duration": 0.1,
     "mouse_movements": [{"x":0,"y":0,"timestamp":0}] * 2100},
    # Wrong types
    {"user_id": 123, "timestamp": "2026-01-01T00:00:00Z", "request_path": "/x", "request_duration": 0.1},
    # Unknown fields
    {"user_id": "u", "timestamp": "2026-01-01T00:00:00Z", "request_path": "/x", "request_duration": 0.1, "evil": 1},
])
def test_detect_bot_fuzz(payload, monkeypatch):
    monkeypatch.setenv("BOT_API_KEY", "k")
    headers = {"X-API-Key": "k"}
    res = client.post("/detect-bot", json=payload, headers=headers)
    assert res.status_code in (400, 422)


def test_header_missing_api_key(monkeypatch):
    monkeypatch.setenv("BOT_API_KEY", "k")
    res = client.get("/health")
    assert res.status_code == 200
    # API endpoints should reject without key
    res = client.post("/detect-bot", json={})
    assert res.status_code in (400, 401, 422)
