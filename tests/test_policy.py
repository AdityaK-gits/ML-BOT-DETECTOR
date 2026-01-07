import os
import importlib
import json
from pathlib import Path


def reset_policy(main):
    main.policy_cache = {}
    main.policy_mtime = 0.0


def test_env_only_defaults(monkeypatch):
    import main
    reset_policy(main)
    # Ensure env-only and no overrides -> defaults
    monkeypatch.setenv("POLICY_ENV_ONLY", "true")
    for k in [
        "FUSION_SUPERVISED",
        "FUSION_ANOMALY",
        "FUSION_SEQUENCE",
        "RISK_LOW",
        "RISK_HIGH",
        "RL_PER_IP_PER_MIN",
        "RL_PER_USER_PER_MIN",
        "CALIBRATION_METHOD",
    ]:
        monkeypatch.delenv(k, raising=False)

    pol = main.load_policy()
    assert pol["fusion_weights"]["supervised"] == 0.6
    assert pol["fusion_weights"]["anomaly"] == 0.2
    assert pol["fusion_weights"]["sequence"] == 0.2
    assert pol["risk_thresholds"]["low"] == 0.45
    assert pol["risk_thresholds"]["high"] == 0.75


def test_env_overrides(monkeypatch):
    import main
    reset_policy(main)
    monkeypatch.setenv("POLICY_ENV_ONLY", "true")
    monkeypatch.setenv("FUSION_SUPERVISED", "0.7")
    monkeypatch.setenv("FUSION_ANOMALY", "0.15")
    monkeypatch.setenv("FUSION_SEQUENCE", "0.15")
    monkeypatch.setenv("RISK_LOW", "0.6")
    monkeypatch.setenv("RISK_HIGH", "0.85")
    monkeypatch.setenv("RL_PER_IP_PER_MIN", "10")
    monkeypatch.setenv("RL_PER_USER_PER_MIN", "20")
    monkeypatch.setenv("CALIBRATION_METHOD", "sigmoid")

    pol = main.load_policy()
    fw = pol["fusion_weights"]
    assert fw["supervised"] == 0.7
    assert fw["anomaly"] == 0.15
    assert fw["sequence"] == 0.15
    assert pol["risk_thresholds"]["low"] == 0.6
    assert pol["risk_thresholds"]["high"] == 0.85
    assert pol["rate_limits"]["per_ip_per_min"] == 10
    assert pol["rate_limits"]["per_user_per_min"] == 20
    assert pol.get("calibration", {}).get("method") == "sigmoid"


def test_file_fallback(monkeypatch, tmp_path):
    # Create a temp policy file and ensure when POLICY_ENV_ONLY is false, file values are loaded
    import main
    reset_policy(main)

    cfg_dir = Path("configs")
    cfg_dir.mkdir(exist_ok=True)
    temp_policy = {
        "fusion_weights": {"supervised": 0.55, "anomaly": 0.25, "sequence": 0.2},
        "risk_thresholds": {"low": 0.5, "high": 0.9},
        "rate_limits": {"per_ip_per_min": 5, "per_user_per_min": 8},
        "calibration": {"method": "isotonic"},
    }
    policy_path = cfg_dir / "policy.json"
    # backup existing if present
    backup = None
    if policy_path.exists():
        backup = policy_path.read_bytes()
    policy_path.write_text(json.dumps(temp_policy))

    try:
        monkeypatch.setenv("POLICY_ENV_ONLY", "false")
        # clear env overrides so we verify file is used
        for k in [
            "FUSION_SUPERVISED",
            "FUSION_ANOMALY",
            "FUSION_SEQUENCE",
            "RISK_LOW",
            "RISK_HIGH",
            "RL_PER_IP_PER_MIN",
            "RL_PER_USER_PER_MIN",
            "CALIBRATION_METHOD",
        ]:
            monkeypatch.delenv(k, raising=False)
        pol = main.load_policy()
        fw = pol["fusion_weights"]
        assert fw["supervised"] == 0.55
        assert fw["anomaly"] == 0.25
        assert fw["sequence"] == 0.2
        assert pol["risk_thresholds"]["low"] == 0.5
        assert pol["risk_thresholds"]["high"] == 0.9
        assert pol["rate_limits"]["per_ip_per_min"] == 5
        assert pol["rate_limits"]["per_user_per_min"] == 8
    finally:
        # restore
        if backup is not None:
            policy_path.write_bytes(backup)
