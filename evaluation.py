import json
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import joblib
import os

from features import extract_features_from_activity, features_to_dataframe
from sessionization import compute_session_features
from sequence import load_bigram_model, sequence_negative_log_likelihood, nll_to_score

def load_policy():
    defaults = {"fusion_weights": {"supervised": 0.6, "anomaly": 0.2, "sequence": 0.2},
                "risk_thresholds": {"low": 0.45, "high": 0.75}}
    path = os.path.join("configs", "policy.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {**defaults, **data}
    except Exception:
        return defaults

LABELS = {0: "human", 1: "scraper", 2: "api_abuse", 3: "fake_account"}


def simulate_low_and_slow_human(n: int = 200) -> List[List[Dict]]:
    sessions = []
    for i in range(n):
        base = 1_700_000_000_000  # ms
        evs = []
        t = base
        for path in ["/home", "/product", "/cart"]:
            t += int(np.random.uniform(1500, 5000))
            evs.append({
                "user_id": f"h{i}",
                "timestamp": pd.to_datetime(t, unit="ms").isoformat() + "Z",
                "request_path": path,
                "request_duration": float(abs(np.random.normal(0.3, 0.1))),
                "mouse_movements": [],
                "click_pattern": [],
                "typing_speed": float(abs(np.random.normal(120, 40))),
                "scroll_behavior": {"speed": abs(np.random.normal(6, 2)), "direction": "down"},
            })
        sessions.append(evs)
    return sessions


def simulate_bursty_scraping(n: int = 200) -> List[List[Dict]]:
    sessions = []
    for i in range(n):
        base = 1_700_000_500_000
        evs = []
        t = base
        for page in range(1, 8):
            t += 200  # very regular
            evs.append({
                "user_id": f"s{i}",
                "timestamp": pd.to_datetime(t, unit="ms").isoformat() + "Z",
                "request_path": f"/list?page={page}",
                "request_duration": float(abs(np.random.normal(0.05, 0.02))),
                "mouse_movements": [],
                "click_pattern": [],
                "typing_speed": 0.0,
                "scroll_behavior": {"speed": 0.0, "direction": "none"},
            })
        sessions.append(evs)
    return sessions


def simulate_api_abuse(n: int = 200) -> List[List[Dict]]:
    sessions = []
    for i in range(n):
        base = 1_700_001_000_000
        evs = []
        t = base
        for _ in range(10):
            t += int(np.random.uniform(100, 800))
            evs.append({
                "user_id": f"a{i}",
                "timestamp": pd.to_datetime(t, unit="ms").isoformat() + "Z",
                "request_path": "/api/search",
                "request_duration": float(abs(np.random.normal(0.06, 0.03))),
                "mouse_movements": [],
                "click_pattern": [],
                "typing_speed": 0.0,
                "scroll_behavior": {"speed": 0.0, "direction": "none"},
            })
        sessions.append(evs)
    return sessions


def simulate_fake_accounts(n: int = 200) -> List[List[Dict]]:
    sessions = []
    for i in range(n):
        base = 1_700_001_500_000
        evs = []
        t = base
        for path in ["/signup", "/verify", "/complete"]:
            t += int(np.random.uniform(300, 1500))
            evs.append({
                "user_id": f"f{i}",
                "timestamp": pd.to_datetime(t, unit="ms").isoformat() + "Z",
                "request_path": path,
                "request_duration": float(abs(np.random.normal(0.08, 0.04))),
                "mouse_movements": [],
                "click_pattern": [],
                "typing_speed": 0.0,
                "scroll_behavior": {"speed": abs(np.random.normal(3, 1)), "direction": "down"},
            })
        sessions.append(evs)
    return sessions


def load_artifacts():
    model = joblib.load("bot_detection_model.pkl")
    with open("feature_list.json", "r") as f:
        feature_order = json.load(f)
    with open("label_map.json", "r") as f:
        lm = json.load(f)
        # find human index
        human_idx = 0
        for k, v in (lm.items() if isinstance(lm, dict) else enumerate(lm)):
            try:
                k = int(k)
            except Exception:
                pass
            if v == "human":
                human_idx = int(k)
                break
    iso = joblib.load("unsupervised_isoforest.pkl")
    bigram = load_bigram_model("sequence_bigram.json")
    return model, feature_order, human_idx, iso, bigram


def score_session(sess_events: List[Dict], model, feature_order, human_idx, iso, bigram) -> float:
    feats = compute_session_features(sess_events)
    df_row = features_to_dataframe(feats, feature_order)
    proba = model.predict_proba(df_row)
    if proba.shape[1] > 2:
        sup = float(1.0 - proba[0][human_idx])
    else:
        sup = float(proba[0][1])
    # anomaly
    try:
        df_val = float(iso.decision_function(df_row.values)[0])
    except Exception:
        df_val = float(iso.score_samples(df_row.values)[0])
    ano = float(1.0 / (1.0 + np.exp(2.0 * df_val)))
    # sequence
    seq = [e.get("request_path", "") for e in sess_events]
    nll = sequence_negative_log_likelihood(bigram, seq)
    seq_score = float(nll_to_score(nll, bias=2.0, scale=1.0))
    policy = load_policy()
    w = policy.get("fusion_weights", {})
    fused = float(w.get("supervised", 0.6) * sup + w.get("anomaly", 0.2) * ano + w.get("sequence", 0.2) * seq_score)
    return fused


def evaluate():
    model, feature_order, human_idx, iso, bigram = load_artifacts()
    policy = load_policy()
    low_thr = float(policy.get("risk_thresholds", {}).get("low", 0.45))
    high_thr = float(policy.get("risk_thresholds", {}).get("high", 0.75))

    scenarios = [
        ("human_low_and_slow", simulate_low_and_slow_human(), 0),
        ("scraper_bursty", simulate_bursty_scraping(), 1),
        ("api_abuse", simulate_api_abuse(), 2),
        ("fake_accounts", simulate_fake_accounts(), 3),
    ]
    y_true = []
    y_score = []
    by_name = {}

    for name, sessions, label in scenarios:
        scores = [score_session(s, model, feature_order, human_idx, iso, bigram) for s in sessions]
        y_true.extend([1 if label != 0 else 0] * len(scores))  # 0=human->0, others->1
        y_score.extend(scores)
        by_name[name] = scores

    # Metrics
    y_true_arr = np.array(y_true)
    y_pred = (np.array(y_score) > 0.5).astype(int)
    precision = (np.sum((y_pred == 1) & (y_true_arr == 1)) / max(np.sum(y_pred == 1), 1))
    recall = (np.sum((y_pred == 1) & (y_true_arr == 1)) / max(np.sum(y_true_arr == 1), 1))
    fpr = (np.sum((y_pred == 1) & (y_true_arr == 0)) / max(np.sum(y_true_arr == 0), 1))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true_arr, y_score))
    except Exception:
        auc = float('nan')

    print("Policy thresholds: low=", low_thr, " high=", high_thr)
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("FPR:", round(fpr, 3))
    print("ROC-AUC:", round(auc, 3))

    # Scenario summaries
    for name, scores in by_name.items():
        arr = np.array(scores)
        print(f"{name}: n={len(arr)}, mean={arr.mean():.3f}, p90={np.quantile(arr, 0.9):.3f}")


if __name__ == "__main__":
    evaluate()
