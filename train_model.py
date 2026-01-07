import os
import json
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib

from features import extract_features_from_activity, features_to_dataframe
from sequence import train_bigram_model, save_bigram_model
from mlops.data_validation import (
    validate_training_dataframe,
    save_baseline_profile,
)

np.random.seed(42)

LABEL_MAP = {
    0: "human",
    1: "scraper",
    2: "api_abuse",
    3: "fake_account",
}


def gen_timestamp(start: datetime, span_minutes: int = 30) -> str:
    dt = start + timedelta(seconds=int(np.random.rand() * span_minutes * 60))
    return dt.isoformat() + "Z"


def simulate_human_activity(session_id: str, start: datetime) -> Dict:
    # Increase variance in humans: more jitter, uneven timings, occasional bursts
    num_moves = max(2, int(np.random.normal(25, 10)))
    mouse = []
    x, y = 100.0, 200.0
    ts = int(start.timestamp() * 1000)
    for i in range(max(2, num_moves)):
        x += np.random.normal(0, 8)
        y += np.random.normal(0, 8)
        ts += np.random.randint(5, 200)
        mouse.append({"x": x, "y": y, "timestamp": ts})

    clicks = []
    for _ in range(np.random.binomial(4, 0.5)):
        ts += np.random.randint(50, 3000)
        clicks.append({"x": x, "y": y, "timestamp": ts, "button": "left"})

    keystrokes = []
    t0 = int(start.timestamp())
    for i in range(np.random.randint(5, 30)):
        t0 += np.random.randint(50, 800) / 1000.0
        keystrokes.append(t0)

    return {
        "user_id": f"u_{session_id}",
        "timestamp": gen_timestamp(start),
        "request_path": "/home",
        "request_duration": float(abs(np.random.normal(0.3, 0.15))),
        "mouse_movements": mouse,
        "click_pattern": clicks,
        "typing_speed": None,
        "keystroke_timestamps": keystrokes,
        "scroll_behavior": {"speed": abs(np.random.normal(8, 3)), "direction": "down"},
    }


def simulate_scraper_activity(session_id: str, start: datetime) -> Dict:
    # Less trivial than before: mostly regular, slight jitter
    mouse = []
    clicks = []
    # Regular intervals for clicks to simulate automation
    ts = int(start.timestamp() * 1000)
    for _ in range(3):
        ts += int(np.random.normal(1000, 50))
        clicks.append({"x": 0, "y": 0, "timestamp": ts, "button": "left"})
    return {
        "user_id": f"b_{session_id}",
        "timestamp": gen_timestamp(start),
        "request_path": "/list?page=1",
        "request_duration": float(abs(np.random.normal(0.07, 0.03))),
        "mouse_movements": mouse,
        "click_pattern": clicks,
        "typing_speed": 0.0,
        "keystroke_timestamps": [],
        "scroll_behavior": {"speed": abs(np.random.normal(2, 1)), "direction": "down"},
    }


def simulate_api_abuse_activity(session_id: str, start: datetime) -> Dict:
    # Many API hits, low UI interaction, moderate jitter
    return {
        "user_id": f"a_{session_id}",
        "timestamp": gen_timestamp(start),
        "request_path": "/api/search",
        "request_duration": float(abs(np.random.normal(0.08, 0.04))),
        "mouse_movements": [],
        "click_pattern": [],
        "typing_speed": 0.0,
        "keystroke_timestamps": [],
        "scroll_behavior": {"speed": 0.0, "direction": "none"},
    }


def simulate_fake_account_activity(session_id: str, start: datetime) -> Dict:
    # Fast signup flow, templated but with some jitter
    mouse = [
        {"x": 100.0 + 2*i + np.random.normal(0, 1), "y": 100.0, "timestamp": int(start.timestamp()*1000) + int(50*i + np.random.normal(0, 10))}
        for i in range(3)
    ]
    clicks = [
        {"x": 100.0, "y": 200.0, "timestamp": int(start.timestamp()*1000) + int(800*i + np.random.normal(0, 100)), "button": "left"}
        for i in range(2)
    ]
    return {
        "user_id": f"f_{session_id}",
        "timestamp": gen_timestamp(start),
        "request_path": "/signup",
        "request_duration": float(abs(np.random.normal(0.12, 0.06))),
        "mouse_movements": mouse,
        "click_pattern": clicks,
        "typing_speed": 0.0,
        "keystroke_timestamps": [],
        "scroll_behavior": {"speed": abs(np.random.normal(3, 1)), "direction": "down"},
    }


def simulate_dataset(n_per_class: int = 400) -> pd.DataFrame:
    rows = []
    start = datetime.utcnow()
    for i in range(n_per_class):
        rows.append((simulate_human_activity(f"h{i}", start), 0))
        rows.append((simulate_scraper_activity(f"s{i}", start), 1))
        rows.append((simulate_api_abuse_activity(f"p{i}", start), 2))
        rows.append((simulate_fake_account_activity(f"f{i}", start), 3))
    # Extract features
    feats = []
    labels = []
    for payload, label in rows:
        f = extract_features_from_activity(payload)
        feats.append(f)
        labels.append(label)
    df = pd.DataFrame(feats)
    df.fillna(0.0, inplace=True)
    df["label"] = labels
    return df


def main():
    print("Simulating dataset...")
    df = simulate_dataset(n_per_class=300)
    feature_cols = [c for c in df.columns if c != "label"]

    print("Running Great Expectations validation on training dataframe...")
    ge_stats = validate_training_dataframe(df)
    print("Validation statistics:", ge_stats)

    baseline_path = os.getenv("BASELINE_PROFILE_PATH", "mlops/baselines/training_profile.json")
    print(f"Saving baseline feature profile to {baseline_path}...")
    save_baseline_profile(df, baseline_path, feature_cols)

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Further split train into base-train and calib-val for calibration
    X_base, X_calib, y_base, y_calib = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    print("Training XGBClassifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_jobs=4,
    )
    model.fit(X_base, y_base)

    # Evaluate
    y_proba = model.predict_proba(X_test)
    # Compute macro AUC by binarizing one-vs-rest
    try:
        aucs = []
        for k in range(4):
            aucs.append(roc_auc_score((y_test == k).astype(int), y_proba[:, k]))
        print("Macro AUC:", float(np.mean(aucs)))
    except Exception:
        pass

    y_pred = np.argmax(y_proba, axis=1)
    print(classification_report(y_test, y_pred, target_names=[LABEL_MAP[i] for i in range(4)]))

    # Calibrate probabilities (env-first method: isotonic or sigmoid/Platt)
    calib_method = os.getenv("CALIBRATION_METHOD", "isotonic").lower()
    if calib_method not in ("isotonic", "sigmoid"):
        calib_method = "isotonic"
    print(f"Calibrating supervised probabilities ({calib_method})...")
    calibrator = CalibratedClassifierCV(estimator=model, method=calib_method, cv="prefit")
    calibrator.fit(X_calib, y_calib)

    # Persist supervised artifacts
    print("Saving supervised artifacts (calibrated and base)...")
    joblib.dump(calibrator, "bot_detection_model_calibrated.pkl")
    joblib.dump(model, "bot_detection_model.pkl")
    with open("feature_list.json", "w") as f:
        json.dump(feature_cols, f)
    with open("label_map.json", "w") as f:
        json.dump(LABEL_MAP, f)

    # Train unsupervised model (IsolationForest) on human-only data
    print("Training IsolationForest on human-only sessions...")
    human_df = df[df["label"] == 0]
    X_human = human_df[feature_cols].values
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=4,
    )
    iso.fit(X_human)
    joblib.dump(iso, "unsupervised_isoforest.pkl")
    
    # Train a simple bigram model on synthetic endpoint sequences
    print("Training bigram sequence model...")
    human_paths = [
        ["/home", "/product", "/cart"],
        ["/home", "/search", "/product", "/profile"],
        ["/home", "/product", "/product", "/cart"],
    ]
    scraper_paths = [[f"/list?page={i}" for i in range(1, 6)] for _ in range(3)]
    api_abuse_paths = [["/api/search", "/api/search", "/api/search", "/api/search"] for _ in range(3)]
    fake_account_paths = [["/signup", "/verify", "/complete"] for _ in range(3)]
    sequences = human_paths + scraper_paths + api_abuse_paths + fake_account_paths
    bigram = train_bigram_model(sequences, k_smoothing=1.0)
    save_bigram_model(bigram, "sequence_bigram.json")
    
    print("Done. Artifacts: bot_detection_model.pkl, unsupervised_isoforest.pkl, sequence_bigram.json, feature_list.json, label_map.json")


if __name__ == "__main__":
    main()
