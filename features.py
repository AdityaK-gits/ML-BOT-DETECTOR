from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from utils import (
    calculate_typing_speed,
    analyze_mouse_movements,
    detect_automation_patterns,
)


def extract_features_from_activity(activity: Dict) -> Dict[str, float]:
    """
    Convert a single activity payload into a flat feature dict suitable for model input.
    Expected activity keys align with UserActivity schema in main.py.
    """
    features: Dict[str, float] = {}

    # Basic timing
    request_duration = float(activity.get("request_duration", 0.0) or 0.0)
    features["request_duration"] = request_duration

    # Mouse movements
    mouse_movements: Optional[List[Dict[str, float]]] = activity.get("mouse_movements") or []
    movement_stats = analyze_mouse_movements(mouse_movements)
    features["mouse_avg_speed"] = float(movement_stats.get("avg_speed", 0.0))
    features["mouse_straightness"] = float(movement_stats.get("straightness", 0.0))
    features["num_mouse_movements"] = float(len(mouse_movements))

    # Clicks
    click_pattern: Optional[List[Dict[str, float]]] = activity.get("click_pattern") or []
    features["num_clicks"] = float(len(click_pattern))

    # Typing
    typing_speed = activity.get("typing_speed")
    if typing_speed is None:
        # if keystroke_timestamps provided, compute
        keystrokes = activity.get("keystroke_timestamps") or []
        typing_speed = calculate_typing_speed(keystrokes)
    features["typing_speed"] = float(typing_speed or 0.0)

    # Automation patterns (regularity, etc.)
    auto_feats = detect_automation_patterns(activity)
    # rename keys to keep feature space consistent
    features["click_regularity"] = float(auto_feats.get("click_regularity", 0.0))
    features["mouse_speed_copy"] = float(auto_feats.get("mouse_avg_speed", 0.0))
    features["mouse_straightness_copy"] = float(auto_feats.get("mouse_straightness", 0.0))
    features["typing_speed_cpm"] = float(auto_feats.get("typing_speed_cpm", 0.0))

    # Simple time-of-day bucket (optional feature)
    ts = activity.get("timestamp")
    try:
        hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour if isinstance(ts, str) else 0
    except Exception:
        hour = 0
    features["hour_of_day"] = float(hour)

    return features


def features_to_dataframe(features: Dict[str, float], feature_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert features dict to a single-row DataFrame in a consistent column order.
    If feature_order is provided, ensure columns are aligned and missing values are filled with 0.
    """
    if feature_order is None:
        return pd.DataFrame([features])
    row = {k: float(features.get(k, 0.0) or 0.0) for k in feature_order}
    return pd.DataFrame([row])
