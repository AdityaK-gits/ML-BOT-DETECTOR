from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

from features import extract_features_from_activity


def endpoint_entropy(paths: List[str]) -> float:
    if not paths:
        return 0.0
    counts = Counter(paths)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def compute_session_features(events: List[Dict]) -> Dict[str, float]:
    """
    Aggregate a list of activity events (as per UserActivity schema) into session-level features.
    Includes temporal stats and flow/entropy-like metrics.
    """
    if not events:
        return {}

    # Sort by timestamp
    def to_epoch_ms(ts: str) -> int:
        try:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        except Exception:
            return 0

    events_sorted = sorted(events, key=lambda e: to_epoch_ms(e.get("timestamp", "")))

    # Temporal
    times = [to_epoch_ms(e.get("timestamp", "")) for e in events_sorted]
    inter = np.diff(times) if len(times) > 1 else np.array([0.0])
    inter_ms = inter[inter >= 0]

    duration_ms = float((times[-1] - times[0]) if len(times) > 1 else 0.0)
    duration_min = max(duration_ms / 60000.0, 1e-6)
    requests_per_min = len(events_sorted) / duration_min
    inter_request_var = float(np.var(inter_ms)) if inter_ms.size > 0 else 0.0

    # Paths and entropy
    paths = [e.get("request_path", "") for e in events_sorted]
    ent = endpoint_entropy(paths)

    # Per-event features then simple aggregates
    per_rows = [extract_features_from_activity(e) for e in events_sorted]
    df = pd.DataFrame(per_rows).fillna(0.0)
    # Aggregate numeric features
    agg = df.mean(numeric_only=True).to_dict()
    # Add counts and temporal
    agg.update({
        "session_event_count": float(len(events_sorted)),
        "session_duration_ms": float(duration_ms),
        "requests_per_min": float(requests_per_min),
        "inter_request_var": float(inter_request_var),
        "endpoint_entropy": float(ent),
    })
    return {k: float(v) for k, v in agg.items()}
