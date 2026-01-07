from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import numpy as np
import os
import json
import joblib
import time
from datetime import datetime
import pathlib
import io
import base64
import hashlib
import random
from pydantic import BaseModel, Field, Extra, validator
from typing import Dict, List, Optional, Tuple
from features import extract_features_from_activity, features_to_dataframe
from sessionization import compute_session_features
from sequence import load_bigram_model, sequence_negative_log_likelihood, nll_to_score
try:
    import redis  # optional, for distributed rate limiting
except Exception:  # pragma: no cover
    redis = None
try:
    import shap  # optional explainability
except Exception:  # pragma: no cover
    shap = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

app = FastAPI(title="Bot Detection API")

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    REQ_COUNTER = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
    ERR_COUNTER = Counter("api_errors_total", "Total API errors", ["endpoint", "type"])
    RL_COUNTER = Counter("api_rate_limited_total", "Total rate limited requests", ["endpoint"])
    LATENCY_HIST = Histogram("api_latency_seconds", "Request latency", ["endpoint"])
except Exception:
    REQ_COUNTER = ERR_COUNTER = RL_COUNTER = LATENCY_HIST = None

# OpenTelemetry tracing (env-driven, no-op if OTEL envs absent)
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
except Exception:
    pass

# This would be loaded from a trained model file in production
# model = joblib.load('bot_detection_model.pkl')

# Mock model for demonstration
class MockModel:
    def predict_proba(self, X):
        # This is a mock prediction - in a real scenario, this would use a trained model
        return np.array([[0.7, 0.3]])  # 70% human, 30% bot

# Attempt to load real model artifacts if available
feature_order: Optional[List[str]] = None
human_idx: int = 0
unsup_model = None
bigram_model = None
calibrated_model = None
base_model_for_importance = None
policy_cache: Dict[str, object] = {}
policy_mtime: float = 0.0
rate_buckets: Dict[str, List[float]] = {}
redis_client = None
try:
    if os.path.exists("bot_detection_model.pkl"):
        base_model_for_importance = joblib.load("bot_detection_model.pkl")
        # Prefer calibrated model if available for predict_proba
        if os.path.exists("bot_detection_model_calibrated.pkl"):
            calibrated_model = joblib.load("bot_detection_model_calibrated.pkl")
        model = calibrated_model or base_model_for_importance
        if os.path.exists("feature_list.json"):
            with open("feature_list.json", "r") as f:
                feature_order = json.load(f)
        if os.path.exists("label_map.json"):
            with open("label_map.json", "r") as f:
                label_map = json.load(f)
                # label_map is like {"0": "human", ...} or {0: "human"}
                # Normalize keys to int
                normalized = {}
                for k, v in label_map.items():
                    try:
                        normalized[int(k)] = v
                    except Exception:
                        normalized[k] = v
                for k, v in normalized.items():
                    if v == "human":
                        human_idx = int(k)
                        break
        if os.path.exists("unsupervised_isoforest.pkl"):
            unsup_model = joblib.load("unsupervised_isoforest.pkl")
        if os.path.exists("sequence_bigram.json"):
            bigram_model = load_bigram_model("sequence_bigram.json")
        # Initialize Redis client if configured
        redis_url = os.getenv("REDIS_URL")
        if redis and redis_url:
            try:
                redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                # ping to verify
                redis_client.ping()
            except Exception:
                redis_client = None
    else:
        model = MockModel()
except Exception:
    model = MockModel()

def choose_scoring_model() -> Tuple[object, str]:
    """Simple canary: route CANARY_PERCENT of requests to base model; else calibrated (if available)."""
    canary_pct = float(os.getenv("CANARY_PERCENT", "0"))
    rnd = random.random()
    if base_model_for_importance is not None and rnd < canary_pct:
        return base_model_for_importance, os.getenv("MODEL_VERSION", "base")
    if calibrated_model is not None:
        return calibrated_model, os.getenv("MODEL_VERSION", "calibrated")
    return (base_model_for_importance or model), os.getenv("MODEL_VERSION", "base")

def load_policy() -> Dict:
    """Load policy from configs/policy.json with basic defaults and hot-reload on mtime change."""
    global policy_cache, policy_mtime
    path = os.path.join("configs", "policy.json")
    defaults = {
        "fusion_weights": {"supervised": 0.6, "anomaly": 0.2, "sequence": 0.2},
        "risk_thresholds": {"low": 0.45, "high": 0.75},
        "rate_limits": {"per_ip_per_min": 60, "per_user_per_min": 120},
    }
    policy_env_only = os.getenv("POLICY_ENV_ONLY", "false").lower() in ("1", "true", "yes")
    try:
        if not policy_env_only:
            mtime = os.path.getmtime(path)
            if not policy_cache or mtime != policy_mtime:
                with open(path, "r") as f:
                    data = json.load(f)
                policy_cache = {**defaults, **data}
                policy_mtime = mtime
        else:
            policy_cache = defaults
    except Exception:
        policy_cache = defaults
    # Environment-first overrides (numbers expected as strings)
    try:
        fw_sup = os.getenv("FUSION_SUPERVISED")
        fw_ano = os.getenv("FUSION_ANOMALY")
        fw_seq = os.getenv("FUSION_SEQUENCE")
        if fw_sup or fw_ano or fw_seq:
            policy_cache.setdefault("fusion_weights", {})
            if fw_sup: policy_cache["fusion_weights"]["supervised"] = float(fw_sup)
            if fw_ano: policy_cache["fusion_weights"]["anomaly"] = float(fw_ano)
            if fw_seq: policy_cache["fusion_weights"]["sequence"] = float(fw_seq)
        rl_ip = os.getenv("RL_PER_IP_PER_MIN")
        rl_user = os.getenv("RL_PER_USER_PER_MIN")
        if rl_ip or rl_user:
            policy_cache.setdefault("rate_limits", {})
            if rl_ip: policy_cache["rate_limits"]["per_ip_per_min"] = int(rl_ip)
            if rl_user: policy_cache["rate_limits"]["per_user_per_min"] = int(rl_user)
        low_thr = os.getenv("RISK_LOW")
        high_thr = os.getenv("RISK_HIGH")
        if low_thr or high_thr:
            policy_cache.setdefault("risk_thresholds", {})
            if low_thr: policy_cache["risk_thresholds"]["low"] = float(low_thr)
            if high_thr: policy_cache["risk_thresholds"]["high"] = float(high_thr)
        calib_method = os.getenv("CALIBRATION_METHOD")
        if calib_method:
            policy_cache.setdefault("calibration", {})
            policy_cache["calibration"]["method"] = calib_method
    except Exception:
        pass
    return policy_cache

def check_api_key(request: Request) -> None:
    """Validate API key from header X-API-Key against env BOT_API_KEY."""
    expected = os.getenv("BOT_API_KEY")
    if not expected:
        return  # allow if not configured for local/demo
    provided = request.headers.get("X-API-Key")
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _bucket_key(ip: str, user_id: Optional[str]) -> Tuple[str, str]:
    return (ip or "unknown", user_id or "anonymous")

def enforce_rate_limit(request: Request, user_id: Optional[str]) -> None:
    """Simple in-memory token bucket per minute for IP and user. Returns 429 on exceed."""
    policy = load_policy()
    ip_limit = int(policy.get("rate_limits", {}).get("per_ip_per_min", 60))
    user_limit = int(policy.get("rate_limits", {}).get("per_user_per_min", 120))
    now = time.time()
    ip = request.client.host if request and request.client else "unknown"
    # Prefer Redis if configured
    if redis_client is not None:
        pipe = redis_client.pipeline()
        ip_key = f"rl:ip:{ip}"
        user_key = f"rl:user:{user_id}" if user_id else None
        # increment and set expiry if new
        pipe.incr(ip_key)
        pipe.expire(ip_key, 60)
        ip_count, _ = pipe.execute()
        if int(ip_count) > ip_limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded for IP. Try again later.")
        if user_key:
            pipe = redis_client.pipeline()
            pipe.incr(user_key)
            pipe.expire(user_key, 60)
            user_count, _ = pipe.execute()
            if int(user_count) > user_limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for user. Try again later.")
        return

    # Clean old timestamps older than 60s
    def _allow(key: str, limit: int) -> bool:
        window = 60.0
        lst = rate_buckets.get(key, [])
        lst = [t for t in lst if now - t < window]
        if len(lst) >= limit:
            rate_buckets[key] = lst
            return False
        lst.append(now)
        rate_buckets[key] = lst
        return True

    # Per-IP
    if not _allow(f"ip:{ip}", ip_limit):
        if RL_COUNTER: RL_COUNTER.labels(endpoint="/rate").inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded for IP. Try again later.")
    # Per-user (if provided)
    if user_id:
        if not _allow(f"user:{user_id}", user_limit):
            if RL_COUNTER: RL_COUNTER.labels(endpoint="/rate").inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded for user. Try again later.")

# Simple SHAP image cache (Redis if configured, else in-memory)
_shap_cache: Dict[str, str] = {}

def _cache_set(key: str, val: str) -> None:
    try:
        if redis_client is not None:
            redis_client.setex(f"shap:{key}", 600, val)
            return
    except Exception:
        pass
    if len(_shap_cache) > 1000:
        _shap_cache.clear()
    _shap_cache[key] = val

def _cache_get(key: str) -> Optional[str]:
    try:
        if redis_client is not None:
            v = redis_client.get(f"shap:{key}")
            return v
    except Exception:
        pass
    return _shap_cache.get(key)

def log_audit(entry: Dict) -> None:
    try:
        log_dir = pathlib.Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

class UserActivity(BaseModel, extra=Extra.forbid):
    user_id: str = Field(..., min_length=1, max_length=128)
    timestamp: str = Field(..., min_length=10, max_length=64)
    request_path: str = Field(..., min_length=1, max_length=2048)
    request_duration: float = Field(..., ge=0.0, le=60.0)
    mouse_movements: Optional[List[Dict[str, float]]] = None
    click_pattern: Optional[List[Dict[str, float]]] = None
    typing_speed: Optional[float] = Field(default=None, ge=0.0, le=2000.0)
    scroll_behavior: Optional[Dict[str, float]] = None

    @validator("mouse_movements", pre=True)
    def _limit_mouse(cls, v):
        if v is None:
            return v
        if isinstance(v, list) and len(v) > 2000:
            raise ValueError("mouse_movements too long")
        return v

    @validator("click_pattern", pre=True)
    def _limit_clicks(cls, v):
        if v is None:
            return v
        if isinstance(v, list) and len(v) > 1000:
            raise ValueError("click_pattern too long")
        return v

class SessionPayload(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    events: List[UserActivity]

@app.post("/detect-bot")
async def detect_bot(request: Request, activity: UserActivity):
    try:
        endpoint_name = "/detect-bot"
        timer = LATENCY_HIST.labels(endpoint=endpoint_name).time() if LATENCY_HIST else None
        check_api_key(request)
        enforce_rate_limit(request, activity.user_id)

        # Convert the activity data into features for the model
        features = extract_features_from_activity(activity.dict())
        
        # Convert to DataFrame with a single row
        features_df = features_to_dataframe(features, feature_order)
        
        scoring_model, model_version = choose_scoring_model()
        # Get prediction (probability of being a bot)
        probas = scoring_model.predict_proba(features_df)
        if probas.shape[1] > 2:
            # Multiclass: compute bot prob as 1 - P(human)
            bot_probability = float(1.0 - probas[0][human_idx])
        else:
            bot_probability = float(probas[0][1])  # Probability of being a bot

        # Unsupervised anomaly probability via IsolationForest (if available)
        anomaly_prob = 0.0
        if unsup_model is not None:
            try:
                # decision_function: positive for normal, negative for anomalies
                df_val = float(unsup_model.decision_function(features_df.values)[0])
            except Exception:
                # Fallback to score_samples if decision_function missing
                df_val = float(unsup_model.score_samples(features_df.values)[0])
            # Map to [0,1]: higher when more anomalous
            anomaly_prob = float(1.0 / (1.0 + np.exp(2.0 * df_val)))

        # Fusion from policy
        policy = load_policy()
        w = policy.get("fusion_weights", {})
        fused_bot_prob = float(
            (w.get("supervised", 0.6) * bot_probability)
            + (w.get("anomaly", 0.2) * anomaly_prob)
            + (w.get("sequence", 0.2) * 0.0)
        )
        low_thr = float(policy.get("risk_thresholds", {}).get("low", 0.45))
        high_thr = float(policy.get("risk_thresholds", {}).get("high", 0.75))
        risk_band = "low" if fused_bot_prob < low_thr else ("medium" if fused_bot_prob < high_thr else "high")

        # Simple explanation using feature importances (tree-based models)
        top_features = []
        try:
            importances = getattr(base_model_for_importance or model, "feature_importances_", None)
            if importances is not None and feature_order is not None:
                pairs = sorted(zip(feature_order, importances), key=lambda x: x[1], reverse=True)[:5]
                for name, imp in pairs:
                    top_features.append({"name": name, "importance": float(imp), "value": float(features.get(name, 0.0))})
        except Exception:
            pass
        
        # Audit log (no sensitive payloads; include IP, endpoint, decisions)
        try:
            log_audit({
                "ts": datetime.utcnow().isoformat() + "Z",
                "endpoint": "/detect-bot",
                "ip": request.client.host if request and request.client else None,
                "user_id": activity.user_id,
                "bot_probability": fused_bot_prob,
                "risk_band": risk_band,
                "supervised_prob": bot_probability,
                "anomaly_prob": anomaly_prob,
                "top_features": top_features,
            })
        except Exception:
            pass

        if REQ_COUNTER: REQ_COUNTER.labels(endpoint=endpoint_name, status="200").inc()
        if timer: timer.observe_duration()
        return {
            "user_id": activity.user_id,
            "is_bot": fused_bot_prob > 0.5,
            "bot_probability": fused_bot_prob,
            "risk_band": risk_band,
            "model_version": model_version,
            "features_used": features,
            "debug": {
                "supervised_prob": bot_probability,
                "anomaly_prob": anomaly_prob,
            },
            "explanations": {
                "top_features": top_features
            }
        }
    except Exception as e:
        if REQ_COUNTER: REQ_COUNTER.labels(endpoint="/detect-bot", status="500").inc()
        if ERR_COUNTER: ERR_COUNTER.labels(endpoint="/detect-bot", type=type(e).__name__).inc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/score-session")
async def score_session(request: Request, payload: SessionPayload):
    try:
        endpoint_name = "/score-session"
        timer = LATENCY_HIST.labels(endpoint=endpoint_name).time() if LATENCY_HIST else None
        check_api_key(request)
        enforce_rate_limit(request, payload.user_id)
        # Build session-level features (includes per-event means; extra fields are ignored by feature_order)
        events = [e.dict() for e in payload.events]
        sess_feats = compute_session_features(events)
        df_row = features_to_dataframe(sess_feats, feature_order)

        # Supervised prob
        scoring_model, model_version = choose_scoring_model()
        probas = scoring_model.predict_proba(df_row)
        if probas.shape[1] > 2:
            sup_prob = float(1.0 - probas[0][human_idx])
        else:
            sup_prob = float(probas[0][1])

        # Unsupervised anomaly
        anomaly_prob = 0.0
        if unsup_model is not None:
            try:
                df_val = float(unsup_model.decision_function(df_row.values)[0])
            except Exception:
                df_val = float(unsup_model.score_samples(df_row.values)[0])
            anomaly_prob = float(1.0 / (1.0 + np.exp(2.0 * df_val)))

        # Sequence score via bigram NLL over request_path sequence
        seq_score = 0.0
        if bigram_model is not None:
            seq = [e.get("request_path", "") for e in events]
            nll = sequence_negative_log_likelihood(bigram_model, seq)
            seq_score = float(nll_to_score(nll, bias=2.0, scale=1.0))

        # Fusion from policy
        policy = load_policy()
        w = policy.get("fusion_weights", {})
        fused = float(
            w.get("supervised", 0.6) * sup_prob
            + w.get("anomaly", 0.2) * anomaly_prob
            + w.get("sequence", 0.2) * seq_score
        )
        low_thr = float(policy.get("risk_thresholds", {}).get("low", 0.45))
        high_thr = float(policy.get("risk_thresholds", {}).get("high", 0.75))
        risk_band = "low" if fused < low_thr else ("medium" if fused < high_thr else "high")

        # Explanation (top features)
        top_features = []
        try:
            importances = getattr(base_model_for_importance or model, "feature_importances_", None)
            if importances is not None and feature_order is not None:
                pairs = sorted(zip(feature_order, importances), key=lambda x: x[1], reverse=True)[:5]
                for name, imp in pairs:
                    top_features.append({"name": name, "importance": float(imp), "value": float(sess_feats.get(name, 0.0))})
        except Exception:
            pass

        # Audit log
        try:
            log_audit({
                "ts": datetime.utcnow().isoformat() + "Z",
                "endpoint": "/score-session",
                "ip": request.client.host if request and request.client else None,
                "user_id": payload.user_id,
                "session_id": payload.session_id,
                "bot_probability": fused,
                "risk_band": risk_band,
                "supervised_prob": sup_prob,
                "anomaly_prob": anomaly_prob,
                "sequence_score": seq_score,
                "top_features": top_features,
            })
        except Exception:
            pass

        if REQ_COUNTER: REQ_COUNTER.labels(endpoint=endpoint_name, status="200").inc()
        if timer: timer.observe_duration()
        return {
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "is_bot": fused > 0.5,
            "bot_probability": fused,
            "risk_band": risk_band,
            "model_version": model_version,
            "debug": {
                "supervised_prob": sup_prob,
                "anomaly_prob": anomaly_prob,
                "sequence_score": seq_score,
            },
            "explanations": {
                "top_features": top_features
            },
        }
    except Exception as e:
        if REQ_COUNTER: REQ_COUNTER.labels(endpoint="/score-session", status="500").inc()
        if ERR_COUNTER: ERR_COUNTER.labels(endpoint="/score-session", type=type(e).__name__).inc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    if REQ_COUNTER is None:
        raise HTTPException(status_code=501, detail="Metrics not available")
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/explain-activity")
async def explain_activity(request: Request, activity: UserActivity, top_n: int = 8):
    check_api_key(request)
    enforce_rate_limit(request, activity.user_id)
    if shap is None:
        raise HTTPException(status_code=501, detail="SHAP not available on server")
    if base_model_for_importance is None or feature_order is None:
        raise HTTPException(status_code=500, detail="Model or feature order not loaded")
    try:
        feats = extract_features_from_activity(activity.dict())
        X = features_to_dataframe(feats, feature_order)
        explainer = shap.TreeExplainer(base_model_for_importance)
        sv = explainer.shap_values(X)
        # For multiclass, aggregate absolute values across classes
        if isinstance(sv, list):
            contrib = np.mean([np.abs(sv_k)[0] for sv_k in sv], axis=0)
        else:
            contrib = np.abs(sv)[0]
        pairs = sorted(zip(feature_order, contrib, X.iloc[0].tolist()), key=lambda x: x[1], reverse=True)[:top_n]
        img_b64 = None
        # Cache key by features + top_n
        h = hashlib.sha256(json.dumps({"f": activity.dict(), "n": top_n}, sort_keys=True).encode()).hexdigest()
        cached = _cache_get(h)
        if cached:
            img_b64 = cached
        elif plt is not None:
            fig, ax = plt.subplots(figsize=(6, max(2, top_n * 0.4)))
            names = [p[0] for p in pairs][::-1]
            values = [p[1] for p in pairs][::-1]
            ax.barh(names, values, color="#4e79a7")
            ax.set_title("Top SHAP |contributions|")
            ax.set_xlabel("|SHAP value|")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            _cache_set(h, img_b64)
        return {
            "top_contributions": [
                {"name": name, "abs_shap": float(val), "value": float(v)} for name, val, v in pairs
            ],
            "plot_base64": img_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain-session")
async def explain_session(request: Request, payload: SessionPayload, top_n: int = 8):
    check_api_key(request)
    enforce_rate_limit(request, payload.user_id)
    if shap is None:
        raise HTTPException(status_code=501, detail="SHAP not available on server")
    if base_model_for_importance is None or feature_order is None:
        raise HTTPException(status_code=500, detail="Model or feature order not loaded")
    try:
        events = [e.dict() for e in payload.events]
        sess_feats = compute_session_features(events)
        X = features_to_dataframe(sess_feats, feature_order)
        explainer = shap.TreeExplainer(base_model_for_importance)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            contrib = np.mean([np.abs(sv_k)[0] for sv_k in sv], axis=0)
        else:
            contrib = np.abs(sv)[0]
        pairs = sorted(zip(feature_order, contrib, X.iloc[0].tolist()), key=lambda x: x[1], reverse=True)[:top_n]
        img_b64 = None
        h = hashlib.sha256(json.dumps({"s": payload.dict(), "n": top_n}, sort_keys=True).encode()).hexdigest()
        cached = _cache_get(h)
        if cached:
            img_b64 = cached
        elif plt is not None:
            fig, ax = plt.subplots(figsize=(6, max(2, top_n * 0.4)))
            names = [p[0] for p in pairs][::-1]
            values = [p[1] for p in pairs][::-1]
            ax.barh(names, values, color="#e15759")
            ax.set_title("Top SHAP |contributions| (Session)")
            ax.set_xlabel("|SHAP value|")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            _cache_set(h, img_b64)
        return {
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "top_contributions": [
                {"name": name, "abs_shap": float(val), "value": float(v)} for name, val, v in pairs
            ],
            "plot_base64": img_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
