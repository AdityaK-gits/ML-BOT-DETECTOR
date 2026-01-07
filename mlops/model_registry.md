# Simple Model Registry (Stub)

This stub demonstrates a simple, file-based registry to track model artifacts and versions.
Use a real registry like MLflow in production with experiment tracking, lineage, and governance.

## Layout

- models/
  - v1/
    - bot_detection_model.pkl
    - bot_detection_model_calibrated.pkl
    - feature_list.json
    - label_map.json
  - v2/
    - ...
- registry.json (points "production" to a version)

## Commands

- Promote a model to production by updating `registry.json`:

```json
{
  "production": "v1",
  "canary": "v2",
  "notes": "v2 includes improved simulator and calibration"
}
```

- Set envs for deployment to pick the active version:
  - `MODEL_VERSION=v1`
  - `CANARY_PERCENT=0.1` # optional canary traffic split

In a real setup, wire this to MLflow Model Registry and your CD pipeline.
