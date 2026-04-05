# Churn Showcase Demo Pack

This folder contains ready-to-use assets for a live **customer churn classification** demo.

## Files in this folder

- `sample_request.json` — JSON payload for `POST /predict`.
- `talk_track.md` — a concise presenter script for a 10-15 minute demo.

## Quick live flow

1. Prepare dataset:

```bash
uv run python -m ml_project.cli prepare-churn-demo
```

2. Train model:

```bash
uv run python -m ml_project.cli train \
  --data processed/churn_features.parquet \
  --target Churn \
  --experiment churn_demo
```

3. Start API:

```bash
uv run python -m ml_project.cli serve
```

4. Call endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "content-type: application/json" \
  -d @demo/churn_showcase/sample_request.json
```

## Notes

- Use a model URI from training output (`runs:/<run_id>/model`) and set `INFERENCE_MODEL_URI` in `.env` before running `serve`.
- Keep this folder under version control so anyone on the team can run the same demo flow.
