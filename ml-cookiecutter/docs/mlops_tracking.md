# MLOps & Tracking with MLflow

This template uses **MLflow** as the central nervous system for experiment tracking and model management.

## 📊 Experiment Tracking
Every time you run the Training Pipeline, a new "Run" is created in MLflow.
- **Parameters:** Learning rate, n_estimators, etc.
- **Metrics:** Accuracy, F1-Score, RMSE, etc.
- **Artifacts:** The serialized model file (`.pkl`), feature importance plots.

## 🗃️ Model Registry
Instead of passing file paths like `model_v1.pkl`, we use the **MLflow Model Registry**.
- Models are referenced by name and alias: `models:/my_model@latest`.
- The training pipeline updates the configured alias after registration, allowing deployments to change model versions without changing API code.

## 🐳 Docker Stack
The provided `docker-compose.yml` spins up:
1. **MLflow Tracking Server:** The UI and API for logging.
2. **Postgres DB:** Stores the metadata (metrics/params).
3. **Artifact Store:** Local directory where the actual model files live.

### Accessing the UI
Once running, you can view your experiments at:
`http://localhost:5000`
