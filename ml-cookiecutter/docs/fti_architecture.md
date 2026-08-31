# FTI Architecture Overview

This project follows the **Feature-Training-Inference (FTI)** architecture, a modern design pattern for building scalable and maintainable Machine Learning systems.

## 🏗️ The Three Pillars

### 1. Feature Pipeline
The foundation of the system. Its job is to turn raw data into "knowledge" (features).
- **Input:** Raw data (CSV, SQL, API, etc.)
- **Logic:** Cleaning, encoding, scaling, and feature engineering.
- **Output:** Processed features (often stored in a Feature Store or Parquet files).
- **Core Component:** `src/ml_project/pipelines/feature.py`

### 2. Training Pipeline
The laboratory where models are created.
- **Input:** Engineered features from the Feature Pipeline.
- **Logic:** Model selection, hyperparameter tuning, and validation.
- **Integration:** Hooks into **MLflow** to log every experiment, metric, and artifact.
- **Output:** A registered model in the Model Registry.
- **Core Component:** `src/ml_project/pipelines/training.py`

### 3. Inference Pipeline
The production front-end that serves users.
- **Input:** Real-time user data.
- **Logic:** Loads the "best" model from the Registry and applies the same transformations used during training.
- **Output:** Predictions.
- **Core Component:** `src/ml_project/pipelines/inference.py`

## 🚀 Why FTI?
1. **Decoupling:** You can update your training logic without touching the API code.
2. **Consistency:** Using the same feature logic in both Training and Inference prevents "Training-Serving Skew".
3. **Scalability:** Each pipeline can run on different hardware (e.g., Training on GPUs, Inference on lightweight CPUs).
