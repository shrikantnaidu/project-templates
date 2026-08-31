# Project Directory Structure

```text
├── docs/                   # 👈 You are here (Core concepts & notes)
├── src/ml_project/          # Source code
│   ├── api.py               # FastAPI Inference Layer
│   ├── cli.py               # Universal CLI
│   ├── config.py            # Pydantic Settings
│   ├── data/                # Data Loaders (Pandas/Polars)
│   ├── features/            # Feature Processors (Base classes)
│   ├── models/              # Training & Evaluation logic
│   └── pipelines/           # FTI Implementation (Feature, Training, Inference)
├── tests/                   # Pytest suite
├── data/                    # Local data storage (Gitignored)
│   ├── raw/                 # Immutable source data
│   └── processed/           # Engineereed features (Parquet)
├── models/                  # Local model cache
├── notebooks/               # EDA & Prototyping
├── Dockerfile               # Production build
├── docker-compose.yml       # Dev/Prod Stack (MLflow + API)
├── pyproject.toml           # Modern Build System (uv)
└── Makefile                 # Developer shortcuts
```

## Key Paths
- **Raw Data:** Place your CSVs in `data/raw/`
- **Trained Models:** Automatically saved to `models/` or MLflow.
- **Logs:** Default to stdout, can be configured to write to files in `ml_project/logging.py`.
