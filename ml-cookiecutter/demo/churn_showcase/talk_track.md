# Presenter Talk Track (10–15 min)

## 1) Problem framing (1 min)
"We want to proactively identify customers likely to churn so retention teams can act early."

## 2) Architecture framing (2 min)
"This project uses the FTI pattern: Feature pipeline -> Training pipeline -> Inference service."

## 3) Data and feature prep (2–3 min)
- Run `prepare-churn-demo`
- Mention data cleaning (`TotalCharges` casting, missing handling, one-hot encoding)

## 4) Training and experiment traceability (3–4 min)
- Run `train`
- Show run ID and model URI output
- Mention MLflow tracking and model registration

## 5) Inference API (2–3 min)
- Start `serve`
- Send `sample_request.json` to `/predict`
- Explain how this maps to production integration

## 6) Close (1 min)
"This template gives us a reusable baseline for repeatable ML delivery, not a one-off notebook."
