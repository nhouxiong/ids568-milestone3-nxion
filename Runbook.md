# Train Pipeline Runbook

## Pipeline Overview

- **DAG ID:** train_pipeline
- **Schedule:** Daily at 2:00 AM UTC
- **Owner:** ML Platform Team
- **Slack Channel:** #ml-pipeline-alerts

## Normal Operation

### Triggering a Manual Run
```bash
airflow dags trigger train_pipeline --conf '{"learning_rate": 0.01}'
```

### Monitoring

- Airflow UI: http://airflow.internal:8080
- MLflow UI: http://mlflow.internal:5000
- Grafana Dashboard: http://grafana.internal:3000/d/ml-pipeline

## Incident Response

### Task Failed After Retries

1. Check Airflow logs:
   ```bash
   airflow tasks logs train_pipeline train_model 2024-01-15
   ```

2. Identify failure cause (OOM, data issue, network)

3. If data issue, clear upstream tasks:
   ```bash
   airflow tasks clear train_pipeline \
       --task-regex 'preprocess_data' \
       --start-date 2024-01-15 \
       --downstream --yes
   ```

4. If infrastructure issue, wait for resolution and retry:
   ```bash
   airflow tasks clear train_pipeline \
       --task-regex 'train_model' \
       --start-date 2024-01-15 --yes
   ```

### Model Quality Regression

1. Verify current production model:
   ```bash
   python scripts/evaluate_production_model.py
   ```

2. If confirmed regression, rollback:
   ```bash
   python rollback.py --model production-classifier --version 4 \
       --reason "Accuracy dropped from 0.95 to 0.82"
   ```

3. Investigate root cause before next training run

### Complete Pipeline Failure

1. Stop the DAG to prevent retries:
   ```bash
   airflow dags pause train_pipeline
   ```

2. Investigate and fix root cause

3. Clear all failed tasks:
   ```bash
   airflow tasks clear train_pipeline \
       --start-date 2024-01-15 \
       --end-date 2024-01-15 --yes
   ```

4. Resume the DAG:
   ```bash
   airflow dags unpause train_pipeline
   ```

## Escalation Path

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| P1 (Production down) | 15 min | Page on-call, notify manager |
| P2 (Degraded) | 1 hour | Slack alert, on-call reviews |
| P3 (Non-urgent) | Next business day | Create ticket |