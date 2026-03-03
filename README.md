# IDS568 Milestone 3 — MLOps Training Pipeline

End-to-end ML operations project integrating **Apache Airflow** for orchestration, **MLflow** for experiment tracking and model registry, and **GitHub Actions** for CI/CD-based model governance.


## DAG Idempotency and Lineage Guarantees

### Idempotency

The Airflow DAG (`train_pipeline`) is designed for safe re-execution. The preprocessing step writes to fixed output paths (`data/processed/train.csv` and `data/processed/test.csv`), so re-running the same configuration overwrites the same files deterministically. Each training run creates a new MLflow run with a unique `run_id`, and the register step reads the latest `run_id` from `artifacts/run_id.txt` — ensuring that only the most recent training output gets registered.

The DAG uses `catchup=False` and `max_active_runs=1` (via manual trigger) to prevent overlapping executions that could cause race conditions.

### Lineage

Every MLflow run captures full provenance:

- **Data version**: A SHA-256 hash computed over the processed train/test data and preprocessing config, stored in `data/processed/manifest.json` and logged as an MLflow parameter (`data_version`).
- **Hyperparameters**: All model configuration (n_estimators, max_depth, min_samples_split, seed) logged as MLflow parameters.
- **Metrics**: accuracy, f1_weighted, and training_time_seconds logged as MLflow metrics.
- **Artifacts**: The serialized model, metrics.json report, and preprocessing manifest are all stored as MLflow artifacts.

This means any registered model version can be traced back to the exact data, code, and configuration that produced it.

---

## CI-Based Model Governance

The GitHub Actions workflow (`.github/workflows/train_and_validate.yml`) enforces automated quality gates on every push to `main` and on pull requests.

### Pipeline Steps

1. **Install dependencies** from `requirements.txt`
2. **Run preprocessing** to generate the train/test split
3. **Train the model** and log parameters, metrics, and artifacts to MLflow
4. **Validate metrics** against quality thresholds — the pipeline **fails** if any metric is below the threshold

### Quality Gate Thresholds

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| Accuracy | ≥ 0.90 | Higher is better |
| F1 Score | ≥ 0.85 | Higher is better |

If validation fails, the pipeline exits with code 1, blocking the merge. This ensures no model that fails to meet minimum quality standards can be deployed without explicit human override.

### Governance Principles

- **Reproducibility**: CI runs in a clean Ubuntu environment with pinned dependencies.
- **Auditability**: Metrics and validation reports are uploaded as GitHub Actions artifacts with 90-day retention.
- **Separation of concerns**: Model registration only happens through the Airflow pipeline or manual approval, not automatically in CI.

---

## Experiment Tracking Methodology

We use MLflow to track all training experiments under the `ids568-milestone3` experiment. Each run logs:

- **Parameters**: model_type, n_estimators, max_depth, min_samples_split, seed, test_size, data_version
- **Metrics**: accuracy, f1_weighted, training_time_seconds
- **Artifacts**: trained model (sklearn flavor), metrics.json, preprocessing manifest

Five experiments were run with systematically varied hyperparameters (n_estimators: 50–300, max_depth: 3–7 or None, min_samples_split: 2–5). The best-performing model was promoted through the MLflow Model Registry stages: None → Staging → Production.

Run comparison and production candidate justification are documented in `lineage_report.md`.

---

## Setup and Execution Instructions

### Prerequisites

- Docker Desktop installed and running
- Git
- Python 3.11+ (for local development only)

### Quick Start

```bash
# Clone the repo
git clone https://github.com/nhouxiong/ids568-milestone3-nxion.git
cd ids568-milestone3-nxion

# Start Airflow + MLflow
cd airflow-local
docker compose up

# Wait 2-3 minutes for initialization, then open:
#   Airflow: http://localhost:8080  (login: airflow / airflow)
#   MLflow:  http://localhost:5001
```

### Running the Pipeline

1. Open the Airflow UI at http://localhost:8080
2. Find `train_pipeline` and toggle it ON
3. Click Trigger to start a run
4. Watch all 3 tasks complete: preprocess_data → train_model → register_model

### Running Experiments Manually

```bash
# From airflow-local/ directory
docker compose exec airflow-worker bash -c "cd /opt/airflow/project && \
  MLFLOW_TRACKING_URI=http://mlflow:5000 python src/train.py \
  --experiment-name ids568-milestone3 \
  --n-estimators 200 --max-depth 5"
```

### Stopping

```bash
cd airflow-local
docker compose down
```

---

## Retry Strategies and Failure Handling

### DAG Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| `retries` | 2 (default) | Maximum retry attempts per task |
| `retry_delay` | 2 minutes | Wait time between retries |
| `retry_exponential_backoff` | True | Progressively longer waits |
| `max_retry_delay` | 15 minutes | Cap on backoff growth |
| `execution_timeout` | 1 hour | Hard timeout per task |
| `on_failure_callback` | Custom function | Logs structured alert with dag_id, task_id, exception |

### Per-Task Overrides

- **preprocess_data**: 2 retries, 1-minute delay — data issues fail fast
- **train_model**: 3 retries, 2-minute delay — allows for transient resource issues
- **register_model**: 2 retries, 1-minute delay — MLflow server blips are usually brief

### Failure Callback

The `on_failure_callback` logs a structured message including the DAG ID, task ID, run ID, and attempt number. In production, this would forward to Slack or PagerDuty.

---

## Monitoring and Alerting Recommendations

### What to Monitor

- **Airflow**: Task success/failure rates, task duration trends, scheduler lag
- **MLflow**: Accuracy and F1 trends across runs — alert if metrics drop below quality gate thresholds for 2 consecutive runs
- **Infrastructure**: Docker container health, disk usage for MLflow artifact storage, PostgreSQL connection pool

### Recommended Stack

- Prometheus + Grafana for metric dashboards
- PagerDuty or Slack webhooks for alerting
- Airflow's built-in health check endpoints for liveness probes

### Data Drift Monitoring

Track feature distributions over time. If Population Stability Index (PSI) exceeds 0.2 for any feature, trigger a retraining alert.

---

## Rollback Procedures

### Model Rollback

If a newly promoted model causes issues in production:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Demote bad model
client.transition_model_version_stage("sklearn_classifier", bad_version, "Archived")

# Promote previous good version back to Production
client.transition_model_version_stage("sklearn_classifier", good_version, "Production")
```

### DAG Rollback

If a DAG code change causes failures:

1. Revert the Git commit: `git revert HEAD`
2. Clear failed task instances in Airflow UI or CLI
3. Re-trigger the DAG

### Data Rollback

Each run records a `data_version` hash. To reproduce a previous model, use the same preprocessing seed and configuration from the manifest.json artifact logged with that run.