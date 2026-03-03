import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def register_model(
    run_id: str,
    model_name: str,
    metrics: dict,
    staging_threshold: float = 0.85,
    production_threshold_acc: float = 0.90,
    production_threshold_f1: float = 0.88,
):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    accuracy = metrics["accuracy"]
    f1 = metrics["f1_score"]
    data_version = metrics.get("data_version", "unknown")

    print(f"Registering model from run_id={run_id}")

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version
    print(f"Registered {model_name} version {version}")

    client.set_model_version_tag(model_name, version, "accuracy", str(accuracy))
    client.set_model_version_tag(model_name, version, "f1_score", str(f1))
    client.set_model_version_tag(model_name, version, "data_version", str(data_version))
    client.set_model_version_tag(model_name, version, "registered_via", "airflow_pipeline")

    client.update_model_version(
        name=model_name,
        version=version,
        description=f"acc={accuracy:.4f} | f1={f1:.4f} | data={data_version}",
    )

    if accuracy >= production_threshold_acc and f1 >= production_threshold_f1:
        client.transition_model_version_stage(model_name, version, "Staging")
        client.transition_model_version_stage(model_name, version, "Production")
        stage = "Production"
    elif accuracy >= staging_threshold:
        client.transition_model_version_stage(model_name, version, "Staging")
        stage = "Staging"
    else:
        stage = "None"

    print(f"Model v{version} -> {stage}")
    return version, stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sklearn_classifier")
    parser.add_argument("--metrics-file", default="metrics.json")
    parser.add_argument("--run-id-file", default="artifacts/run_id.txt")
    args = parser.parse_args()

    run_id_path = Path(args.run_id_file)
    if not run_id_path.exists():
        print(f"run_id file not found: {run_id_path}")
        sys.exit(1)
    run_id = run_id_path.read_text(encoding="utf-8").strip()

    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    register_model(run_id=run_id, model_name=args.model_name, metrics=metrics)


if __name__ == "__main__":
    main()
