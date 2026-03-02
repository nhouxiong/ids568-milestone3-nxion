import argparse
import json
import os
import time
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn


def load_manifest(manifest_path: str) -> dict | None:
    p = Path(manifest_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_and_log(
    experiment_name: str,
    manifest_path: str,
    metrics_out: str,
    test_size: float,
    seed: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
) -> dict:
    """
    Train a model, save metrics.json, and log params/metrics/artifacts/model to MLflow.
    """
    print("=" * 50)
    print("Training Model + MLflow Logging")
    print("=" * 50)

    # --- MLflow tracking setup ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"\n🧭 MLFLOW_TRACKING_URI = {tracking_uri}")
    print(f"📌 Experiment = {experiment_name}")

    # --- Load data (Iris) ---
    print("\n📊 Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=seed, stratify=iris.target
    )

    # --- Optional: load preprocess manifest (for data_version lineage) ---
    manifest = load_manifest(manifest_path)
    data_version = manifest.get("data_version") if manifest else None

    # --- Train + log ---
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n🏃 MLflow run_id = {run_id}")

        # --- Persist run_id for downstream steps (Airflow / registry) ---
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        run_id_path = artifacts_dir / "run_id.txt"
        with run_id_path.open("w", encoding="utf-8") as f:
            f.write(run_id)

        print(f"🆔 run_id written to {run_id_path}")

        # Log key params (required by rubric)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("seed", seed)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth if max_depth is not None else "None")
        mlflow.log_param("min_samples_split", min_samples_split)

        if data_version:
            mlflow.log_param("data_version", data_version)
        else:
            mlflow.log_param("data_version", "manifest_not_found")

        print("🏋️ Training RandomForest classifier...")
        start_time = time.time()
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        print("📈 Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Save metrics.json for your validation script
        metrics = {
            "run_id": run_id,
            "accuracy": round(float(accuracy), 4),
            "f1_score": round(float(f1), 4),
            "training_time_seconds": round(float(training_time), 2),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "seed": seed,
            "test_size": test_size,
        }

        metrics_out_path = Path(metrics_out)
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("f1_weighted", float(f1))
        mlflow.log_metric("training_time_seconds", float(training_time))

        # Log artifacts to MLflow
        mlflow.log_artifact(str(metrics_out_path), artifact_path="reports")

        if manifest:
            # log the exact manifest used for provenance
            mlflow.log_artifact(manifest_path, artifact_path="data_lineage")

        # Log model to MLflow
        # This creates a model artifact at `model/`
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\n✅ Training complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"💾 Metrics saved to: {metrics_out}")
        print("📦 Model + metrics logged to MLflow.")

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="ids568-milestone3", help="MLflow experiment name")
    parser.add_argument("--manifest-path", default="data/processed/manifest.json", help="Path to preprocess manifest.json")
    parser.add_argument("--metrics-out", default="metrics.json", help="Where to write metrics.json for validation")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameters (vary these for your 5 MLflow runs)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--no-max-depth", action="store_true", help="Use max_depth=None")
    parser.add_argument("--min-samples-split", type=int, default=2)

    args = parser.parse_args()
    max_depth = None if args.no_max_depth else args.max_depth

    train_and_log(
        experiment_name=args.experiment_name,
        manifest_path=args.manifest_path,
        metrics_out=args.metrics_out,
        test_size=args.test_size,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
    )


if __name__ == "__main__":
    main()