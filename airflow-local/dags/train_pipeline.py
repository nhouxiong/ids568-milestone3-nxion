from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

import mlflow

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# IMPORTANT: inside docker compose network, use service name "mlflow"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "train_pipeline")

def preprocess_data(**context):
    """Task 1: Load and preprocess training data."""
    import pandas as pd

    df = pd.read_csv("/data/raw/dataset.csv")
    df_clean = df.dropna().reset_index(drop=True)

    output_path = "/data/processed/train.csv"
    df_clean.to_csv(output_path, index=False)

    return {"data_path": output_path, "n_samples": len(df_clean)}

def train_model(**context):
    """Task 2: Train model with MLflow tracking."""
    ti = context["ti"]
    data_info = ti.xcom_pull(task_ids="preprocess_data")

    # Set these inside the task so it's always correct at runtime
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "train_pipeline"))

    with mlflow.start_run():

    # If you use sklearn logging, ensure mlflow + sklearn is installed in Airflow container
    import mlflow.sklearn

    # --- Replace these with your real implementations ---
    def train_sklearn_model(data_path):
        # TODO: implement
        from sklearn.linear_model import LogisticRegression
        import pandas as pd

        df = pd.read_csv(data_path)
        # dummy example: assumes last column is label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        return model

    def evaluate_model(model):
        # TODO: implement
        return 0.91
    # -----------------------------------------------

    with mlflow.start_run(run_name="train_model") as run:
        mlflow.log_param("data_path", data_info["data_path"])
        mlflow.log_param("n_samples", data_info["n_samples"])
        mlflow.log_param("learning_rate", 0.01)

        model = train_sklearn_model(data_info["data_path"])
        accuracy = evaluate_model(model)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    return {"run_id": run_id, "accuracy": accuracy}

def register_model(**context):
    """Task 3: Register model to MLflow registry."""
    ti = context["ti"]
    train_info = ti.xcom_pull(task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model_uri = f"runs:/{train_info['run_id']}/model"

    # NOTE:
    # MLflow *Model Registry* requires a database backend that supports it.
    # SQLite often causes issues for registry usage in real setups.
    result = mlflow.register_model(model_uri=model_uri, name="production-classifier")

    return {"model_name": result.name, "version": result.version}

with DAG(
    "train_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    preprocess >> train >> register