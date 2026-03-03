from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

def on_failure_callback(context):
    ti = context.get("task_instance")
    print(
        f"Task failed: dag={ti.dag_id} task={ti.task_id} "
        f"run_id={context.get('run_id')} try={ti.try_number}"
    )

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": on_failure_callback,
}

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="Preprocess -> Train -> Register",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "ids568"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/preprocess.py "
            "--out-dir data/processed "
            "--seed 42 "
            "--test-size 0.2"
        ),
        retries=2,
        retry_delay=timedelta(minutes=1),
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/train.py "
            "--experiment-name ids568-milestone3 "
            "--manifest-path data/processed/manifest.json "
            "--metrics-out metrics.json "
            "--n-estimators 100 "
            "--max-depth 5"
        ),
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        retries=3,
        retry_delay=timedelta(minutes=2),
    )

    register = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/register.py "
            "--model-name sklearn_classifier "
            "--metrics-file metrics.json "
            "--run-id-file artifacts/run_id.txt"
        ),
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        retries=2,
        retry_delay=timedelta(minutes=1),
    )

    preprocess >> train >> register
