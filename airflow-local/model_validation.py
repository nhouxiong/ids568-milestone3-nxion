import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("ids568-milestone3")

with mlflow.start_run():
    # validation logic
    mlflow.log_metric("mae", 0.31)