import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("ids568-milestone3")

with mlflow.start_run():
    # training logic
    mlflow.log_param("model_type", "linear_regression")
    mlflow.log_metric("rmse", 0.42)