import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

experiment = client.get_experiment_by_name("ids568-milestone3")
runs = client.search_runs(experiment.experiment_id, filter_string="status = 'FINISHED'", order_by=["metrics.accuracy DESC"])

print(f"\n{'Run Name':<28} {'Accuracy':<10} {'F1':<10} {'Run ID'}")
print("-" * 75)
for r in runs:
    name = r.info.run_name
    acc = r.data.metrics.get("accuracy", 0)
    f1 = r.data.metrics.get("f1_weighted", 0)
    print(f"{name:<28} {acc:<10.4f} {f1:<10.4f} {r.info.run_id}")
