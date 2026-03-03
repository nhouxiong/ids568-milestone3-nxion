"""Register 5 experiment runs using the existing register.py logic."""
import json
import os
from pathlib import Path
from register import register_model

os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

runs = [
    ("9c60bc1503624ccdb1004f0b4629569f", 0.9667, 0.9666),
    ("b3f31886571a4862bb079a0dbcd83ec0", 0.9667, 0.9666),
    ("126db7c05d40413081b40c3e62545ab0", 0.9333, 0.9333),
    ("7646d12e9c234b88844d17219ea9a701", 0.9333, 0.9333),
    ("f565aa2970e54626983b1eb20fc9037e", 0.9000, 0.8997),
]

for run_id, acc, f1 in runs:
    metrics = {"accuracy": acc, "f1_score": f1}
    version, stage = register_model(
        run_id=run_id,
        model_name="sklearn_classifier",
        metrics=metrics,
    )
    print(f"  Registered v{version} -> {stage}\n")
