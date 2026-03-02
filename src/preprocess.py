import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessManifest:
    data_source: str
    created_utc: str
    seed: int
    test_size: float
    stratify: bool
    n_rows_total: int
    n_rows_train: int
    n_rows_test: int
    n_features: int
    feature_names: list
    target_name: str
    target_labels: list
    data_version: str  # sha256 over processed content + config


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def compute_data_version(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict
) -> str:
    """
    Deterministic version hash from processed data + key preprocess config.
    This is what you log to MLflow as `data_version`.
    """
    # Stable CSV bytes (sorted columns, fixed float format)
    train_bytes = train_df.sort_index(axis=1).to_csv(index=False, float_format="%.10g").encode("utf-8")
    test_bytes = test_df.sort_index(axis=1).to_csv(index=False, float_format="%.10g").encode("utf-8")

    config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    return sha256_bytes(train_bytes + b"\n" + test_bytes + b"\n" + config_bytes)


def load_and_preprocess(
    seed: int,
    test_size: float,
    scale: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, PreprocessManifest]:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()  # includes target
    target_col = iris.target.name

    # Minimal "cleaning" step (Iris is already clean, but this makes it explicit)
    df = df.dropna().reset_index(drop=True)

    stratify = True  # good practice for classification
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col] if stratify else None,
    )

    # Feature scaling (optional, but good for demonstrating preprocess logic)
    if scale:
        feature_cols = [c for c in df.columns if c != target_col]
        scaler = StandardScaler()

        train_features = scaler.fit_transform(train_df[feature_cols])
        test_features = scaler.transform(test_df[feature_cols])

        train_df_scaled = pd.DataFrame(train_features, columns=feature_cols)
        test_df_scaled = pd.DataFrame(test_features, columns=feature_cols)

        train_df_scaled[target_col] = train_df[target_col].to_numpy()
        test_df_scaled[target_col] = test_df[target_col].to_numpy()

        train_df, test_df = train_df_scaled, test_df_scaled

    config = {"seed": seed, "test_size": test_size, "scale": scale, "dataset": "sklearn.load_iris"}
    data_version = compute_data_version(train_df, test_df, config)

    manifest = PreprocessManifest(
        data_source="sklearn.datasets.load_iris",
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        test_size=test_size,
        stratify=stratify,
        n_rows_total=len(df),
        n_rows_train=len(train_df),
        n_rows_test=len(test_df),
        n_features=len(df.columns) - 1,
        feature_names=[c for c in df.columns if c != target_col],
        target_name=target_col,
        target_labels=[str(x) for x in sorted(df[target_col].unique().tolist())],
        data_version=data_version,
    )
    return train_df, test_df, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for processed data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-scale", action="store_true", help="Disable feature scaling")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = not args.no_scale

    train_df, test_df, manifest = load_and_preprocess(
        seed=args.seed,
        test_size=args.test_size,
        scale=scale,
    )

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    manifest_path = out_dir / "manifest.json"

    # Idempotent writes: overwrite the same target paths each run
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)

    print("✅ Preprocess complete")
    print(f"Train: {train_path} ({len(train_df)} rows)")
    print(f"Test : {test_path} ({len(test_df)} rows)")
    print(f"Manifest: {manifest_path}")
    print(f"data_version: {manifest.data_version}")


if __name__ == "__main__":
    main()