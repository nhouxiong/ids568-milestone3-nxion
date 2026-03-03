# Model Lineage Report

**Project:** IDS568 Milestone 3 — MLOps Training Pipeline  
**Model:** sklearn_classifier (RandomForestClassifier)  
**Experiment:** ids568-milestone3  
**Date:** 2026-03-02  


![Successful Test](/Screenshots/Success_Test_Airflow.png)

---

## 1. Run Comparisons and Analysis

Seven experiments were conducted using the Iris classification dataset (150 samples, 4 features). All runs used identical data with a stratified 80/20 train-test split (seed=42), isolating the effect of hyperparameter variation. Five successful runs with distinct configurations were selected for analysis.

![Comparing Test](/Screenshots/Comparing_Test_MLflow.pngscreenshot.png)

### Results Summary

| Run Name | n_estimators | max_depth | min_samples_split | Accuracy | F1 (weighted) | Stage |
|----------|-------------|-----------|-------------------|----------|---------------|-------|
| dashing-zebra-693 | 200 | 7 | 2 | 0.9667 | 0.9666 | Production |
| skittish-tern-823 | 300 | 4 | 5 | 0.9667 | 0.9666 | Staging |
| beautiful-bear-860 | 100 | 5 | 2 | 0.9333 | 0.9333 | None |
| rare-donkey-380 | 100 | 5 | 2 | 0.9333 | 0.9333 | None |
| silent-snipe-843 | 50 | 3 | 2 | 0.9000 | 0.8997 | None |

![Versions of tests](/Screenshots/Versions_of_Test_MLflow.png)


### Key Observations

**Estimator count and depth trade-off:** The two top-performing runs (dashing-zebra-693 and skittish-tern-823) both achieved 96.67% accuracy but with different configurations. dashing-zebra used 200 estimators with deeper trees (max_depth=7), while skittish-tern used 300 estimators with shallower trees (max_depth=4) but higher min_samples_split. This shows that the model can reach the same accuracy through either deeper individual trees or more constrained but numerous trees.

![Best Test](/Screenshots/Best_Test.png)

**Diminishing returns at lower complexity:** silent-snipe-843 (50 estimators, max_depth=3) scored noticeably lower at 90.0% accuracy, confirming that the Iris dataset benefits from moderate model complexity. The jump from 50 to 100 estimators improved accuracy by 3.3 percentage points, but going from 100 to 200+ only added another 3.3 points.

**Stability of mid-range configurations:** beautiful-bear-860 and rare-donkey-380 both used 100 estimators with max_depth=5 and achieved identical results (93.33%), demonstrating reproducibility of the training pipeline when given the same hyperparameters.

![Runs](/Screenshots/Runs_MLflow.png)

---

## 2. Production Candidate Justification

**Selected model: dashing-zebra-693 → Production**

Run dashing-zebra-693 was selected for production deployment based on three criteria:

1. **Tied for best accuracy (96.67%)** with skittish-tern-823. Both models correctly classified 29 out of 30 test samples.

2. **More efficient at inference.** With 200 estimators vs. 300, dashing-zebra requires ~33% fewer trees to evaluate at prediction time. For a serving environment, this translates to lower latency.

3. **Simpler regularization.** dashing-zebra uses the default min_samples_split=2 with depth as the primary regularization lever (max_depth=7). This is easier to reason about and tune than the combined constraint of max_depth=4 plus min_samples_split=5 used by skittish-tern.

**Runner-up: skittish-tern-823 → Staging**

Retained in Staging as a fallback. Its shallower trees with higher min_samples_split provide stronger regularization, which may generalize better if the production data distribution is noisier than the training set.

**Remaining runs → None**

The other three runs scored below the production threshold (≥0.90 accuracy AND ≥0.88 F1). They remain in the registry for reference but are not candidates for deployment.

### Registry State

| Version | Run Name | Stage | Transition Path |
|---------|----------|-------|-----------------|
| v3 | dashing-zebra-693 | Production | None → Staging → Production |
| v4 | skittish-tern-823 | Staging | None → Staging |
| v5 | beautiful-bear-860 | None | — |
| v6 | rare-donkey-380 | None | — |
| v7 | silent-snipe-843 | None | — |

---

## 3. Identified Risks and Monitoring Needs

### Small Dataset Risk

The Iris dataset has only 150 samples (120 train, 30 test). With a test set this small, a single misclassification changes accuracy by ~3.3%. The high accuracy numbers (90-97%) should be interpreted cautiously — the model has not been validated on a large, independent holdout set. In production, model performance should be monitored on a continuous stream of labeled data.

### Overfitting Risk

dashing-zebra-693 uses max_depth=7 on a dataset with only 4 features. While Random Forests are resistant to overfitting due to bagging, deep trees on a small dataset could memorize training patterns. If production data has more noise or different class distributions, accuracy may degrade. Monitoring recommendation: track per-class precision and recall weekly, and alert if any class drops below 85%.

### Data Drift Risk

The model assumes the input feature distribution matches the Iris training data. In a real deployment, feature distributions may shift over time. Recommended monitoring: compute Population Stability Index (PSI) on each feature over rolling windows. Trigger retraining if PSI exceeds 0.2.

### Model Staleness

If the underlying data-generating process changes, the model will become stale. Recommended cadence: retrain weekly (aligned with the Airflow DAG schedule) and compare the new model's metrics against the current production model. Only promote if the new model passes the CI quality gate AND does not regress more than 2% on any metric.

### Reproducibility

All runs use a fixed random seed (42) and log the data_version hash. Before any production promotion, verify that retraining with the same parameters produces metrics within ±0.5% of the original run. A larger discrepancy indicates non-determinism that should be investigated.