"""
Model Building with Experimentation Tracking
----------------------------------------------
1. Loads train.csv / test.csv from the Hugging Face dataset repo.
2. Builds a preprocessing + XGBoost pipeline.
3. Tunes hyperparameters with RandomizedSearchCV, scoring on F1.
4. Logs every trial's parameters/metrics to MLflow, evaluates the best
   model on the held-out test set, saves it locally, and registers it on
   the Hugging Face Model Hub.
"""

import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402


def load_split(token, filename):
    path = hf_hub_download(
        repo_id=config.DATASET_REPO_ID,
        repo_type="dataset",
        filename=filename,
        token=token,
    )
    return pd.read_csv(path)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    return pipeline


def main():
    token = config.require_token()

    print("Loading train/test splits from the Hub...")
    train_df = load_split(token, config.TRAIN_FILENAME)
    test_df = load_split(token, config.TEST_FILENAME)

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    imbalance_ratio = neg / pos
    print(f"Train class balance -> negative: {neg}, positive: {pos}, "
          f"scale_pos_weight reference: {imbalance_ratio:.2f}")

    pipeline = build_pipeline(X_train)

    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__scale_pos_weight": [1, imbalance_ratio / 2, imbalance_ratio],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    tracking_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir}/mlflow.db")
    mlflow.set_experiment("tourism-wellness-package")

    with mlflow.start_run():
        print("Running hyperparameter search (RandomizedSearchCV, 5-fold, scoring=f1)...")
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params: {best_params}")
        print(f"Best CV F1 score: {search.best_score_:.4f}")

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_f1", search.best_score_)

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(test_metrics)
        print("Test set performance:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

        mlflow.sklearn.log_model(
            best_pipeline, "model", serialization_format="cloudpickle"
        )

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run id: {run_id}")

    deployment_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "deployment")
    )
    os.makedirs(deployment_dir, exist_ok=True)
    model_path = os.path.join(deployment_dir, config.MODEL_ARTIFACT_FILENAME)
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best pipeline: {model_path}")

    from huggingface_hub import HfApi, create_repo

    MODEL_REPO_ID = "Nohafx/tourism-wellness-model"
    api = HfApi(token=token)
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.joblib",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        token=token,
    )
    print(f"Model registered: https://huggingface.co/{MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
