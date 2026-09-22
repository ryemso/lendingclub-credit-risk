"""Portfolio refactor of the original LendingClub Colab analysis.

This module preserves the main analytical choices found in the original
notebook while removing Colab-specific paths and notebook-only code.

Raw data is intentionally not included in this repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


DATA_PATH = Path("data/accepted_2007_to_2018Q4.csv")

COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
    "loan_status",
    "pub_rec_bankruptcies",
    "delinq_2yrs",
    "inq_last_6mths",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load only the columns used in the original notebook."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place the LendingClub CSV under data/ before running."
        )
    return pd.read_csv(path, usecols=COLUMNS, low_memory=False)


def clean_target_and_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the core target definition and missing-value rules."""
    data = df.copy()

    data = data[
        data["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])
    ].copy()

    data["loan_status_binary"] = np.where(
        data["loan_status"].eq("Fully Paid"), 0, 1
    )

    data = data[data["dti"] >= 0].copy()
    data = data.drop(columns=["emp_title"])

    data.loc[
        data["annual_inc"].eq(0) & data["dti"].isna(),
        "dti",
    ] = 0
    data = data.dropna(subset=["dti"])

    data["inq_last_6mths"] = data["inq_last_6mths"].fillna(0)

    data.loc[
        data["revol_util"].isna() & data["revol_bal"].eq(0),
        "revol_util",
    ] = 0
    data = data[
        ~(data["revol_util"].isna() & data["revol_bal"].ne(0))
    ].copy()

    data = data.dropna(subset=["pub_rec_bankruptcies"])

    return data


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create the principal engineered features used in the notebook."""
    data = df.copy().drop(columns=["emp_length", "grade", "loan_status"])

    sub_grade_mapping = {
        grade: i for i, grade in enumerate(sorted(data["sub_grade"].unique()))
    }
    data["sub_grade"] = data["sub_grade"].map(sub_grade_mapping)

    data["term"] = (
        data["term"].str.replace(" months", "", regex=False).astype(int).eq(60)
    ).astype(int)

    data["home_ownership"] = data["home_ownership"].replace(
        ["ANY", "NONE", "OTHER"], "OTHER"
    )

    data["fico_avg"] = (
        data["fico_range_low"] + data["fico_range_high"]
    ) / 2

    data["installment_to_loan"] = (
        data["installment"] / data["loan_amnt"]
    )
    data["acc_ratio"] = data["open_acc"] / data["total_acc"]

    data = data.drop(
        columns=[
            "fico_range_low",
            "fico_range_high",
            "installment",
            "loan_amnt",
            "open_acc",
            "total_acc",
        ]
    )

    categorical = ["home_ownership", "verification_status", "purpose"]
    data = pd.get_dummies(data, columns=categorical, drop_first=True)

    for col in ["annual_inc", "dti", "revol_bal", "revol_util", "fico_avg"]:
        data[col] = np.log1p(data[col])

    return data


def split_and_scale(df: pd.DataFrame):
    """Use a stratified split and fit the scaler on training data only."""
    X = df.drop(columns=["loan_status_binary"])
    y = df["loan_status_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scale_cols = [
        col
        for col in [
            "sub_grade",
            "annual_inc",
            "delinq_2yrs",
            "inq_last_6mths",
            "pub_rec_bankruptcies",
            "fico_avg",
            "int_rate",
            "dti",
            "revol_bal",
            "revol_util",
        ]
        if col in X_train.columns
    ]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test.loc[:, scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test


def resample(X_train, y_train, method: str = "smote"):
    """Apply representative resampling methods from the original notebook."""
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "rus":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("method must be 'smote' or 'rus'")

    return sampler.fit_resample(X_train, y_train)


def candidate_models():
    """Representative models from the original model-comparison section."""
    return {
        "logistic_regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
        ),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Return comparable binary-classification metrics."""
    pred = model.predict(X_test)

    result = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        result["roc_auc"] = roc_auc_score(y_test, proba)

    return result


def run_comparison(resampling: str = "smote") -> pd.DataFrame:
    """Run the cleaned comparison pipeline end to end."""
    data = feature_engineering(clean_target_and_missing_values(load_data()))
    X_train, X_test, y_train, y_test = split_and_scale(data)
    X_resampled, y_resampled = resample(X_train, y_train, resampling)

    rows = []
    for name, model in candidate_models().items():
        model.fit(X_resampled, y_resampled)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows).sort_values(
        ["f1", "recall"], ascending=False
    )


if __name__ == "__main__":
    print(run_comparison("smote"))
