"""Training and evaluation helpers for SVM models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_generator import generate_dataset


@dataclass(frozen=True)
class ModelBundle:
    linear_model: Pipeline
    selected_model: Pipeline
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float | str = "scale",
    degree: int = 3,
) -> Pipeline:
    """Train a scaled SVC model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    degree=degree,
                ),
            ),
        ]
    ).fit(X, y)


def train_svm_models(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float | str = "scale",
    degree: int = 3,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ModelBundle:
    """Train a reference linear model and the selected interactive model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    linear_model = train_svm(X_train, y_train, kernel="linear", C=C, gamma="scale")
    selected_model = train_svm(X_train, y_train, kernel=kernel, C=C, gamma=gamma, degree=degree)
    return ModelBundle(linear_model, selected_model, X_train, X_test, y_train, y_test)


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    """Return common classification metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def get_support_vectors(model: Pipeline) -> np.ndarray:
    """Return support vectors in the original feature scale."""
    scaler = model.named_steps["scaler"]
    svc = model.named_steps["svc"]
    return scaler.inverse_transform(svc.support_vectors_)


if __name__ == "__main__":
    X_demo, y_demo = generate_dataset()
    bundle = train_svm_models(X_demo, y_demo)
    metrics = evaluate_model(bundle.selected_model, bundle.X_test, bundle.y_test)
    print({key: value for key, value in metrics.items() if key != "confusion_matrix"})
    print(metrics["confusion_matrix"])
