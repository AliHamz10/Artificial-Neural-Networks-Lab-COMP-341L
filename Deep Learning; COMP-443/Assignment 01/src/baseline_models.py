"""
Classical (non-deep) baseline models for COMP-443 Assignment 01.

We keep these deliberately simple but reasonably strong:

- Logistic Regression
- Random Forest

Both are implemented using scikit-learn and operate on flattened
Fashion-MNIST features (784-dimensional vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    confusion: np.ndarray
    report: str


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> BaselineResult:
    """
    Multinomial logistic regression with lbfgs solver.
    """
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=200,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    acc = accuracy_score(y_eval, y_pred)
    conf = confusion_matrix(y_eval, y_pred)
    rep_str: str = str(classification_report(y_eval, y_pred, digits=4))
    return BaselineResult(
        name="Logistic Regression",
        accuracy=acc,
        confusion=conf,
        report=rep_str,
    )


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> BaselineResult:
    """
    Random Forest classifier with a modest number of trees.
    """
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(x_train, y_train)
    # Predict on evaluation data
    y_pred = clf.predict(x_eval)
    acc = accuracy_score(y_eval, y_pred)
    conf = confusion_matrix(y_eval, y_pred)
    rep_str: str = str(classification_report(y_eval, y_pred, digits=4))
    return BaselineResult(
        name="Random Forest",
        accuracy=acc,
        confusion=conf,
        report=rep_str,
    )


def as_dict(result: BaselineResult) -> Dict[str, object]:
    return {
        "name": result.name,
        "accuracy": float(result.accuracy),
        "confusion": result.confusion.tolist(),
        "report": result.report,
    }


__all__ = [
    "BaselineResult",
    "train_logistic_regression",
    "train_random_forest",
    "as_dict",
]

