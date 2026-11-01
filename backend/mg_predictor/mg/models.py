"""The five required algorithms, plus helpers for scoring and calibration.

Naive Bayes, Decision Tree, Linear Regression, Logistic Regression, Random Forest.

Linear Regression is a regressor, not a classifier. It is used here as a
"linear probability model": it is fitted on the 0/1 label and its raw output is
treated as a risk score (thresholded to give a class label).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C


class LinearRegressionClassifier(ClassifierMixin, BaseEstimator):
    """Ordinary least squares on a 0/1 target, exposed through a classifier API."""

    def __init__(self, class_weight="balanced"):
        self.class_weight = class_weight

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        weights = compute_sample_weight(self.class_weight, y) if self.class_weight else None
        self.reg_ = LinearRegression().fit(self.scaler_.transform(X), y, sample_weight=weights)
        return self

    def raw_score(self, X):
        """Unbounded regression output (used for ranking / thresholding)."""
        return self.reg_.predict(self.scaler_.transform(np.asarray(X, dtype=float)))

    def predict_proba(self, X):
        p = np.clip(self.raw_score(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.raw_score(X) >= 0.5).astype(int)


def get_models() -> dict:
    """Fresh, unfitted instances of the five algorithms."""
    rs = C.RANDOM_STATE
    return {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=rs
        ),
        "Linear Regression": LinearRegressionClassifier(class_weight="balanced"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=rs),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=rs,
        ),
    }


def positive_score(model, X) -> np.ndarray:
    """Continuous score where higher = more likely MG, for any of the five models."""
    if hasattr(model, "raw_score"):
        return np.asarray(model.raw_score(X))
    return model.predict_proba(X)[:, 1]


class PlattCalibrator:
    """Maps a raw model score to a probability on the *natural* class balance.

    Several models are trained with class weights, so their raw scores are not
    probabilities. Fitting a 1-D logistic regression on out-of-fold scores
    (which keep the dataset's real class balance) gives honest probabilities.
    """

    def fit(self, scores, y):
        s = np.asarray(scores, dtype=float)
        self.mu_ = float(s.mean())
        self.sd_ = float(s.std()) or 1.0
        self.lr_ = LogisticRegression(C=100.0, max_iter=1000)
        self.lr_.fit(((s - self.mu_) / self.sd_).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return self.lr_.predict_proba(((s - self.mu_) / self.sd_).reshape(-1, 1))[:, 1]

    def threshold_at(self, prob: float = 0.5) -> float:
        """Raw-score threshold at which the calibrated probability equals `prob`."""
        a = float(self.lr_.coef_[0, 0])
        b = float(self.lr_.intercept_[0])
        z = (np.log(prob / (1.0 - prob)) - b) / a
        return float(self.mu_ + self.sd_ * z)
