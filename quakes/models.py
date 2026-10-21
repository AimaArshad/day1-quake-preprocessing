"""Part 2, Task B: an sklearn-style estimator and model factories."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from . import config
from .transform import build_preprocessor


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Port your vectorized notebook version here.

    fit(X, y) must set: coef_ (array), intercept_ (float), loss_history_ (list
    of the loss at every iteration) and return self.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            err = X @ w + b - y
            self.loss_history_.append(float(np.mean(err ** 2)))  # MSE this iteration
            w -= self.lr * (2 / n) * (X.T @ err)
            b -= self.lr * (2 / n) * err.sum()
        self.coef_ = w
        self.intercept_ = float(b)
        return self   

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_    
        


def make_regressor(kind: str = "linear", degree: int = 2, alpha: float = 1.0):
    """Return Pipeline([('prep', build_preprocessor(...)), ('model', estimator)]).

    kinds: 'dummy' (mean baseline), 'linear', 'ridge', 'poly' (Ridge on
    polynomial features of the numeric columns). Unknown kind: raise ValueError.
    First update transform.build_preprocessor to accept degree=1 and add
    PolynomialFeatures(degree, include_bias=False) to the numeric branch
    when degree > 1.
    """
    if kind == "dummy":
        est, deg = DummyRegressor(strategy="mean"), 1
    elif kind == "linear":
        est, deg = LinearRegression(), 1
    elif kind == "ridge":
        est, deg = Ridge(alpha=alpha), 1
    elif kind == "poly":
        est, deg = Ridge(alpha=alpha), degree
    else:
        raise ValueError(f"Unknown regressor kind '{kind}'")
    return Pipeline([("prep", build_preprocessor("robust", degree=deg)),
                     ("model", est)])
    


def make_classifier(kind: str = "logistic", C: float = 1.0, class_weight=None):
    """Return Pipeline([('prep', ...), ('model', estimator)]).

    kinds: 'dummy' (majority class, still needs predict_proba) and 'logistic'.
    """
    if kind == "dummy":
        est = DummyClassifier(strategy="prior")   # predicts majority, proba = class priors
    elif kind == "logistic":
        est = LogisticRegression(C=C, class_weight=class_weight,
                                 max_iter=1000, random_state=config.SEED)
    else:
        raise ValueError(f"Unknown classifier kind '{kind}'")
    return Pipeline([("prep", build_preprocessor("robust")), ("model", est)])

