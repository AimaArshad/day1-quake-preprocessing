"""Part 2, Task B: an sklearn-style estimator and model factories."""
from sklearn.base import BaseEstimator, RegressorMixin


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Port your vectorized notebook version here.

    fit(X, y) must set: coef_ (array), intercept_ (float), loss_history_ (list
    of the loss at every iteration) and return self.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...


def make_regressor(kind: str = "linear", degree: int = 2, alpha: float = 1.0):
    """Return Pipeline([('prep', build_preprocessor(...)), ('model', estimator)]).

    kinds: 'dummy' (mean baseline), 'linear', 'ridge', 'poly' (Ridge on
    polynomial features of the numeric columns). Unknown kind: raise ValueError.
    First update transform.build_preprocessor to accept degree=1 and add
    PolynomialFeatures(degree, include_bias=False) to the numeric branch
    when degree > 1.
    """
    ...


def make_classifier(kind: str = "logistic", C: float = 1.0, class_weight=None):
    """Return Pipeline([('prep', ...), ('model', estimator)]).

    kinds: 'dummy' (majority class, still needs predict_proba) and 'logistic'.
    """
    ...


