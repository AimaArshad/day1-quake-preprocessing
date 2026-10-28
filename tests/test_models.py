import numpy as np

from quakes.models import GDLinearRegression, make_classifier, make_regressor


def test_gd_linear_regression_recovers_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    w = np.array([2.0, -1.0, 0.5])
    y = X @ w + 3 + rng.normal(0, 0.01, 200)
    m = GDLinearRegression(lr=0.1, n_iter=2000).fit(X, y)
    assert np.allclose(m.coef_, w, atol=0.05) and abs(m.intercept_ - 3) < 0.05
    assert m.predict(X).shape == (200,)
    assert m.loss_history_[-1] < m.loss_history_[0]


def test_regressor_factory(toy_df):
    X, y = toy_df.drop(columns=["mag", "big_quake"]), toy_df["mag"]
    for kind in ["dummy", "linear", "ridge", "poly"]:
        pipe = make_regressor(kind)
        assert pipe.fit(X, y).predict(X).shape == (len(X),)


def test_classifier_factory(toy_df):
    X, y = toy_df.drop(columns=["mag", "big_quake"]), toy_df["big_quake"]
    for kind in ["dummy", "logistic"]:
        pipe = make_classifier(kind).fit(X, y)
        assert pipe.predict_proba(X).shape == (len(X), 2)
