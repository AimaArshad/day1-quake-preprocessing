import numpy as np

from quakes.evaluate import (best_threshold, classification_report_dict,
                             regression_report)


def test_regression_report():
    r = regression_report([1, 2, 3], [1, 2, 4])
    assert {"mae", "rmse", "r2"} <= set(r)
    assert abs(r["mae"] - 1 / 3) < 1e-9
    assert abs(r["rmse"] - np.sqrt(1 / 3)) < 1e-9


def test_classification_report():
    r = classification_report_dict([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert r["accuracy"] == 0.75 and r["recall"] == 1.0
    assert abs(r["precision"] - 2 / 3) < 1e-9 and r["roc_auc"] == 1.0


def test_best_threshold():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.42, 0.62, 0.9])
    assert 0.42 < best_threshold(y, proba) <= 0.62
