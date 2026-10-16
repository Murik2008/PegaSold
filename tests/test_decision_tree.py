"""Correctness tests for our DecisionTree, against scikit-learn.

Run with:
    pytest tests/ -v

scikit-learn is used here only as a reference to check our implementation
against, which is what the project brief allows it for.
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree

DATA = "data/processed"
SUBSET = 2000          # enough to be meaningful, small enough to stay fast
SEED = 42


def _load(target):
    """A seeded subsample of the real project data, or skip if it is absent."""
    try:
        X = np.load(f"{DATA}/X_train.npy")
        y = np.load(f"{DATA}/{target}")
    except FileNotFoundError:
        pytest.skip(f"{DATA} not built — run `python3 -m src.data_prep` first")
    idx = np.random.default_rng(SEED).choice(len(X), SUBSET, replace=False)
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def clf_data():
    """Features and the premium/standard tier label."""
    return _load("tier_train.npy")


@pytest.fixture(scope="module")
def reg_data():
    """Features and the log-price target."""
    return _load("y_train_log.npy")


@pytest.mark.parametrize("crit", ["gini", "entropy"])
def test_root_split_matches(clf_data, crit):
    """Our first split should be the same feature and threshold as sklearn's."""
    X, y = clf_data
    mine = DecisionTree(criterion=crit, max_depth=3).fit(X, y)
    ref = DecisionTreeClassifier(criterion=crit, max_depth=3, random_state=0).fit(X, y)
    assert mine.root.feature_index == ref.tree_.feature[0]
    assert mine.root.threshold == pytest.approx(ref.tree_.threshold[0], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("d", [1, 2, 3, 5, None])
def test_training_accuracy_close_to_sklearn(clf_data, d):
    """Training accuracy should track sklearn's at every depth."""
    X, y = clf_data
    mine = (DecisionTree(max_depth=d).fit(X, y).predict(X) == y).mean()
    ref = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y).score(X, y)
    assert abs(mine - ref) < 0.02


@pytest.mark.parametrize("d", [2, 4])
def test_regression_root_and_fit_match(reg_data, d):
    """Regression: same root split, and MSE within 2% of sklearn's."""
    X, y = reg_data
    mine = DecisionTree(task="regression", max_depth=d).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
    assert mine.root.feature_index == ref.tree_.feature[0]
    mse_m = np.mean((mine.predict(X) - y) ** 2)
    mse_r = np.mean((ref.predict(X) - y) ** 2)
    assert mse_m == pytest.approx(mse_r, rel=0.02)
