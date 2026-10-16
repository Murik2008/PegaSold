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


# predict_proba and prediction-time input validation


@pytest.fixture(scope="module")
def small_clf():
    X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    return DecisionTree(max_depth=5).fit(X, y), X


def test_predict_proba_matches_sklearn(clf_data):
    """Leaf class frequencies should equal sklearn's predict_proba."""
    X, y = clf_data
    mine = DecisionTree(max_depth=5).fit(X, y).predict_proba(X)
    ref = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y).predict_proba(X)
    assert np.abs(mine - ref).mean() < 0.02


def test_predict_proba_rows_sum_to_one(small_clf):
    t, X = small_clf
    p = t.predict_proba(X)
    assert p.shape == (len(X), len(t.classes_))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_predict_proba_argmax_matches_predict(small_clf):
    t, X = small_clf
    np.testing.assert_array_equal(t.classes_[t.predict_proba(X).argmax(axis=1)], t.predict(X))


def test_impure_leaf_gives_frequencies():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1])
    p = DecisionTree(max_depth=0).fit(X, y).predict_proba(X)   # one leaf: 3 vs 1
    np.testing.assert_allclose(p, [[0.75, 0.25]] * 4)


def test_predict_proba_rejects_regression():
    t = DecisionTree(task="regression").fit(np.random.rand(20, 2), np.random.rand(20))
    with pytest.raises(ValueError):
        t.predict_proba(np.zeros((1, 2)))


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_before_fit_raises(method):
    with pytest.raises(ValueError):
        getattr(DecisionTree(), method)(np.zeros((1, 2)))


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((3, 5)),
                                 np.full((2, 8), np.nan), np.full((2, 8), np.inf),
                                 np.full((2, 8), "a")])
def test_bad_input_raises(small_clf, method, bad):
    """1-D, wrong feature count, non-finite, non-numeric."""
    t, _ = small_clf
    with pytest.raises(ValueError):
        getattr(t, method)(bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_fit_rejects_non_finite_X(bad):
    X = np.array([[0.0, 1.0], [bad, 2.0], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        DecisionTree().fit(X, np.array([0, 1, 0, 1]))

