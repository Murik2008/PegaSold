import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from svm import PegasosSVM


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
    return StandardScaler().fit_transform(X), y


class TestLambdaSensitivity:
    def test_larger_lambda_shrinks_weights(self, clf_data):
        X, y = clf_data
        norms = [np.linalg.norm(PegasosSVM(lambda_param=l, epochs=30).fit(X, y).w)
                 for l in (1e-3, 1e-2, 1e-1, 1.0)]
        assert all(a > b for a, b in zip(norms, norms[1:])), norms

    def test_huge_lambda_gives_near_zero_weights(self, clf_data):
        X, y = clf_data
        m = PegasosSVM(lambda_param=1e3, epochs=10).fit(X, y)
        assert np.linalg.norm(m.w) < 0.05

    def test_weights_respect_pegasos_norm_bound_loosely(self, clf_data):
        # Optimal ||w|| <= 1/sqrt(lambda)
        X, y = clf_data
        lam = 0.01
        m = PegasosSVM(lambda_param=lam, epochs=100).fit(X, y)
        assert np.linalg.norm(m.w) <= 1.2 / np.sqrt(lam)

    @pytest.mark.parametrize("lam", [0.1, 0.01, 0.001, 0.0001])
    def test_each_swept_lambda_learns(self, clf_data, lam):
        X, y = clf_data
        m = PegasosSVM(lambda_param=lam, epochs=50).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.8

    def test_smaller_lambda_not_worse_on_validation(self, clf_data):
        # Project finding: 0.01 -> 0.001 gained +1.3 pp. Synthetic data can't
        # reproduce the size, only that the smaller lambda doesn't hurt.
        X, y = clf_data
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)
        acc = {l: (PegasosSVM(lambda_param=l, epochs=50, random_state=0)
                   .fit(Xtr, ytr).predict(Xva) == yva).mean()
               for l in (0.01, 0.001)}
        assert acc[0.001] >= acc[0.01] - 0.01, acc