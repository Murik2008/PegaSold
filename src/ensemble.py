"""Bagging (bootstrap aggregating) ensemble built on our own DecisionTree.

A single tree fit to the whole training set has low bias but high variance:
move a few rows and the splits change. Bagging trains many trees on different
bootstrap samples (rows drawn with replacement) and averages their predictions.
Each tree stays just as wrong, but their errors are only partly correlated, so
averaging cancels much of the variance and leaves the bias where it was.

This wraps `src.decision_tree.DecisionTree` without modifying it. Each tree
trains on a sample of the data rather than every row, which keeps the runtime
manageable.

Why no feature subsampling: a real random forest also picks a random subset of
features at each node, to decorrelate the trees further. We measured it on this
dataset and it did not help (plain bagging 0.04213 MSE vs 0.04290 with 50% of
features per node, at 30 trees x 6,000 rows). 59 of our 82 features are sparse
location one-hots, and dropping them costs more signal than the extra
decorrelation buys back. So this stays plain bagging.
"""

from __future__ import annotations

import time

import numpy as np

from src.decision_tree import DecisionTree


class BaggedTrees:
    """
    An ensemble of DecisionTrees, each fit to its own bootstrap sample.

    A single tree has low bias but high variance: changing a few rows
    changes its splits. Training many trees on different random samples
    and averaging their predictions cancels much of that variance.
    Regression averages the member predictions.
    """

    def __init__(
            self,
            task="regression",
            n_estimators=30,
            max_samples=6000,
            max_depth=16,
            min_samples_leaf=2,
            criterion=None,
            random_state=42,
            oob_score=False,
            verbose=False,
    ):
        """
        Save ensemble settings and initialize attributes populated by fit.

        :param task: "regression" or "classification".
        :param n_estimators: How many trees to train.
        :param max_samples: Rows drawn per tree. An int is a row count; a float
            in (0, 1] is a fraction of the training set. Rows are drawn WITH
            replacement, so each tree sees roughly 63% distinct rows.
        :param max_depth: Depth cap passed to every member tree.
        :param min_samples_leaf: Leaf-size floor passed to every member tree.
            Lower than a single tree would use: averaging controls the
            variance, so each member is allowed to fit harder.
        :param criterion: Passed through to DecisionTree; None picks its default.
        :param random_state: Seeds the bootstrap draws, so runs are reproducible.
        :param oob_score: Also compute the out-of-bag estimate after fitting.
            Costs an extra pass over the training rows, so it is off by default.
        :param verbose: Print progress while fitting.
        """

        if task not in {"regression", "classification"}:    raise ValueError("task must be either 'regression' or 'classification'.")
        if n_estimators < 1:                                raise ValueError("n_estimators must be at least 1.")
        if isinstance(max_samples, float):
            if not 0.0 < max_samples <= 1.0:                raise ValueError("a float max_samples must be in (0, 1].")
        elif max_samples < 1:                               raise ValueError("an int max_samples must be at least 1.")

        self.task = task
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.oob_score = oob_score
        self.verbose = verbose

        self.estimators_ = []
        self.classes_ = None
        self.oob_score_ = None

    def _n_draw(self, n_rows):
        """
        Work out how many rows each bootstrap sample should contain.

        :param n_rows: Number of rows in the training set.
        :return: The row count to draw per tree.
        """

        if isinstance(self.max_samples, float):     return max(1, int(round(self.max_samples * n_rows)))
        return min(int(self.max_samples), n_rows)

    def fit(self, X, y):
        """
        Train every member tree on its own bootstrap sample.

        :param X: 2D array of numeric features, one row per example.
        :param y: 1D array of targets, one per row of X.
        :return: self, so calls can be chained.
        :raises ValueError: if X and y have the wrong shape or do not match.
        """

        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:                  raise ValueError("X must be 2D and y must be 1D")
        if X.shape[0] != y.size or X.shape[0] == 0:     raise ValueError("X and y must contain matching, non-empty samples")

        n_rows = X.shape[0]
        k = self._n_draw(n_rows)
        rng = np.random.default_rng(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.estimators_ = []
        # Row indices each tree did NOT see, kept only for the OOB estimate.
        oob_masks = []

        started = time.time()
        for i in range(self.n_estimators):
            idx = rng.integers(0, n_rows, size=k)
            tree = DecisionTree(
                task=self.task,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            ).fit(X[idx], y[idx])
            self.estimators_.append(tree)

            if self.oob_score:
                seen = np.zeros(n_rows, dtype=bool)
                seen[idx] = True
                oob_masks.append(~seen)

            if self.verbose:
                done = i + 1
                rate = (time.time() - started) / done
                print(f"    tree {done:3d}/{self.n_estimators}  "
                      f"({rate:.1f}s each, ~{rate * (self.n_estimators - done):.0f}s left)",
                      flush=True)

        if self.oob_score:
            self.oob_score_ = self._compute_oob(X, y, oob_masks)
            if self.verbose:
                print(f"    out-of-bag score: {self.oob_score_:.4f}")
        return self

    def _compute_oob(self, X, y, oob_masks) -> float:
        """Score each training row using only the trees that never saw it.

        This is an honest held-out estimate that costs no extra data: on
        average each row is out-of-bag for about a third of the trees.
        """
        n_rows = len(X)
        if self.task == "regression":
            total = np.zeros(n_rows)
            count = np.zeros(n_rows)
            for tree, mask in zip(self.estimators_, oob_masks):
                if mask.any():
                    total[mask] += tree.predict(X[mask])
                    count[mask] += 1
            ok = count > 0
            if not ok.any():
                return float("nan")
            pred = total[ok] / count[ok]
            ss_res = float(((pred - y[ok]) ** 2).sum())
            ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
            return float(1.0 - ss_res / ss_tot)  # OOB R²

        votes = np.zeros((n_rows, len(self.classes_)))
        count = np.zeros(n_rows)
        for tree, mask in zip(self.estimators_, oob_masks):
            if mask.any():
                votes[mask] += self._aligned_proba(tree, X[mask])
                count[mask] += 1
        ok = count > 0
        if not ok.any():
            return float("nan")
        pred = self.classes_[np.argmax(votes[ok], axis=1)]
        return float((pred == y[ok]).mean())  # OOB accuracy

    def _check(self, X) -> np.ndarray:
        """Validate X at prediction time and return it as an array.

        Both predict and predict_proba go through here, so the guards live in
        one place. A model that was never fitted has no estimators_, and a
        matrix with the wrong number of columns would otherwise fail deep
        inside a member tree with a confusing message.
        """
        if not self.estimators_:
            raise ValueError("fit must be called before prediction")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D with the fitted feature count")
        return X

    def predict(self, X):
        """
        Predict a target for every row in X.

        Regression returns the mean of the member predictions. Classification
        takes the class with the highest averaged probability.

        :param X: 2D array of numeric features.
        :return: 1D array of predictions.
        """

        X = self._check(X)
        if self.task == "regression":
            total = np.zeros(len(X))
            for tree in self.estimators_:
                total += tree.predict(X)
            return total / len(self.estimators_)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """Mean class probability across members (soft voting)."""
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        X = self._check(X)
        total = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            total += self._aligned_proba(tree, X)
        return total / len(self.estimators_)

    def _aligned_proba(self, tree: DecisionTree, X) -> np.ndarray:
        """A member's class probabilities, widened to the ensemble's classes.

        A bootstrap sample can miss a class entirely, so a member tree may know
        fewer classes than the ensemble. Its columns are placed under the
        matching ensemble classes and the rest left at zero.
        """
        proba = tree.predict_proba(X)
        if tree.classes_ is not None and len(tree.classes_) == len(self.classes_):
            return proba
        out = np.zeros((len(X), len(self.classes_)))
        for j, label in enumerate(tree.classes_):
            out[:, int(np.searchsorted(self.classes_, label))] = proba[:, j]
        return out

