"""Bagging (bootstrap aggregating) ensemble built on our own DecisionTree."""
from __future__ import annotations

import time

import numpy as np

from src.decision_tree import DecisionTree


class BaggedTrees:
    """An ensemble of DecisionTrees, each fit to its own bootstrap sample.

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
        ...

    def _n_draw(self, n_rows):
        ...

    def fit(self, X, y):
        ...

    def predict(self, X):
        """
        Predict a target for every row in X (regression)
        """
        ...