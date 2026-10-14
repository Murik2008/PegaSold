"""Decision tree for classification and regression, implemented from scratch with NumPy."""

import numpy as np

class TreeNode:
    """One point in the tree: either a prediction leaf or a split node.

        A leaf stores its prediction. A split node stores which feature to inspect,
        the threshold to compare against, and references to its left and right
        child nodes.
    """

    def __init__( self,
            value = None,
            feature_index = None,
            threshold = None,
            left = None,
            right = None,
    ):
        """
        Store the information needed to represent this node.
        :param value: Prediction to return if this node is a leaf.
        :param feature_index: Column of X used by this node's split; None for leaf.
        :param  threshold: Rows <= this value go left; other rows go right.
        :param left: Left child TreeNode.
        :param right: Right child TreeNode.
        """

        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    """Decision tree interface for classification and regression.

        Classification predicts a category (such as a price tier). Regression
        predicts a numeric value (such as a house price). X should be a 2D NumPy
        array of numeric features; y should be a 1D array of targets.
    """

    def __init__( self,
            task="classification",
            criterion = None,
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            min_impurity_decrease = 0.0,
    ):
        """
        Save model settings and initialize attributes populated by fit.

        :param task: Either "classification" or "regression".
        :param criterion: Classification split measure ("gini" or "entropy") or
            regression measure ("mse"). Choose a sensible default from task
            when this is None.
        :param max_depth: Maximum number of split levels; None means no depth cap.
        :param min_samples_split: Minimum number of rows required to try splitting
            a node.
        :param min_samples_leaf: Minimum number of rows every leaf must keep.
        :param min_impurity_decrease: A split is only made if its gain is at
            least this value.
        """
        ...

    def fit(self, X, y):
        ...

    def _build_tree(self, X, y, depth):
        ...

    def _best_split(self, X, y):
        ...

    def _should_stop(self, y, depth):
        ...

    def _leaf_value(self, y):
        ...

    def _make_leaf(self, y, prediction):
        ...






