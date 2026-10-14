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
        if task not in {"classification", "regression"}:
            raise ValueError("task must be either 'classification' or 'regression'.")
        if criterion is None:
            criterion = "gini" if task == "classification" else "mse"
        if task == "classification":
            if criterion not in {"gini", "entropy"}:
                raise ValueError("classification criterion must be either 'gini' or 'entropy'.")
        else:
            if criterion != "mse":
                raise ValueError("regression criterion must be either 'mse'.")

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease cannot be negative.")

        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        # This will contain the root TreeNode after fit()
        self.root = None


    def fit(self, x, y):
        """
        Learn the tree from training data and return this model.

        :param x: 2D array of numeric features, one row per example.
        :param y: 1D array of targets, one per row of x.
        :return: self, so calls can be chained.
        """

        x = np.array(x)
        y = np.array(y)

        if x.ndim != 2:                             raise ValueError("x must be a 2D NumPy array.")
        if y.ndim != 1:                             raise ValueError("y must be a 1D NumPy array.")
        if x.shape[0] != y.shape[0]:                raise ValueError("x and y must have the same number of samples.")
        if x.shape[0] == 0:                         raise ValueError("x and y can't be empty.")
        if x.shape[1] == 0:                         raise ValueError("x must contain at least one feature.")
        if not np.issubdtype(x.dtype, np.number):   raise ValueError("x must contain numeric values")
        if not np.isfinite(x).all():                raise ValueError("x must contain only finite values")
        if self.task == "regression":
            try:    y = y.astype(float)
            except (TypeError, ValueError) as exc:  raise ValueError("Regression targets must be numeric") from exc
            if not np.isfinite(y).all():            raise ValueError("Regression targets must be finite")

        self.n_features_in_ = x.shape[1]
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.root = self._build_tree(x, y, depth=0)

        return self


    def _build_tree(self, x, y, depth):
        ...

    def _best_split(self, x, y):
        ...

    def _should_stop(self, y, depth):
        ...

    def _leaf_value(self, y):
        ...

    def _make_leaf(self, y, prediction):
        ...






