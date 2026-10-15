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
        self.feature_index = feature_index   # None marks this node as a leaf
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
        # Validate every setting up front so a bad value fails at construction,
        # not halfway through training
        if task not in {"classification", "regression"}:
            raise ValueError("task must be either 'classification' or 'regression'.")
        # Default criterion depends on the task
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

        # Overfitting controls
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


    def fit(self, X, y):
        """
        Learn the tree from training data and return this model.

        :param X: 2D array of numeric features, one row per example.
        :param y: 1D array of targets, one per row of X.
        :return: self, so calls can be chained.
        """

        # asarray does not copy when the input is already an array
        X = np.asarray(X)
        y = np.asarray(y)

        # Validate inputs before doing any work (shape first, then contents)
        if X.ndim != 2:                             raise ValueError("X must be a 2D NumPy array.")
        if y.ndim != 1:                             raise ValueError("y must be a 1D NumPy array.")
        if X.shape[0] != y.shape[0]:                raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:                         raise ValueError("X and y can't be empty.")
        if X.shape[1] == 0:                         raise ValueError("X must contain at least one feature.")
        if not np.issubdtype(X.dtype, np.number):   raise ValueError("X must contain numeric values")
        if not np.isfinite(X).all():                raise ValueError("X must contain only finite values")
        if self.task == "regression":
            # Regression targets must be real numbers; classification labels can be anything
            try:    y = y.astype(float)
            except (TypeError, ValueError) as exc:  raise ValueError("Regression targets must be numeric") from exc
            if not np.isfinite(y).all():            raise ValueError("Regression targets must be finite")

        self.n_features_in_ = X.shape[1]
        # Remember the original labels so predictions can be mapped back to them
        self.classes_ = np.unique(y) if self.task == "classification" else None
        # Grow the whole tree recursively, starting at depth 0
        self.root = self._build_tree(X, y, depth=0)

        return self


    def _build_tree(self, X, y, depth):
        """
        Build one node recursively and return it.

        Makes a leaf if a stopping rule applies or no useful split exists.
        Otherwise, splits on the best feature and threshold and recurses on
        both halves.

        :param X: 2D feature array for the rows that reached this node.
        :param y: 1D targets for those rows.
        :param depth: Depth of this node (the root is 0).
        :return: A leaf TreeNode or a split TreeNode with two children.
        """

        # Computed up front because every early exit below returns this leaf
        prediction = self._leaf_value(y)

        # Stopping rules: depth cap, too few rows, or a pure node
        if self._should_stop(y, depth):                     return self._make_leaf(y, prediction)

        best_split = self._best_split(X, y)

        # No split satisfied min_samples_leaf and improved the impurity
        if best_split is None:                              return self._make_leaf(y, prediction)
        # Safety check: _best_split already only returns gains above this value
        if best_split[2] < self.min_impurity_decrease:      return self._make_leaf(y, prediction)

        feature_index, threshold, _ = best_split

        # Rows <= threshold go left, the rest go right
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        # Safety net: float rounding in the midpoint threshold could leave one side empty
        if len(y_left) == 0 or len(y_right) == 0:           return self._make_leaf(y, prediction)

        # Each child is grown independently on its own subset of rows
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return TreeNode( feature_index=feature_index, threshold=threshold, left=left_child, right=right_child )

    def _best_split(self, X, y):
        """
        Search every feature and threshold and return the best split.

        Each feature is sorted once, and cumulative sums give the impurity of
        every possible left/right division in one pass, instead of recomputing
        it for each threshold.

        :param X: 2D feature array for the rows at this node.
        :param y: 1D targets for those rows.
        :return: (feature_index, threshold, gain), or None if no split helps.
        """
        n_samples, n_features = X.shape
        # Starting at min_impurity_decrease means only splits that beat it can win
        best_gain, best_feature, best_threshold = self.min_impurity_decrease, None, None

        # Quantities that do not depend on the feature, computed once per node
        if self.task == "classification":
            _, encoded = np.unique(y, return_inverse=True)       # labels -> 0..k-1
            class_count = encoded.max() + 1
            onehot = np.eye(class_count, dtype=float)[encoded]   # one row per sample, one column per class
            total_counts = onehot.sum(axis=0)
            parent_score = self._impurity(y)
        else:
            # Centre y first: variance from sums of squares loses precision on large values
            yy = y.astype(float) - np.mean(y)
            total_sum, total_sq = yy.sum(), np.dot(yy, yy)
            parent_score = float(total_sq / n_samples)           # MSE of the parent node

        for feature_index in range(n_features):
            # Sort rows by this feature; mergesort is stable, so results are reproducible
            order = np.argsort(X[:, feature_index], kind="mergesort")
            values = X[order, feature_index]

            # A split is only possible between two different neighbouring values
            positions = np.flatnonzero(values[:-1] < values[1:]) + 1
            # min_samples_leaf: keep only splits where both sides are big enough
            positions = positions[(positions >= self.min_samples_leaf) & (n_samples - positions >= self.min_samples_leaf)]
            if not positions.size:
                continue

            # Threshold halfway between the two neighbouring values
            thresholds = values[positions - 1] + (values[positions] - values[positions - 1]) / 2
            nl, nr = positions.astype(float), (n_samples - positions).astype(float)   # left/right sizes per candidate

            if self.task == "classification":
                # Class counts on the left of every candidate; the right side is the remainder
                left_counts = np.cumsum(onehot[order], axis=0)[positions - 1]
                right_counts = total_counts - left_counts

                if self.criterion == "gini":
                    # Gini = 1 - sum(p^2), with p = count / size
                    il = 1.0 - np.sum(left_counts ** 2, axis=1) / nl ** 2
                    ir = 1.0 - np.sum(right_counts ** 2, axis=1) / nr ** 2
                else:
                    def entropy(counts, sizes):
                        probs = counts / sizes[:, None]
                        # maximum() avoids log2(0); where() makes those terms contribute 0
                        return -np.sum(np.where(probs > 0, probs * np.log2(np.maximum(probs, 1e-300)), 0), axis=1)

                    il, ir = entropy(left_counts, nl), entropy(right_counts, nr)
            else:
                # Variance = E[y^2] - E[y]^2, computed from prefix sums for each candidate
                sorted_y = yy[order]
                prefix = np.cumsum(sorted_y)
                prefix_sq = np.cumsum(sorted_y ** 2)
                left_sum = prefix[positions - 1]
                left_sq = prefix_sq[positions - 1]
                right_sum, right_sq = total_sum - left_sum, total_sq - left_sq
                # maximum(0, ...) guards against tiny negative values from rounding
                il = np.maximum(0.0, left_sq / nl - (left_sum / nl) ** 2)
                ir = np.maximum(0.0, right_sq / nr - (right_sum / nr) ** 2)

            # Gain = parent impurity minus the size-weighted child impurity
            gains = parent_score - (nl / n_samples * il + nr / n_samples * ir)
            local = int(np.argmax(gains))   # best candidate for this feature
            gain = float(gains[local])

            # Keep it only if it beats the best split found on earlier features
            if gain > best_gain:
                best_gain, best_feature = gain, feature_index
                best_threshold = float(thresholds[local])

        # Nothing beat min_impurity_decrease (or no valid split existed)
        if best_feature is None:
            return None

        return best_feature, best_threshold, best_gain

    def _should_stop(self, y, depth):
        """
        Decide whether the current node should become a leaf.

        :param y: 1D array of targets that reached this node.
        :param depth: Depth of this node (the root is 0).
        :return: True if growth should stop here.
        """
        # Too few rows left to try a split
        if len(y) < self.min_samples_split:                             return True
        # Depth cap reached
        if self.max_depth is not None and depth >= self.max_depth:      return True
        # Pure classification node: only one class left
        if self.task == "classification" and len(np.unique(y)) == 1:    return True

        # Regression equivalent of "pure": all targets identical, so MSE is 0
        # (relies on _impurity, which is implemented by a teammate)
        return self.task == "regression" and self._impurity(y) == 0

    def _leaf_value(self, y):
        """
        Calculate the prediction a leaf holding targets y would return.

        :param y: 1D array of targets that reached this node.
        :return: Most frequent label (classification) or mean target (regression).
        """
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            # argmax picks the first maximum, so ties go to the smaller label
            return values[np.argmax(counts)]

        return np.mean(y)

    def _make_leaf(self, y, prediction):
        """
        Create a leaf node.

        For classification the leaf also stores how many rows of each class
        reached it, so predict_proba can later turn them into frequencies.

        :param y: 1D array of targets that reached this node.
        :param prediction: Value the leaf returns (from _leaf_value).
        :return: A TreeNode with no split, only a value.
        """
        node = TreeNode(value=prediction)

        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            # Added here rather than in TreeNode.__init__, so only classification leaves have it
            node.class_counts = dict(zip(values.tolist(), counts.tolist()))

        return node
