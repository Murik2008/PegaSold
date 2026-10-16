"""Linear soft-margin SVM trained with the Pegasos sub-gradient method.

Objective: lambda/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b))). Scale input
features before fitting. Labels may be any two distinct values and are mapped
internally to -1/+1.
"""

import numpy as np


class PegasosSVM:
    """Binary linear SVM using stochastic sub-gradient updates."""

    def _objective(self, X, signs):
        """
        Compute the regularised hinge-loss objective at the current weights.

        Objective: lambda/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b))).
        It is recorded once per epoch in fit, so the training curve can be
        checked for convergence.

        :param X: 2D array of scaled features.
        :param signs: 1D array of labels mapped to -1/+1.
        :return: The objective value as a float.
        """
        hinge = np.maximum(0.0, 1.0 - signs * (X @ self.w + self.b))
        return float(0.5 * self.lambda_param * np.dot(self.w, self.w) + hinge.mean())

    def __init__(self, lambda_param=0.01, epochs=100, learning_rate="pegasos", random_state=42):
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if learning_rate != "pegasos" and (not np.isscalar(learning_rate) or learning_rate <= 0):
            raise ValueError("learning_rate must be 'pegasos' or a positive number")
        self.lambda_param = float(lambda_param)
        self.epochs = int(epochs)
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        if X.shape[0] != y.size or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X and y must contain matching, non-empty samples and features")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("PegasosSVM requires exactly two target classes")

        self.classes_ = classes
        signs = np.where(y == classes[1], 1.0, -1.0)
        self.n_features_in_ = X.shape[1]
        self.w = np.zeros(self.n_features_in_, dtype=float)
        self.b = 0.0
        rng = np.random.default_rng(self.random_state)
        self.objective_history_ = []
        step = 0

        for _ in range(self.epochs):
            for i in rng.permutation(X.shape[0]):
                step += 1
                eta = (1.0 / (self.lambda_param * step)
                       if self.learning_rate == "pegasos"
                       else float(self.learning_rate))
                margin = signs[i] * (np.dot(self.w, X[i]) + self.b)
                # Regularization applies to w; the intercept is not penalized.
                self.w *= 1.0 - eta * self.lambda_param
                if margin < 1.0:
                    self.w += eta * signs[i] * X[i]
                    self.b += eta * signs[i]
            self.objective_history_.append(self._objective(X, signs))

        self.support_vector_mask_ = signs * self.decision_function(X) <= 1.0
        self.support_vectors_ = X[self.support_vector_mask_]
        norm = np.linalg.norm(self.w)
        self.margin_width_ = float(2.0 / norm) if norm > 0 else np.inf

        return self