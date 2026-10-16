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


    # Waiting for fit() to be completed so I can finish the support vectors and margin_width_ part.