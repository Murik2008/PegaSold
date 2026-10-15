from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from decision_tree import DecisionTree
from svm import PegasosSVM

def fit_sklearn_baselines(
    X_tr, y_tr, t_tr,
    X_va, y_va, t_va,
    best_depth_reg, best_depth_clf,
) -> dict:
    """Train scikit-learn baseline models and evaluate them on validation data.

    Fits regression and classification baselines using the same training and
    validation splits as our custom models. The decision-tree baselines use
    the depths selected by our validation sweeps and the same minimum leaf
    size, making the tree-to-tree comparison more consistent.

    Regression baselines:
        - DecisionTreeRegressor: tree-based reference model.
        - Ridge: regularized linear regression.
        - DummyRegressor: predicts the training target mean.

    Classification baselines:
        - DecisionTreeClassifier: tree-based reference model.
        - LinearSVC: linear classification baseline.
        - DummyClassifier: stratified random predictions.

    Args:
        X_tr: Training feature matrix.
        y_tr: Training log-price targets.
        t_tr: Training binary class labels.
        X_va: Validation feature matrix.
        y_va: Validation log-price targets.
        t_va: Validation binary class labels.
        best_depth_reg: Selected depth for the regression tree.
        best_depth_clf: Selected depth for the classification tree.

    Returns:
        A dictionary of validation metrics using the same keys as the
        existing metrics dictionary in ``run()``.

    Notes:
        This function uses the existing module-level constants,
        ``MIN_SAMPLES_LEAF`` and ``RANDOM_SEED``, and the existing metric
        helpers, ``regression_metrics`` and ``classification_metrics``.
    """
    # Regression baselines
    sk_r = DecisionTreeRegressor(
        max_depth=best_depth_reg,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
    ).fit(X_tr, y_tr)

    ridge = Ridge(
        alpha=1.0,
        random_state=RANDOM_SEED,
    ).fit(X_tr, y_tr)

    dummy_r = DummyRegressor(
        strategy="mean",
    ).fit(X_tr, y_tr)

    # Classification baselines
    sk_c = DecisionTreeClassifier(
        max_depth=best_depth_clf,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
    ).fit(X_tr, t_tr)

    lsvc = LinearSVC(
        C=0.1,
        random_state=RANDOM_SEED,
    ).fit(X_tr, t_tr)

    dummy_c = DummyClassifier(
        strategy="stratified",
        random_state=RANDOM_SEED,
    ).fit(X_tr, t_tr)

    return {
        "sklearn_tree_regression_valid": regression_metrics(
            y_va, sk_r.predict(X_va)
        ),
        "ridge_valid": regression_metrics(
            y_va, ridge.predict(X_va)
        ),
        "dummy_regression_valid": regression_metrics(
            y_va, dummy_r.predict(X_va)
        ),
        "sklearn_tree_classification_valid": classification_metrics(
            t_va,
            sk_c.predict(X_va),
            sk_c.predict_proba(X_va)[:, 1],
        ),
        "linear_svc_valid": classification_metrics(
            t_va,
            lsvc.predict(X_va),
            lsvc.decision_function(X_va),
        ),
        "dummy_classification_valid": classification_metrics(
            t_va, dummy_c.predict(X_va)
        ),
    }


def fig_bias_variance(sweep_r: dict, sweep_c: dict, out: Path) -> None:
    """Plot training and validation performance across decision-tree depths.

    Creates two subplots from the results of ``sweep_depth``:
    one for log-price regression and one for binary classification.
    Regression uses RMSE, where lower values are better; classification
    uses accuracy, where higher values are better. A vertical dashed line
    marks the best depth selected by the validation sweep.

    Args:
        sweep_r: Regression sweep results containing ``depths``, ``train``,
            ``val``, and ``best``.
        sweep_c: Classification sweep results containing the same keys.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig1_bias_variance.png`` and closes the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Bias–Variance — our DecisionTree",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(
        sweep_r["depths"], sweep_r["train"],
        "o-", label="Train RMSE",
    )
    ax.plot(
        sweep_r["depths"], sweep_r["val"],
        "s-", label="Validation RMSE",
    )
    ax.axvline(
        sweep_r["best"], linestyle="--",
        label=f"Best depth = {sweep_r['best']}",
    )
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("RMSE (log-price)")
    ax.set_title("Regression")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(
        sweep_c["depths"], sweep_c["train"],
        "o-", label="Train accuracy",
    )
    ax.plot(
        sweep_c["depths"], sweep_c["val"],
        "s-", label="Validation accuracy",
    )
    ax.axvline(
        sweep_c["best"], linestyle="--",
        label=f"Best depth = {sweep_c['best']}",
    )
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    save(fig, out, "fig1_bias_variance.png")


def fig_classification(
    t_true, tree_scores, svm_scores, out: Path
) -> None:
    """Plot ROC curves and a confusion matrix for binary classification.

    The left subplot compares the ROC curves of our DecisionTree and
    PegasosSVM using their respective positive-class probabilities or
    decision scores. Each curve includes its ROC-AUC value. The right
    subplot displays the DecisionTree confusion matrix using a threshold
    of 0.5 on its positive-class probabilities.

    Args:
        t_true: One-dimensional array of true binary labels encoded as 0
            and 1.
        tree_scores: Positive-class probabilities from our DecisionTree.
        svm_scores: Decision scores from our PegasosSVM. Larger values
            should indicate stronger evidence for class 1.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig3_classification.png`` and closes the figure.

    Notes:
        The confusion matrix uses a fixed 0.5 threshold for the tree.
        The SVM decision scores are used for ranking in ROC analysis,
        not as probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Classification Quality",
        fontsize=13,
        fontweight="bold",
    )

    # ROC curves
    ax = axes[0]
    for name, scores in (
        ("Our DecisionTree", tree_scores),
        ("Our PegasosSVM", svm_scores),
    ):
        fpr, tpr, _ = roc_curve(t_true, scores)
        auc = roc_auc_score(t_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Confusion matrix for the decision tree
    pred = (np.asarray(tree_scores) >= 0.5).astype(int)
    true = np.asarray(t_true)

    cm = np.array([
        [int(((true == i) & (pred == j)).sum()) for j in (0, 1)]
        for i in (0, 1)
    ])

    ax = axes[1]
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("DecisionTree confusion matrix")

    save(fig, out, "fig3_classification.png")


def fig_model_comparison(
    reg_bars: dict, clf_bars: dict, out: Path
) -> None:
    """Compare our models with scikit-learn baselines on validation data.

    Creates two horizontal bar charts. The regression chart compares
    validation RMSE for log-price prediction, where lower values are
    better. The classification chart compares validation accuracy,
    where higher values are better. Model names beginning with ``OUR``
    are highlighted to distinguish our implementations from baselines.

    Args:
        reg_bars: Mapping of regression model names to validation RMSE.
        clf_bars: Mapping of classification model names to validation
            accuracy.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig4_model_comparison.png`` and closes the figure.

    Notes:
        The dictionaries should contain already-computed validation
        metrics. This function only visualizes results; it does not
        train or evaluate models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Our Models vs Scikit-learn Baselines",
        fontsize=13,
        fontweight="bold",
    )

    # Regression comparison: lower RMSE is better
    ax = axes[0]
    names = list(reg_bars)
    values = [reg_bars[name] for name in names]
    colors = [
        "seagreen" if name.startswith("OUR") else "lightsteelblue"
        for name in names
    ]

    ax.barh(names, values, color=colors)
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

    ax.set_xlabel("RMSE (log-price) — lower is better")
    ax.set_title("Regression")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Classification comparison: higher accuracy is better
    ax = axes[1]
    names = list(clf_bars)
    values = [clf_bars[name] for name in names]
    colors = [
        "seagreen" if name.startswith("OUR") else "lightsteelblue"
        for name in names
    ]

    ax.barh(names, values, color=colors)
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Accuracy — higher is better")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Classification")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    save(fig, out, "fig4_model_comparison.png")