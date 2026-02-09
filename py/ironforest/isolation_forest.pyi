"""Type stubs for ironforest.isolation_forest module."""

from typing import Optional
from ironforest._core import Array

class IsolationForest:
    """Isolation Forest for anomaly detection."""

    n_estimators: int
    max_samples: int
    contamination: float
    max_features: Optional[int]
    random_state: int
    trees_: list
    n_features_: Optional[int]
    offset_: Optional[float]
    threshold_: Optional[float]

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        """Isolation forest for anomaly detection.

        Args:
            n_estimators: int, default=100
                The number of isolation trees in the forest.
            max_samples: int, default=256
                The number of samples to draw from X to train each tree.
                If max_samples is larger than the number of samples provided,
                all samples will be used for all trees (no sampling).
            contamination: float, default=0.1
                The proportion of outliers in the dataset. Used to define the
                threshold for the decision function. Must be in (0, 0.5].
            max_features: int or None, default=None
                The number of features to consider when looking for the best split.
                If None, all features are considered.
            random_state: int, default=42
                Controls the randomness of the sampling and tree construction.
        """
        ...

    def fit(self, X: Array, y: Optional[Array] = None) -> IsolationForest:
        """Build an isolation forest from the training set X.

        The forest is built by training multiple isolation trees on
        random subsamples of the dataset.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Ignored
                Not used, present for API consistency by convention.

        Returns:
            self : IsolationForest
        """
        ...

    def predict(self, X: Array) -> Array:
        """Predict if a particular sample is an outlier or not.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                Returns -1 for anomalies/outliers and 1 for inliers.
        """
        ...

    def score_samples(self, X: Array) -> Array:
        """Compute the anomaly score of each sample.

        The anomaly score is computed as the negative average path length
        in the trees, normalized by the expected path length. Lower scores
        indicate anomalies.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                The anomaly score of the input samples. Lower scores indicate
                anomalies, higher scores indicate normal instances.
        """
        ...

    def decision_function(self, X: Array) -> Array:
        """Average anomaly score of X.

        The anomaly score is based on the average path length in the trees.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            Array of shape (n_samples,)
                The anomaly score. The more negative, the more anomalous.
        """
        ...
