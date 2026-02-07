"""Type stubs for ironforest.decision_tree module."""

from typing import Optional, Literal
from ironforest._core import Array

class DecisionTreeClassifier:
    """A decision tree classifier."""

    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int
    max_features: Optional[int]
    criterion: Literal["gini", "entropy"]
    random_state: int
    tree_: Optional[object]
    n_classes_: Optional[int]
    n_features_: Optional[int]

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Literal["gini", "entropy"] = "gini",
        random_state: int = 42,
    ) -> None: ...

    def fit(self, X: Array, y: Array) -> DecisionTreeClassifier:
        """Build a decision tree classifier from the training set (X, y).

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values (class labels).

        Returns:
            self : DecisionTreeClassifier
                Fitted estimator.
        """
        ...

    def predict(self, X: Array) -> Array:
        """Predict class for X.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            y: Array of shape (n_samples,)
                The predicted classes.
        """
        ...

    @property
    def n_nodes(self) -> Optional[int]:
        """Number of nodes in the tree."""
        ...

    @property
    def max_depth_reached(self) -> Optional[int]:
        """Maximum depth actually reached during training."""
        ...


class DecisionTreeRegressor:
    """A decision tree regressor."""

    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int
    max_features: Optional[int]
    random_state: int
    tree_: Optional[object]
    n_features_: Optional[int]

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None: ...

    def fit(self, X: Array, y: Array) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y).

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The training input samples.
            y: Array or array-like of shape (n_samples,)
                The target values.

        Returns:
            self : DecisionTreeRegressor
                Fitted estimator.
        """
        ...

    def predict(self, X: Array) -> Array:
        """Predict regression target for X.

        Args:
            X: Array or array-like of shape (n_samples, n_features)
                The input samples.

        Returns:
            y: Array of shape (n_samples,)
                The predicted values.
        """
        ...

    @property
    def n_nodes(self) -> Optional[int]:
        """Number of nodes in the tree."""
        ...

    @property
    def max_depth_reached(self) -> Optional[int]:
        """Maximum depth actually reached during training."""
        ...
