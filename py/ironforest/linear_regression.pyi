"""Type stubs for ironforest.linear_regression module."""

from typing import Optional
from ironforest._core import Array

class LinearRegression:
    """Linear regression model using least squares."""

    fit_intercept: bool
    coef_: Optional[Array]
    intercept_: Optional[float]

    def __init__(self, fit_intercept: bool = True) -> None:
        """Initialize linear regression model.

        Args:
            fit_intercept: Whether to calculate the intercept for this model.
        """
        ...

    def fit(self, X: Array, y: Array) -> LinearRegression:
        """Fit linear model.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            self: Fitted estimator.
        """
        ...

    def predict(self, X: Array) -> Array:
        """Predict using the linear model.

        Args:
            X: Samples of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,) or (n_samples, n_targets).

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        ...

    def score(self, X: Array, y: Array) -> float:
        """Return the coefficient of determination (R²) of the prediction.

        Args:
            X: Test samples of shape (n_samples, n_features).
            y: True values of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            R² score.

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        ...

    def residuals(self, X: Array, y: Array) -> Array:
        """Calculate residuals (y - y_pred).

        Args:
            X: Samples of shape (n_samples, n_features).
            y: True values of shape (n_samples,) or (n_samples, n_targets).

        Returns:
            Residuals array.

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        ...
