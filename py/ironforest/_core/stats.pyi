"""Type stubs for ironforest._core.stats module."""

from typing import overload
from ironforest._core import Array, ArrayLike


def sum(a: ArrayLike) -> float:
    """Sum of all elements."""
    ...

def mean(a: ArrayLike) -> float:
    """Mean of all elements."""
    ...

def var(a: ArrayLike) -> float:
    """Variance of all elements (population variance)."""
    ...

def std(a: ArrayLike) -> float:
    """Standard deviation of all elements."""
    ...

def median(a: ArrayLike) -> float:
    """Median of all elements."""
    ...

def mode(a: ArrayLike) -> float | int:
    """Mode of all elements."""
    ...

@overload
def quantile(a: ArrayLike, q: float) -> float:
    """q-th quantile of all elements (q in [0, 1])."""
    ...

@overload
def quantile(a: ArrayLike, q: ArrayLike) -> Array[float]:
    """Compute multiple quantiles at once (vectorized).

    Args:
        a: Input array.
        q: Array of quantile values, each in [0, 1].

    Returns:
        1D array of quantile values corresponding to each q.
    """
    ...

def any(a: ArrayLike) -> bool:
    """True if any element is non-zero."""
    ...

def all(a: ArrayLike) -> bool:
    """True if all elements are non-zero."""
    ...

def pearson(a: ArrayLike, b: ArrayLike) -> float:
    """Compute Pearson correlation coefficient between two arrays.

    Args:
        a: First 1D array.
        b: Second 1D array of the same length.

    Returns:
        Pearson correlation coefficient between -1 and 1.
    """
    ...

def spearman(a: ArrayLike, b: ArrayLike) -> float:
    """Compute Spearman rank correlation coefficient between two arrays.

    Args:
        a: First 1D array.
        b: Second 1D array of the same length.

    Returns:
        Spearman correlation coefficient between -1 and 1.
    """
    ...
