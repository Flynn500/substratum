"""IronForest: A library supporting spatial queries & tree-based machine learning."""

from ironforest._core import (
    Array,
    zeros,
    ones,
    full,
    asarray,
    eye,
    diag,
    column_stack,
    linalg,
    stats,
    random,
    spatial,
)

from . import models

__all__ = [
    "Array",
    "zeros",
    "ones",
    "full",
    "asarray",
    "eye",
    "diag",
    "column_stack",
    "linalg",
    "stats",
    "random",
    "spatial",
    "models",
]

__version__ = "0.2.2"
