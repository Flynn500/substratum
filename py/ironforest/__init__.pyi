"""IronForest: A library supporting spatial queries & tree-based machine learning."""

from ironforest._core import (
    Array as Array,
    zeros as zeros,
    ones as ones,
    full as full,
    asarray as asarray,
    eye as eye,
    diag as diag,
    column_stack as column_stack,
    linalg as linalg,
    stats as stats,
    random as random,
    spatial as spatial,
)

from . import models as models

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

__version__: str
