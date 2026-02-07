"""Type stubs for ironforest.models module."""

from .linear_regression import LinearRegression as LinearRegression
from .decision_tree import DecisionTreeClassifier as DecisionTreeClassifier
from .decision_tree import DecisionTreeRegressor as DecisionTreeRegressor

__all__ = [
    "LinearRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]
