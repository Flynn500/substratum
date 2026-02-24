"""Machine learning models built on top of ironforest."""

from .linear_regression import LinearRegression as LinearRegression
from .local_regression import LocalRegression as LocalRegression
from .decision_tree import DecisionTreeClassifier as DecisionTreeClassifier
from .decision_tree import DecisionTreeRegressor as DecisionTreeRegressor
from .random_forest import RandomForestRegressor as RandomForestRegressor
from .random_forest import RandomForestClassifier as RandomForestClassifier
from .isolation_forest import IsolationForest as IsolationForest

__all__ = [
    "LinearRegression",
    "LocalRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "IsolationForest"
]
