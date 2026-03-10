from typing import Literal
import ironforest as irn


class KNNClassifier:
    """
    k-Nearest Neighbors classifier.

    This classifier predicts the class of a sample based on the majority
    class among its k nearest neighbors in the training data.

    Args:
        k: Number of neighbors to use
        weights: Weighting strategy ("uniform" or "distance")
        tree: Spatial tree type used for neighbor search ("kd", "ball", "vp", "rp")
        metric: Distance metric used to compute neighbors
        leaf_size: Maximum number of points stored in a tree leaf
    """
    def __init__(
        self,
        k: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        tree: Literal["kd", "ball", "vp", "rp"] = "kd",
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        leaf_size: int = 20,
    ):
        self.k = k
        self.weights = weights
        self.tree_type = tree
        self.metric = metric
        self.leaf_size = leaf_size

    def fit(self, X, y) -> "KNNClassifier":
        """
        Fit the KNN classifier.

        Builds the spatial tree and stores the training labels.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target class labels of shape (n_samples,)

        Returns:
            self: Fitted estimator
        """
        self.labels = irn.ndutils.asarray(y)
        self.classes = sorted(set(y))

        match self.tree_type:
            case "kd":
                self.tree = irn.spatial.KDTree(X, self.leaf_size, self.metric) # type: ignore
            case "ball":
                self.tree = irn.spatial.BallTree(X, self.leaf_size, self.metric) # type: ignore
            case "vp":
                self.tree = irn.spatial.VPTree(X, self.leaf_size, self.metric) # type: ignore
            case "rp":
                self.tree = irn.spatial.RPTree(X, self.leaf_size, self.metric) # type: ignore

        return self

    def predict(self, X) -> irn.Array | int:
        """
        Predict class labels for samples.

        Args:
            X: Input data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Predicted class labels. Returns a single label if one sample
            is provided, otherwise an array of shape (n_samples,)
        """
        X = irn.ndutils.asarray(X)
        if X.ndim == 1:
            X = X.reshape([1, X.shape[0]])

        result = self.tree.query_knn(X, self.k)
        results = result.split()

        predictions = []

        for r in results:
            neighbor_targets = self.labels[r.indices] # type: ignore

            if self.weights == "distance":
                w = 1.0 / (r.distances + 1e-10)
                counts = {}
                for i, idx in enumerate(r.indices):
                    label = self.labels[idx]
                    counts[label] = counts.get(label, 0.0) + w[i]
                predictions.append(max(counts, key=counts.get)) # type: ignore
            else:
                predictions.append(neighbor_targets.mode())

        if len(predictions) == 1:
            return predictions[0]
        return irn.ndutils.asarray(predictions)

    def predict_proba(self, X) -> irn.Array:
        """
        Estimate class probabilities for samples.

        The probability of each class is computed from the class
        distribution among the k nearest neighbors.

        Args:
            X: Input data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of shape (n_samples, n_classes) containing class probabilities
        """
        X = irn.ndutils.asarray(X)
        if X.ndim == 1:
            X = X.reshape([1, X.shape[0]])

        results = self.tree.query_knn(X, self.k).split()
        n_classes = len(self.classes)

        probas = []
        for r in results:
            counts = {c: 0.0 for c in self.classes}
            if self.weights == "distance":
                w = 1.0 / (r.distances + 1e-10)
                for i, idx in enumerate(r.indices):
                    counts[self.labels[idx]] += w[i] # type: ignore
            else:
                for idx in r.indices:
                    counts[self.labels[idx]] += 1.0 # type: ignore

            total = sum(counts.values())
            probas.extend([counts[c] / total for c in self.classes])

        return irn.ndutils.asarray(probas).reshape([len(results), n_classes])


class KNNRegressor:
    """
    k-Nearest Neighbors regressor.

    This regressor predicts the target value of a sample based on the
    average value of its k nearest neighbors in the training data.

    Args:
        k: Number of neighbors to use
        weights: Weighting strategy ("uniform" or "distance")
        tree: Spatial tree type used for neighbor search ("kd", "ball", "vp", "rp")
        metric: Distance metric used to compute neighbors
        leaf_size: Maximum number of points stored in a tree leaf
    """
    def __init__(
        self,
        k: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        tree: Literal["kd", "ball", "vp", "rp"] = "kd",
        metric: Literal["euclidean", "manhattan", "chebyshev", "cosine"] = "euclidean",
        leaf_size: int = 20,
    ):
        self.k = k
        self.weights = weights
        self.tree_type = tree
        self.metric = metric
        self.leaf_size = leaf_size

    def fit(self, X, y) -> "KNNRegressor":
        """
        Fit the KNN regressor.

        Builds the spatial tree and stores the training targets.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self: Fitted estimator
        """
        self.labels = irn.ndutils.asarray(y)

        match self.tree_type:
            case "kd":
                self.tree = irn.spatial.KDTree(X, self.leaf_size, self.metric) # type: ignore
            case "ball":
                self.tree = irn.spatial.BallTree(X, self.leaf_size, self.metric) # type: ignore
            case "vp":
                self.tree = irn.spatial.VPTree(X, self.leaf_size, self.metric) # type: ignore
            case "rp":
                self.tree = irn.spatial.RPTree(X, self.leaf_size, self.metric) # type: ignore

        return self

    def predict(self, X) -> irn.Array | float:
        """
        Predict target values for samples.

        Args:
            X: Input data of shape (n_samples, n_features) or (n_features,)

        Returns:
            Predicted target values. Returns a single value if one sample
            is provided, otherwise an array of shape (n_samples,)
        """
        X = irn.ndutils.asarray(X)
        if X.ndim == 1:
            X = X.reshape([1, X.shape[0]])

        result = self.tree.query_knn(X, self.k)
        results = result.split()

        predictions = []

        for r in results:
            neighbor_targets = self.labels[r.indices] # type: ignore

            if self.weights == "distance":
                w = 1.0 / (r.distances + 1e-10)
                predictions.append((w * neighbor_targets).sum() / w.sum())
            else:
                predictions.append(neighbor_targets.mean())

        if len(predictions) == 1:
            return predictions[0]
        return irn.ndutils.asarray(predictions)