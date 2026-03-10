import ironforest as irn

def _tricube(d, max_d):
    """Tricube kernel: (1 - (d/max_d)^3)^3 for d < max_d, else 0."""
    if max_d == 0.0:
        return irn.ndutils.ones(d.shape[0])
    u = d / max_d
    mask = u < 1.0
    w = (1.0 - u ** 3) ** 3
    return w * mask

def _epanechnikov(d, max_d):
    if max_d == 0.0:
        return irn.ndutils.ones(d.shape[0])
    u = d / max_d
    mask = u < 1.0
    return (1.0 - u ** 2) * mask

def _gaussian(d, max_d):
    u = d / (max_d / 3.0) if max_d > 0.0 else d
    return irn.Array.exp(-0.5 * u ** 2)

class LocalRegression:
    """
    Local polynomial regression estimator.

    Predictions are computed by fitting a weighted polynomial model
    to the k nearest neighbors of each query point. Neighbor weights
    are determined by a kernel function applied to their distances.

    Args:
        tree: Spatial index built from the predictor data
        y: Response values of shape (n_samples,)
        kernel: Kernel function used for weighting neighbors
        degree: Local polynomial degree (1 or 2)
        k: Number of neighbors used for each local fit
    """
    KERNELS = {
        "tricube": _tricube,
        "epanechnikov": _epanechnikov,
        "gaussian": _gaussian,
    }

    def __init__(self, tree, y, kernel="tricube", degree=1, k=None):
        if kernel not in self.KERNELS:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")

        self.tree = tree
        self.y = y
        self.kernel = kernel
        self.degree = degree
        self.k = k
    
    @staticmethod
    def from_array(arr: irn.Array, y, kernel, degree, k=None) -> "LocalRegression":
        """
        Construct a LocalRegression model from raw predictor data.

        A KDTree is built internally from the provided array.

        Args:
            arr: Predictor data of shape (n_samples, n_features)
            y: Response values of shape (n_samples,)
            kernel: Kernel function used for weighting neighbors
            degree: Local polynomial degree (1 or 2)
            k: Number of neighbors used for each local fit

        Returns:
            LocalRegression: Initialized regression model
        """
        tree = irn.spatial.KDTree.from_array(arr)
        return LocalRegression(tree, y, kernel, degree, k)

    def _build_design(self, X_local):
        X = irn.ndutils.asarray(X_local)
        n = X.shape[0]
        cols = [irn.ndutils.ones([n])]

        p = X.shape[1] if X.ndim > 1 else 1

        if p == 1:
            cols.append(X)
            if self.degree >= 2:
                cols.append(X ** 2)
        else:
            for j in range(p):
                cols.append(X[:, j])
            if self.degree >= 2:
                for j in range(p):
                    cols.append(X_local[:, j] ** 2)

        return irn.ndutils.column_stack(cols)
    
    def _get_points(self, indices):
        return self.tree.data(indices)
    
    def predict(self, X_query):
        """
        Predict response values for query points.

        A weighted local polynomial model is fitted to the k nearest
        neighbors of each query point using the specified kernel.

        Args:
            X_query: Query points of shape (n_samples, n_features)
                or (n_features,) for a single sample

        Returns:
            Predicted values of shape (n_samples,). Returns a single
            value if one query point is provided.
        """
        
        kernel_fn = self.KERNELS[self.kernel]
        single = X_query.ndim == 1
        if single:
            X_query = X_query.reshape([1, X_query.shape[0]])

        m = X_query.shape[0]
        k = self.k or max(int(self.y.shape[0] * 0.3), 3)
        preds = irn.ndutils.zeros([m])

        for i in range(m):
            q = X_query[i]
            result = self.tree.query_knn(q, k)

            max_d = result.distances[k - 1]
            w = kernel_fn(result.distances, max_d)

            X_local = self._get_points(result.indices)

            y_local = self.y[result.indices].reshape([k, 1])

            A = self._build_design(X_local)
            q_design = self._build_design(q.reshape([1, q.shape[0]]))

            coeffs, _ = irn.linalg.weighted_lstsq(A, y_local, w)

            preds[i] = (q_design @ coeffs).item()

        if single:
            return preds[0]
        return preds

if __name__ == "__main__":
    X = irn.ndutils.linspace(-5, 5, 200).reshape([200, 1])
    y = irn.Array.sin(X.reshape([200])) + irn.random.Generator.from_seed(0).normal(0, 0.1, [200])

    # Fit model
    model = LocalRegression.from_array(
        X,
        y,
        kernel="tricube",
        degree=1,
        k=40,
    )

    # Query points
    X_test = irn.ndutils.linspace(-5, 5, 50).reshape([50, 1])

    preds = model.predict(X_test)

    # Should roughly match sin(x)
    y_true = irn.Array.sin(X_test.reshape([50]))
    mse = ((preds - y_true) ** 2).mean()

    print("MSE:", mse)





