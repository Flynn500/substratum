import ironforest as irn

#MAKE SLICES ACCEPT NDARRAY

def _tricube(d, max_d):
    """Tricube kernel: (1 - (d/max_d)^3)^3 for d < max_d, else 0."""
    if max_d == 0.0:
        return irn.Array.ones(d.shape[0])
    u = d / max_d
    mask = u < 1.0
    w = (1.0 - u ** 3) ** 3
    return w * mask

def _epanechnikov(d, max_d):
    if max_d == 0.0:
        return irn.Array.ones(d.shape[0])
    u = d / max_d
    mask = u < 1.0
    return (1.0 - u ** 2) * mask

def _gaussian(d, max_d):
    u = d / (max_d / 3.0) if max_d > 0.0 else d
    return irn.Array.exp(-0.5 * u ** 2)

class LocalRegression:
    KERNELS = {
        "tricube": _tricube,
        "epanechnikov": _epanechnikov,
        "gaussian": _gaussian,
    }

    def __init__(self, tree, y, kernel="tricube", degree=1, k=None):
        """
        Parameters
        ----------
        tree : spatial index (KDTree, BallTree, VPTree)
            Built from the predictor points X.
        y : irn.Array
            Response values, shape (n,).
        kernel : str
            One of 'tricube', 'epanechnikov', 'gaussian'.
        degree : int
            Local polynomial degree (1 or 2).
        k : int or None
            Number of neighbors. Defaults to n * 0.3 if None.
        """
        self.tree = tree
        self.y = y
        self.kernel = kernel
        self.degree = degree
        self.k = k
    
    @staticmethod
    def from_array(arr: irn.Array, y, kernel, degree, k=None) -> "LocalRegression":
        """
        Parameters
        ----------
        arr : irn.Array
            Data to build BallTree tree with.
        y : irn.Array
            Response values, shape (n,).
        kernel : str
            One of 'tricube', 'epanechnikov', 'gaussian'.
        degree : int
            Local polynomial degree (1 or 2).
        k : int or None
            Number of neighbors. Defaults to n * 0.3 if None.
        """
        tree = irn.spatial.BallTree.from_array(arr)
        return LocalRegression(tree, y, kernel, degree, k)

    def _build_design(self, X_local):
        """Build design matrix from local neighbor coordinates.
        
        X_local: shape (k, p) array of neighbor positions.
        Returns column_stack of [1, x1, x2, ...] for degree=1,
        or [1, x1, x2, ..., x1^2, x2^2, ...] for degree=2.
        """
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
        Parameters
        ----------
        X_query : ArrayLike
            Query points, shape (m, p) or (p,) for a single point.

        Returns
        -------
        irn.Array of predicted values, shape (m,).
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
            indices, distances = self.tree.query_knn(q, k)

            max_d = distances[k - 1]
            w = kernel_fn(distances, max_d)

            X_local = self._get_points(indices.tolist())

            y_local = self.y[indices].reshape([k, 1])

            A = self._build_design(X_local)
            q_design = self._build_design(q.reshape([1, q.shape[0]]))

            coeffs, _ = irn.linalg.weighted_lstsq(A, y_local, w)

            preds[i] = (q_design @ coeffs).item()

        if single:
            return preds[0]
        return preds




