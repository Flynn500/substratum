"""A Rust-based ndarray library. """

from typing import Iterator, List, Sequence, overload

class Array:
    """N-dimensional array of float64 values."""

    def __init__(self, shape: Sequence[int], data: Sequence[float]) -> None:
        """Create array from shape and flat data list."""
        ...

    @staticmethod
    def zeros(shape: Sequence[int]) -> Array:
        """Create array filled with zeros."""
        ...

    @staticmethod
    def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
        """Create a 2D identity matrix with ones on the k-th diagonal."""
        ...

    @staticmethod
    def diag(v: Sequence[float], k: int | None = None) -> Array:
        """Create a 2D array with v on the k-th diagonal."""
        ...

    @staticmethod
    def outer(a: Sequence[float], b: Sequence[float]) -> Array:
        """Compute the outer product of two 1D arrays."""
        ...

    @property
    def shape(self) -> List[int]:
        """Get the shape as a list."""
        ...

    def get(self, indices: Sequence[int]) -> float:
        """Get element at indices."""
        ...

    def tolist(self) -> List[float]:
        """Return data as flat list."""
        ...

    def diagonal(self, k: int | None = None) -> Array:
        """Extract the k-th diagonal from a 2D array."""
        ...
    
    def transpose(self) -> Array:
        """Transpose a 2D matrix."""
        ...

    def t(self) -> Array:
        """Transpose (alias for transpose())."""
        ...

    def matmul(self, other: Array) -> Array:
        """Matrix multiplication."""
        ...

    def dot(self, other: Array) -> Array:
        """Dot/matrix product."""
        ...
    
    def cholesky(self) -> Array:
        """Compute Cholesky decomposition.

        Returns lower triangular matrix L where A = L @ L.T.

        Raises:
            ValueError: If matrix is not square or not positive-definite.
        """
        ...

    def qr(self) -> tuple[Array, Array]:
        """QR decomposition.

        Returns (Q, R) where A = Q @ R, Q is orthogonal and R is upper triangular.

        Raises:
            ValueError: If array is not 2D.
        """
        ...

    def eig(self) -> tuple[Array, Array]:
        """Compute eigenvalues and eigenvectors.

        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
            and eigenvectors is a 2D array with eigenvectors as columns.
            Eigenvalues are sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    def eig_with_params(self, max_iter: int = 1000, tol: float = 1e-10) -> tuple[Array, Array]:
        """Eigendecomposition with custom iteration parameters.

        Args:
            max_iter: Maximum number of QR iterations.
            tol: Convergence tolerance for off-diagonal elements.

        Returns:
            Tuple of (eigenvalues, eigenvectors).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    def eigvals(self) -> Array:
        """Compute eigenvalues only.

        More efficient than eig() when eigenvectors are not needed.
        Similar to numpy.linalg.eigvals.

        Returns:
            1D array of eigenvalues sorted by absolute value (descending).

        Raises:
            ValueError: If matrix is not square.
        """
        ...

    def __matmul__(self, other: Array) -> Array:
        """Matrix multiplication operator (@)."""
        ...


    def sin(self) -> Array: ...
    def cos(self) -> Array: ...
    def exp(self) -> Array: ...
    def sqrt(self) -> Array: ...
    def clip(self, min: float, max: float) -> Array: ...

    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> Array: ...
    def __iter__(self) -> Iterator[float]: ...


    @overload
    def __add__(self, other: Array) -> Array: ...
    @overload
    def __add__(self, other: float) -> Array: ...
    def __radd__(self, other: float) -> Array: ...

    @overload
    def __sub__(self, other: Array) -> Array: ...
    @overload
    def __sub__(self, other: float) -> Array: ...
    def __rsub__(self, other: float) -> Array: ...

    @overload
    def __mul__(self, other: Array) -> Array: ...
    @overload
    def __mul__(self, other: float) -> Array: ...
    def __rmul__(self, other: float) -> Array: ...

    @overload
    def __truediv__(self, other: Array) -> Array: ...
    @overload
    def __truediv__(self, other: float) -> Array: ...
    def __rtruediv__(self, other: float) -> Array: ...

    def __neg__(self) -> Array: ...
    def __repr__(self) -> str: ...


class Generator:
    """Random number generator."""

    def __init__(self) -> None:
        """Create time-seeded generator."""
        ...

    @staticmethod
    def from_seed(seed: int) -> Generator:
        """Create generator with explicit seed."""
        ...

    def uniform(self, low: float, high: float, shape: Sequence[int]) -> Array: ...
    def standard_normal(self, shape: Sequence[int]) -> Array: ...
    def normal(self, mu: float, sigma: float, shape: Sequence[int]) -> Array: ...
    def randint(self, low: int, high: int, shape: Sequence[int]) -> Array: ...


def zeros(shape: Sequence[int]) -> Array:
    """Create array filled with zeros."""
    ...

def eye(n: int, m: int | None = None, k: int | None = None) -> Array:
    """Create a 2D identity matrix with ones on the k-th diagonal."""
    ...

def diag(v: Sequence[float], k: int | None = None) -> Array:
    """Create a 2D array with v on the k-th diagonal."""
    ...

def outer(a: Sequence[float], b: Sequence[float]) -> Array:
    """Compute the outer product of two 1D arrays."""
    ...
