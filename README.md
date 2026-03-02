# IronForest
Fast spatial indexing and approximate density estimation for Python, powered by Rust. Zero external dependencies.

## Quickstart

Quickly find the k nearest neighbours in high-dimensional space using our Random Projection Tree.

```python
import ironforest as irn

dims = 256
n = 100_000
k = 10

#Randomly generate data
gen = irn.random.Generator.from_seed(0)
data = gen.uniform(0.0, 100.0, [n, dims])
query_point = ([50.0] * dims)

#Use random projections to reduce the dimensionality
reducer, points = irn.spatial.ProjectionReducer.fit_transform(data, 50)
tree = irn.spatial.KDTree.from_array(points, leaf_size=50)

#Use our reducer to convert the query
query_point = reducer.transform(query_point)
result = tree.query_knn(query_point, k=k)

#k nearest neighbours
for output_idx, original_idx in enumerate(result.indices):
    print(f"index: {original_idx}, dist: {result.distances[output_idx]}")

#print mean, meadian and max distances
print(f"{result.mean():.2f}, {result.median():.2f}, {result.radius():.2f}")
```


## Installation

`pip install ironforest`

You can also build with `maturin build --release` assuming maturin is installed.

## Spatial
Spatial trees support kNN, radius, and KDE queries. All spatial trees support serialization via `save()` & `load()`, alternatively you can use pickle.

- KDTree - axis-aligned splits, best for low-to-moderate dimensions
- BallTree - pivot-based splits, handles higher dimensions well
- VPTree - vantage-point splits, strong in general metric spaces
- RPTree - random-projection splits, strong in high dimensions with low intrinsic dimensionality. 
- MTree - pivot-based splits, supports dynamic insertion at the cost of query speed.
- AggTree - approximate KDE via aggregated nodes, tunable accuracy via atol
- ProjectionReducer - use random projections to reduce dimensionality for more effecient spatial queries.

___

Speed comparison of our KDTree vs SciPy & Scikit-Learn on a randomly generated uniform dataset. KDE is not exposed directly on either of their trees, but `algorithm="kd_tree"` was specified for scikit-learn's `KernelDensity` object for the below comparison.

<div align="center">

| Dataset | Structure | Build (s) | kNN (s) | Radius (s) | KDE (s) | k | radius | bandwidth |
|---|---|---|---|---|---|---|---|---|
| 50000x8 | sklearn KDTree | 0.048036 | 0.367485 | 1.977652 | 15.304560 | 10 | 0.5 | 0.5 |
| 50000x8 | scipy KDTree | 0.011791 | 0.152537 | 1.146680 | N/A | 10 | 0.5 | 0.5 |
| 50000x8 | irn KDTree | 0.010846 | 0.021540 | 0.093523 | 0.719237 | 10 | 0.5 | 0.5 |

</div>

KDTree is generally seen as the baseline spatial indexing tree. Our other trees scale better with dimensionality or provide better results depending on the nature of the dataset used, see `docs/spatial.md` & `docs/agg_tree.md` for more detailed information.

## Tree-Based Models
IronForest includes tree-based ML models that run entirely on the Rust core no external dependencies at runtime.

- Decision Trees
- Random Forest
- Isolation Forest

### Additional Models
We include a handful of additional models built on core features our library already supports.
- Linear Regression
- Local Regression

## Supporting Modules

These modules are not the primary focus of the library but we still expose them through our python bindings. Numpy and SciPy should be preffered if a wider array of linear algebra and statistical methods are needed, all our functions support `ArrayLike` inputs, which can be numpy NdArrays, python lists or our own internal Array object. Our arrays can also be converted to alternative  formats via to_numpy() & tolist() for display and use alongside other libraries.

### Array
- An N-dimensional array object with broadcasting
- Matrix operations & constructors
- Numpy interoperability via `to_numpy()` & `from_numpy()`

### NdUtils
- Array constructors
- numpy conversions
- Column stack (additional stack methods planned)
- Linspace

### Random
`Generator` object that can sample from uniform, normal, lognormal, gamma and beta distributions.

### Linalg
- Standard matrix methods and constructors.
- cholesky and eigen and qr decomposition.
- Least Squares & Weighted Least Squares solver.

### Stats
- Basic statistical methods for `Array` objects, mean, median, var std, and quantile.
- Pearson and Spearman correlation.

## Status
This is largely a learning project and the API is subject to change. Expect less new features going forwards. I intend to expand my tree engine to support a wider variety of decision-tree type ML algorithms, as well as adding a few additional nicher spatial indexing trees but my core focus for the forseeable future is ironing out the kinks of what's already here and documenting what we are capable of and where we fall short.

The core rationale for this project was to build these algorithms from the ground up to understand how they work under the hood. I'm not looking for contributors at this stage but always welcome suggestions and criticism.


