# IronForest
Fast spatial indexing and approximate density estimation for Python, powered by Rust. Zero external dependencies.

## Quickstart

```python
import ironforest as irn

gen = irn.random.Generator.from_seed(0)
points = gen.uniform(0.0, 100.0, [100, 2])

tree = irn.spatial.VPTree.from_array(points, leaf_size=10)

query_point = [50.0, 50.0]
neighbors = tree.query_knn(query_point, k=5)

print(neighbors)
```
Output: 
`[(28, 7.435055148638418), (34, 7.549053283615578), (3, 8.55705353151589), (43, 8.961446938534534), (45, 9.772273124615534)]`

## Installation

`pip install ironforest`

You can also build with `maturin build --release` assuming maturin is installed.

## Spatial Trees
Spatial trees support kNN, radius, and KDE queries.

- KDTree - axis-aligned splits, best for low-to-moderate dimensions
- BallTree - metric-based splits, handles higher dimensions well
- VPTree - vantage-point splits, strong in general metric spaces
- AggTree - approximate KDE via aggregated nodes, tunable accuracy via atol

## Tree-Based Models
IronForest includes tree-based ML models that run entirely on the Rust core no external dependencies at runtime.

- Decision Trees
- Random Forest
- Isolation Forest

## Supporting Modules

### Array
- An N-dimensional array object with broadcasting
- Matrix operations & constructors

### Random
`Generator` object that can sample from uniform, normal, lognormal, gamma and beta distributions. Support for additional distributions is planned.

### Linalg
- Standard matrix methods and constructors.
- cholesky and eigen and qr decomposition.
- Least Squares & Weighted Least Squares solver.

### Stats
- Basic statistical methods for `Array` objects, mean, var std, and quantile.
- Pearson and Spearman correlation.

## Rationale
The core rational behind this project was to deepen my understanding of how libraries I use on a regular basis work, and learn how to write python bindings to offload computationally expensive tasks to tools better suited. Like dubious, I focused on building things from the ground up. I didn't want to glue dependecies together to get something functional, I wanted to understand from input to output how these algorithms worked under the hood. I chose rust because I had read the book around a year prior to starting this project, and pyo3 bindings are relitively easy to get working. This library is only exposed through python as that's where I've actually use its features. I don't intend to package this as a rust create at this stage. 

Because of the nature of this project I don't intend to bring on any other contributors at this point, I am however always open to suggestions and criticism. 

## Status
This is largely a learning project and the API is subject to change. Expect less new features going forwards. I intend to expand my tree engine to support a wider variety of decision-tree type ML algorithms, as well as adding a few additional nicher spatial indexing trees but my core focus for the forseeable future is ironing out the kinks of what's already here and documenting what we are capable of and where we fall short.

The core rationale for this project was to build these algorithms from the ground up to understand how they work under the hood. I'm not looking for contributors at this stage but always welcome suggestions and criticism.


