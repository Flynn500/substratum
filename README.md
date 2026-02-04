# IronForest
IronForest is a rust-powered python library supporting spatial queries, array-based computation and tree-based machine learning. 

I started this project to support my previous python library dubious, a project which had the personal constraint of no external dependecies other than numpy. IronForest started as a way for me to eliminate numpy as dubious's lone dependency but has since grown into a standalone project as I have pivoted towards spatial indexing trees and tree-based models. The core rational behind this project was to get a better grasp of how libraries I use on a regular basis work, and learn how to write python bindings to offload computationally expensive tasks to tools better suited. Like dubious, I focused on building things from the ground up. I didn't want to glue dependecies together to get something functional, I wanted to understand from input to output how these algorithms worked under the hood. I chose rust because I had read the book around a year prior to starting this project, and pyo3 bindings are relitively easy to get working. This library is only exposed through python as that's where I've actually needed its features and I don't intend to package this as a rust create at this stage.

## Status
This is largely a learning project and the API is subject to change. We achieve similar performance to numpy (beating them in a rare few) across most operations but basic broadcasting arithmetic is around 4x slower in most cases. I intend to add a few unsafe methods to speed things up where applicable but I don't intend to optimize much further at this stage. 

## Installation
`pip install ironforest`

You can also build with `maturin build --release` assuming maturin is installed.

## Quickstart

```python
import ironforest as irn

a = irn.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
b = irn.Array([2, 2], [5.0, 6.0, 7.0, 8.0])

print(f"a @ b = {(a @ b).tolist()}")
```
Output: 
`a @ b = [[19.0, 22.0], [43.0, 50.0]]`

## Features
- Array, an N-dimensional array object with broadcasting
- matrix operations and constructors
- KDTree, BallTree & VPTree with knn radius KDE and KDE approx queries
- Linear and Local Regression
- Decision trees, Random Forest and Isolation Forest
- cholesky, qr and eigen decomposition
- Least Squares & Weighted Least Squares Solver
- Random sampling from uniform, normal lognormal, gamma and beta distributions.
- Statistical methods (mean, median, var, std, quantile)
- Pearson and Spearman correlation

### Top-level
- `ironforest.Array`

### Modules
- [ironforest.linalg](#linalg)
- [ironforest.stats](#stats)
- [ironforest.random](#random)
- [ironforest.spatial](#spatial)
- - [ironforest.model](#model)

## Examples
```python
import ironforest as irn

a = irn.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
b = irn.Array([2, 2], [5.0, 6.0, 7.0, 8.0])

print(f"a + b = {(a + b).tolist()}")
print(f"a * b = {(a * b).tolist()}")
```
Output: 
`
a + b = [[6.0, 8.0], [10.0, 12.0]]
a * b = [[5.0, 12.0], [21.0, 32.0]]
`
```python
import ironforest as irn

gen = irn.Generator.from_seed(123)

uniform = gen.uniform(0.0, 1.0, [2, 3])
print(f"Uniform [0, 1): {uniform.tolist()}")

normal = gen.standard_normal([2, 3])
print(f"Standard normal: {normal.tolist()}")
```
Output: 
`Uniform [0, 1): [0.19669435215621578, 0.9695722925002218, 0.46744032361670884, 0.12698379756585432]
Standard normal: [-0.0008585765206425146, 1.4733334715623352, -1.16180050645278, -0.772101732825336]`

## Modules

### Random
`Generator` object that can sample from uniform, normal, lognormal, gamma and beta distributions. Support for additional distributions is planned.

### Linalg
- Standard matrix methods and constructors.
- cholesky and eigen and qr decomposition.
- Least Squares & Weighted Least Squares solver.

### Stats
- Basic statistical methods for `Array` objects, mean, var and quantile.
- Pearson and Spearman correlation.

### Spatial
- `KDTree` kNN, Kernel Density Estimation and radius queries. 
- `BallTree` with kNN, Kernel Density Estimation and radius queries. 
- `VPTree` with kNN, Kernel Density Estimation and radius queries.

### Model
- Linear Regression
- Local Regression
- Decision Trees (soon)
- Random Forest (soon)
- Isolation Forest (soon)
