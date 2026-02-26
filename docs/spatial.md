# Spatial Module
The spatial module holds a variety of tree structures, each possesing kNN, raidus and KDE queries. These queries function nearly the same in each, although pruning rules differ between. They each vastly speed up these queries and excel in different scenarios.

**Wishlist**
- aNN trees,
  - Random Projection trees
  - aNN methods for current trees
- Dynamic Insertion Trees
  - M-Tree
  - R-Tree
 
## Methods

### query_radius()

Find all points within a given radius of the query point.

**Args**:  
query: Query point (scalar, list, or array-like).  
radius: Search radius. All points with distance <= radius are returned.

**Returns**:
Tuple of (indices, distances) where indices is an int64 Array and
distances is a float64 Array, both of length equal to the number of
points found, in arbitrary order.

### query_knn()

Find the k nearest neighbors to the query point.

**Args**:  
query: Query point (scalar, list, or array-like).  
k: Number of nearest neighbors to return.

**Returns**:
Tuple of (indices, distances) where indices is an int64 Array and
distances is a float64 Array, both sorted by distance (closest first).
The indices can be used to look up the actual points in the original data.

### kernel_density()

Estimate kernel density at a single query point.

**Args**:  
queries: Single query point (scalar, list, or array-like).
bandwidth: Bandwidth (smoothing parameter) for the kernel. Larger values
    produce smoother estimates. Defaults to 1.0.

kernel: Kernel function to use for density estimation. Options are:
- "gaussian": Gaussian (normal) kernel (default)
- "epanechnikov": Epanechnikov kernel
- "uniform": Uniform (rectangular) kernel
- "triangular": Triangular kernel

Normalize: Bool to control whether normalized values are returned.

**Returns**:
    Density estimate at the query point (float).




## Benchmarks

I benchmarked my trees against SKlearn by sampling 100,000 points uniformly in two dimensions between 0-1. We then ran 500 batched KDE queries for each tree using euclidian distance and gaussian kernels. Note that a uniformly generated dataset does not equate to real world use cases.


### Performance

**Radius & kNN Queries**

<div align="center">

| Dataset | Structure | Build (s) | kNN (s) | Radius (s) | k | radius |
|---|---|---|---|---|---|---|
| 10000x8 | sklearn.KDTree | 0.005848 | 0.289484 | 0.496998 | 10 | 0.5 |
| 10000x8 | sklearn.BallTree | 0.004917 | 0.466802 | 0.351760 | 10 | 0.5 |
| 10000x8 | irn.KDTree | 0.001696 | 0.013104 | 0.036566 | 10 | 0.5 |
| 10000x8 | irn.BallTree | 0.008095 | 0.058858 | 0.060451 | 10 | 0.5 |
| 50000x8 | sklearn.KDTree | 0.066481 | 0.390550 | 2.064392 | 10 | 0.5 |
| 50000x8 | sklearn.BallTree | 0.035018 | 0.910292 | 1.143048 | 10 | 0.5 |
| 50000x8 | irn.KDTree | 0.012981 | 0.028325 | 0.140391 | 10 | 0.5 |
| 50000x8 | irn.BallTree | 0.054803 | 0.114011 | 0.245173 | 10 | 0.5 |

</div>

One reason our implementations are faster is that queries are parallelized using rayon. We get a huge speed up running multiple queries simultaneously. Scikit-learn supports this through other methods but not on these trees directly. 

Another big reason for the increase is that once you make a IronForest Array, the data is already in our rust core. There is no handling of python objects input -> tree -> output all stays on the rust side of things.

**Kernel Density Estimation**

<div align="center">
  
KDTree       | SKlearn      |  IronForest  | IronForest AggTree|
-------------|--------------|--------------|-------------------|
build_min    | 0.035439 sec | 0.014716 sec | 0.001253 sec      |
build_mean   | 0.037059 sec | 0.016116 sec | 0.001534 sec      |
query_min    | 2.204914 sec | 0.599856 sec | 1.2100e-05 sec    |
query_mean   | 2.252003 sec | 0.613925 sec | 1.6940e-05 sec    |

<br>

BallTree     | SKlearn      |  IronForest  | IronForest AggTree |
-------------|--------------|--------------|--------------------|
build_min    | 0.034704 sec | 0.020255 sec | 0.001253 sec       |
build_mean   | 0.037933 sec | 0.021643 sec | 0.001534 sec       |
query_min    | 1.890896 sec | 0.572458 sec | 1.2100e-05 sec     |
query_mean   | 2.019398 sec | 0.577282 sec | 1.6940e-05 sec     |



</div>
The approximation test is not fair to Scikit-Learn as we are comparing an exact computation to an approximation, I am just trying to convey that in cases where query speed really matters over accuracy, the aggregate tree approximation option is well worth trying.

### Accuracy
This test was a simple 20 point query against 100,000 points distributed norammly in 4 dimensions with a bandwidth of 0.5. Our aggregate tree used a max span of 1.0, which is extreme, this value really should never be set above the bandwidth.

Both results below are compared against scikit-learn's KdTree.

<div align="center">

KDTree                  |  IronForest KdTree  | IronForest AggTree|
------------------------|---------------------|-------------------|
Max Absolute Difference | 5.449846e-06        | 5.939465e+01      |
Mean Absolute Difference| 2.368243e-06        | 1.535262e+01      |


</div>

In this particular test our KdTree gives similar results to scikit-learn, even with the more naive pruning logic. It's important to note that this error may differ depending on the distribution of data you apply our trees to.

The aggregate tree has a higher error margin but significantly better performance and uses far less memory due to us compacting nodes. I'd recommend never setting the max span value above the bandwidth for most use-cases. Increasing the max span beyond this value introduces errors that grow non-linearly.

## General Optmizations
The biggest speedup I've implemented so far was making the trees more cache friendly. Previously the data array remained untouched, while we manipulated an index array to deal with in-tree computations. This seemed fine in principle as we want to return the indices as our result, but it is not cache friendly. After adding a reorder function we increased speeds by 30% across queries. This function just rearranges our data vector so that nodes close to each other a stored nearby. This makes it easier for the CPU to cache values as we aren't jumping to random points in our arrays.

Parallelization via Rayon is another huge win for batch queries. All our trees do this automatically, where if a query size is great enough it is split across cores. Because of the nature of these spatial queries each core is traversing a tree that won't be mutated at least until the query is over. This makes implementation as simple as using rayon's parallel iterator instead of a standard loop.
