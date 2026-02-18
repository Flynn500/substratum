# Spatial Module
The spatial module holds a variety of tree structures, each possesing kNN, raidus and KDE queries. These queries function nearly the same in each, although pruning rules differ between. They each vastly speed up these queries and excel in different scenarios.

**Wishlist**
- aNN trees,
  - Random Projection trees
  - aNN methods for current trees
- Dynamic Insertion Trees
  - M-Tree
  - R-Tree
- Simple Clustering Algorithms
  - K-Means
  - DBScan
 
## Benchmarks

I benchmarked my trees against SKlearn by sampling 100,000 points uniformly in two dimensions between 0-1. We then ran 500 batched KDE queries for each tree using euclidian distance and gaussian kernels. Note that a uniformly generated dataset does not equate to real world use cases.

Two dimensions was chosen here as our greedy pruning only leads to a greater speed up as we move up in dimensions.

### Performance
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
The reason for the difference is largely due to our pruning differences. Our pruning logic is a simple cutoff value, where branches beyond this value are ignored. This does not take into account the possibilty that there could be branches beyond this cuttoff value with enough children to impact the likelihood. Scenarios like this will generally only influence the likelihood value marginally but it is worth noting. SKLearn on the other hand takes into account these scenarios and as such their results are more reliable. 

The approximation test also is not fair to SKlearn as we are comparing an exact computation to an approximation, I am just trying to convey that in cases where query speed really matters over accuracy, the aggregate tree approximation option is well worth trying.

Scenarios where our trees may be better are ones where you are willing to tolerate potential inaccuracies and your data is high-dimensional. More robust pruning methods can devolve into brute force KDE calculations in these cases while more naive hard threshold methods are able to maintain some of the performance benifits of tree based KDE. For these high-dimension  scenarios, I'd recommend our vantage point tree or ball tree. Our vantage point tree isn't compared here as it doesn't have an SKLearn equivalent, but it generally performs best in these situations. If you need even more speed and care even less about error margins our aggregate tree is the best option. It is both much more effiecent with memory usage and its query speeds are much faster.

### Accuracy
This test was a simple 20 point query against 100,000 points distributed norammly in 4 dimensions with a bandwidth of 0.5. Our aggregate tree used a max span of 1.0, which is quite extreme, this value really shouldn't be set above the bandwidth.

Both results below are compared against scikit-learn's KdTree.

<div align="center">

KDTree                  |  IronForest KdTree  | IronForest AggTree|
------------------------|---------------------|-------------------|
Max Absolute Difference | 5.449846e-06        | 5.939465e+01      |
Mean Absolute Difference| 2.368243e-06        | 1.535262e+01      |


</div>

In this particular test our KdTree gives similar results to scikit-learn, even with the more naive pruning logic. It's important to note that this error may differ depending on the distribution of data you apply our trees to.

The aggregate tree has a higher error margin but significantly better performance and uses far less memory due to us compacting nodes. I'd recommend never setting the max span value above the bandwidth for most use-cases. Increasing the max span beyond this value introduces errors that grow non-linearly.

## Aggregate Tree

Our AggTree is a BallTree variant optimized for high query speeds & reduced memory usage at the sacrifice of accuracy. Our AggTree works by trying to reduce our dataset to a series of aggregate nodes based on proximity. Instead of summing the kernel contributions of significant points we sum a mixture from a smaller set of aggregates (alongside raw data that wasn't aggregated).

Tree construction works the exact same as our standard ball tree, the only difference being we stop as soon as we reach a hypersphere that has a diameter smaller than our max span parameter. We then calculate the centroid, variance and number of points in this aggregate node. Our min samples parameter prevents us from aggregating nodes with few children where the exact calculation is cheap. We then recurse through the tree and free all data that belongs to aggregate nodes, the only values we need to calculate their contribution are the centroid, count and variance.

For queries, we recurse through the tree pruning nodes that are too far away to make a meaninful contribution. This works the same as a ball tree until we reach an aggregate node. We use a second order to calculate our aggregate node contribution:

$$\hat{K} = n \cdot \left( K(r_c) + \frac{1}{2} K''(r_c) \cdot \sigma^2 \right)$$

Where r_c is the second derivative of the kernel with respect to r and variance is given by:

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \mu\|^2$$


## Optmizations
The biggest speedup I've implemented so far was making the trees more cache friendly. Previously the data array remained untouched, while we manipulated an index array to deal with in-tree computations. This seemed fine in principle as we want to return the indices as our result, but it is not cache friendly. After adding a reorder function we increased speeds by 30% across queries. This function just rearranges our data vector so that nodes close to each other a stored nearby. This makes it easier for the CPU to cache values as we aren't jumping to random points in our arrays.
