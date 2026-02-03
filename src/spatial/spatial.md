# Spatial Module
The spatial module holds a variety of tree structures, each possesing kNN, raidus and KDE and KDE approximation queries. These queries function nearly the same in each, although pruning rules differ between. They each vastly speed up these queries and excel in different scenarios.

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
 
## Performance

I benchmarked my trees against SKlearn by sampling 100,000 points uniformly in two dimensions between 0-1. We then ran 500 batched KDE queries for each tree using euclidian distance and gaussian kernels. Note that a uniformly generated dataset does not equate to real world use cases. 

<div align="center">
  
KDTree       | SKlearn      |  substratum  | substratum approx |
-------------|--------------|--------------|-------------------|
build_min    | 0.038591 sec | 0.012436 sec | 0.013954 sec      |
build_mean   | 0.040955 sec | 0.014963 sec | 0.014673 sec      |
query_min    | 0.929484 sec | 0.607634 sec | 0.000024 sec      |
query_mean   | 0.986913 sec | 0.635967 sec | 0.000034 sec      |

<br>

BallTree       | SKlearn      |  substratum | substratum approx |
-------------|--------------|---------------|-------------------|
build_min    | 0.037026 sec | 0.019365 sec | 0.019431 sec       |
build_mean   | 0.038487 sec | 0.014963 sec | 0.020502 sec       |
query_min    | 0.948515 sec | 0.607534 sec | 0.000026 sec       |
query_mean   | 0.996769 sec | 0.639607 sec | 0.000040 sec       |

</div>
The reason for the difference is largely due to our pruning differences. Our pruning logic is a simple cutoff value, where branches beyond this value are ignored. This does not take into account the possibilty that there could be branches beyond this cuttoff value with enough children to impact the likelihood. Scenarios like this will generally only influence the likelihood value marginally but it is worth noting. SKLearn on the other hand takes into account these scenarios and as such their results are more reliable. 

The approximation test also is not fair to SKlearn as we are comparing an computation to an approximation, I am just trying to convey that in cases where query speed really matters over accuracy, the approximation option is well worth trying.

Scenarios where our trees may be better are ones where you are willing to tolerate potential inaccuracies and your data is high-dimensional. More robust pruning methods can devolve into brute force KDE calculations in these cases while more naive hard threshold methods are able to maintain some of the performance benifits of tree based KDE. For these high-dimension  scenarios, I'd recommend our vantage point tree or ball tree. Our vantage point tree isn't compared here as it doesn't have an SKLearn equivalent, but it generally performs best in these situations. 

### Optmizations
The biggest speedup I've implemented so far was making the trees more cache friendly. Previously the data array remained untouched, while we manipulated an index array to deal with in-tree computations. This seemed fine in principle as we want to return the indices as our result, but it is not cache friendly. After adding a reorder function we increased speeds by 30% across queries. This function just rearranges our data vector so that nodes close to each other a stored nearby. This makes it easier for the CPU to cache values as we aren't jumping to random points in our arrays.

KDE Approximation: This is somewhat home-baked, I'm sure it has been done before. It does violate several of the assumptions of KDE and the results it gives are not statistically correct. Our KDE approximation works by two criteria, min_samples or span. If a node has either a span and at least minsamples children, we use a single kernel for the entire node (increasing its size by the average possible kernel contribution and its weight by its number of children). This works very well for the particular use case I needed it for, where we had regions of very dense points that could easily be treated as a single kernel instead of summing the contributions of each point individually. In my particular use-case we just needed to check if the likelihood met some threshold, so the loss in precision was largely inconsequential after raising this threshold to account for error.
