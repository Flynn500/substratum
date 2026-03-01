import ironforest as irn

dims = 50
n = 100_000
k = 10

gen = irn.random.Generator.from_seed(0)
points = gen.uniform(0.0, 100.0, [n, dims])
tree = irn.spatial.RPTree.from_array(points, leaf_size=50)
query_point = [50.0] * 50
result = tree.query_knn(query_point, k=k)

#k nearest neighbours
for output_idx, original_idx in enumerate(result.indices):
    print(f"point: {points[original_idx]}, dist: {result.distances[output_idx]:.2f}")


#print mean, meadian and max distances
print(f"{result.mean():.2f}, {result.median():.2f}, {result.radius():.2f}")