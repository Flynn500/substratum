import ironforest as irn

gen = irn.random.Generator.from_seed(0)
points = gen.uniform(0.0, 100.0, [100, 2])
tree = irn.spatial.VPTree.from_array(points, leaf_size=10)
query_point = [50.0, 50.0]
result = tree.query_knn(query_point, k=5)

for output_idx, original_idx in enumerate(result.indices):
    print(type(output_idx))
    print(f"point: {points[original_idx]}, dist: {result.distances[output_idx]:.2f}")

print(f"{result.median_distance()}, {result.mean_distance()}, {result.min_distance()}")

