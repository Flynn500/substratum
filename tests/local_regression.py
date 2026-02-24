import ironforest as irn

rng = irn.random.Generator.from_seed(4)

n = 250
X = rng.uniform(0.0, 6.28, [n, 1])
y = irn.Array.sin(X.ravel()) + rng.normal(0.0, 0.2, [n])

X_test = irn.ndutils.linspace(0.0, 6.28, 150).reshape([150, 1])
y_true = irn.Array.sin(X_test.ravel())

for k in [5, 10, 20, 40, 80]:
    tree = irn.spatial.KDTree.from_array(X)
    model = irn.models.LocalRegression(tree, y, k=k)

    y_pred = model.predict(X_test)
    mse = ((y_pred - y_true) ** 2).mean()
    print(f"k={k}, mse={mse}")
