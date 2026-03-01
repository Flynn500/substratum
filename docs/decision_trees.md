# Tree Models

IronForest provides three tree-based ML models built on a shared Rust tree engine. The Python layer wraps the engine with a scikit-learn-style API: all models expose `fit(X, y)` and `predict(X)`.

**Contents**
- [Decision Tree](#decision-tree)
- [Random Forest](#random-forest)
- [Isolation Forest](#isolation-forest)
- [Engine Architecture](#engine-architecture)

---

## Decision Tree

General-purpose single-tree model for classification and regression.

### API

#### `DecisionTreeClassifier`

```python
from ironforest import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None, 
    criterion="gini",
    random_state=0,
)

clf.fit(X, y)
predictions = clf.predict(X)
```

#### `DecisionTreeRegressor`

```python
from ironforest import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=0,
)

reg.fit(X, y)
predictions = reg.predict(X)
```

Identical to the classifier, but uses MSE as the split criterion (no `criterion` parameter). Post-fit attributes are the same except `n_classes_` is not set.

**Impurity measures:**
- Gini: `1 - Σ pᵢ²`
- Entropy: `-Σ pᵢ log₂(pᵢ)`
- MSE: `(Σy² / n) - (Σy / n)²` clipped to ≥ 0 to guard against floating-point underflow

### Limitations

- **No feature importance.** Split statistics are not recorded after training.
- **Dense arrays only.** Input must be a contiguous f64 array. Sparse matrices are not supported.
- **No missing value handling.** NaN or Inf in input data produces undefined behaviour.
- **No post-hoc pruning.** Trees are grown to their maximum allowed depth and not reduced after the fact.
- **No multi-way splits.** All splits are binary; categorical features must be pre-encoded.
- **Classification labels must be 0-indexed integers.** Labels are expected to be `0, 1, ..., n_classes - 1`. The number of classes is inferred as `max(y) + 1`, so gaps in label values inflate `n_classes_` unnecessarily.

---

## Random Forest

Bootstrap-aggregated ensemble of decision trees. Reduces variance compared to a single tree by training each tree on a different bootstrap sample and aggregating predictions.

### API

#### `RandomForestClassifier`

```python
from ironforest import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    criterion="gini",
    random_state=42,
)

clf.fit(X, y)
predictions = clf.predict(X)
```

#### `RandomForestRegressor`

```python
from ironforest import RandomForestRegressor

reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42,
)

reg.fit(X, y)
predictions = reg.predict(X)
```

### Limitations

All [Decision Tree limitations](#limitations) apply, plus:

- **No out-of-bag (OOB) scoring.** Samples excluded from each bootstrap are not used for validation.
- **No feature importance.** Individual tree split statistics are not aggregated or exposed.
- **Memory scales linearly with `n_estimators`.** Each tree stores its full node vector; large forests with deep trees can be memory-intensive.
- **No `warm_start` or incremental fitting.** The entire forest is rebuilt on every call to `fit`.

---

## Isolation Forest

Unsupervised anomaly detection using random isolation trees. Anomalies are easier to isolate than normal points, so they tend to reach a leaf node in fewer splits, resulting in shorter path lengths.

### API

```python
from ironforest import IsolationForest

iforest = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.1,
    max_features=None,
    random_state=42,
)

iforest.fit(X)

labels = iforest.predict(X)
scores = iforest.score_samples(X)
scores = iforest.decision_function(X)
```

### Limitations

- **`contamination` must be in `(0, 0.5]`.** Values outside this range raise a `ValueError` at construction time.
- **Score range is ≈ [−1, 0].** Scores are negated engine output. This matches sklearn's convention but differs from the raw Rust values.
- **Dense arrays only; no missing value handling.** Same constraints as the supervised trees.
- **Fixed threshold.** The contamination rate is baked into `threshold_` at fit time. There is no mechanism to adjust the anomaly boundary at prediction time without refitting.
- **High-dimensional data.** Without feature subsampling, isolation trees struggle in very high-dimensional spaces because random splits rarely separate anomalies from normal points cleanly. Explicitly setting `max_features` to a small value can help.
- **No label supervision.** `fit` accepts `y` for API consistency but ignores it entirely. The model has no way to use known anomaly labels to improve detection.

---

## Engine Architecture

### Python / Rust blend

All three models fuse the same underlying tree engine, while python is used to manipulate the engine and expose our high level objects. The idea of this is that it should make implementing different decision-tree-style algorithms easier going forwards. I have plans to marry some of the KDE features from our spatial module with an isolation-style anomoly detection tree.

All the heavy computation however is delegated to our rust core, moving the trees entirely to the rust side would give very limited speed benifits.

### Parallelism

Ensemble training uses Rayon under the hood. All trees in a forest or isolation forest are built concurrently across available CPU threads with no user configuration required. Single `DecisionTree` models are single-threaded.

### Serialization

Tree and ensemble objects are in-memory only, there is currently no `save`/`load` method exposed for tree models. If persistence is needed, refit the model from saved training data. (The spatial module does support serialization via `save`/`load` for `BallTree`, `KDTree`, etc.)
