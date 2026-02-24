use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use crate::array::{NdArray, Shape};
use crate::spatial::{BallTree, KDTree, VPTree, AggTree, BruteForce, VantagePointSelection, DistanceMetric, KernelType, SpatialQuery};
use super::{PyArray, ArrayData, ArrayLike};

/// Shared helper: reconstruct original-order data from a reordered tree and return
/// the rows at the requested original indices (or all rows if `indices` is None).
fn get_tree_data(
    tree_indices: &[usize],
    raw_data: &[f64],
    n_points: usize,
    dim: usize,
    indices: Option<ArrayLike>,
) -> PyResult<PyArray> {
    use pyo3::exceptions::PyValueError;

    // Reconstruct data in original index order
    let mut original_data = vec![0.0f64; n_points * dim];
    for (tree_pos, &orig_idx) in tree_indices.iter().enumerate() {
        original_data[orig_idx * dim..(orig_idx + 1) * dim]
            .copy_from_slice(&raw_data[tree_pos * dim..(tree_pos + 1) * dim]);
    }

    match indices {
        None => Ok(PyArray {
            inner: ArrayData::Float(NdArray::from_vec(
                Shape::new(vec![n_points, dim]),
                original_data,
            )),
        }),
        Some(idx) => {
            let idx_arr = idx.into_i64_ndarray()?;
            let k = idx_arr.len();
            let mut result = Vec::with_capacity(k * dim);
            for &orig_idx in idx_arr.as_slice() {
                let i = orig_idx as usize;
                if i >= n_points {
                    return Err(PyValueError::new_err(format!(
                        "Index {} out of bounds for tree with {} points",
                        orig_idx, n_points
                    )));
                }
                result.extend_from_slice(&original_data[i * dim..(i + 1) * dim]);
            }
            Ok(PyArray {
                inner: ArrayData::Float(NdArray::from_vec(
                    Shape::new(vec![k, dim]),
                    result,
                )),
            })
        }
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBallTree>()?;
    m.add_class::<PyKDTree>()?;
    m.add_class::<PyVPTree>()?;
    m.add_class::<PyAggTree>()?;
    m.add_class::<PyBruteForce>()?;
    Ok(())
}

fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric.to_lowercase().as_str() {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "chebyshev" => Ok(DistanceMetric::Chebyshev),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance metric '{}'. Valid options: 'euclidean', 'manhattan', 'chebyshev'",
            metric
        ))),
    }
}

fn parse_kernel(kernel: &str) -> PyResult<KernelType> {
    match kernel.to_lowercase().as_str() {
        "gaussian" => Ok(KernelType::Gaussian),
        "epanechnikov" => Ok(KernelType::Epanechnikov),
        "uniform" => Ok(KernelType::Uniform),
        "triangular" => Ok(KernelType::Triangular),
        _ => Err(PyValueError::new_err(format!(
            "Unknown kernel type '{}'. Valid options: 'gaussian', 'epanechnikov', 'uniform', 'triangular'",
            kernel
        ))),
    }
}

fn parse_vantage_selection(selection: &str) -> PyResult<VantagePointSelection> {
    match selection.to_lowercase().as_str() {
        "first" => Ok(VantagePointSelection::First),
        "random" => Ok(VantagePointSelection::Random),
        _ => Err(PyValueError::new_err(format!(
            "Unknown vantage point selection method '{}'. Valid options: 'first', 'random'",
            selection
        ))),
    }
}

#[pyclass(name = "BallTree")]
pub struct PyBallTree {
    inner: BallTree,
}

#[pymethods]
impl PyBallTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean"))]
    fn from_array(array: &PyArray, leaf_size: Option<usize>, metric: Option<&str>) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric_str = metric.unwrap_or("euclidean");
        let metric = parse_metric(metric_str)?;
        let tree = BallTree::from_ndarray(array.as_float()?, leaf_size, metric);
        Ok(PyBallTree { inner: tree })
    }

    fn query_radius(&self, py: Python<'_>, query: ArrayLike, radius: f64) -> PyResult<Py<PyAny>> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        if is_batch {
            let results = self.inner.query_radius_batch(&queries_arr, radius);
            let mut all_indices = Vec::new();
            let mut all_distances = Vec::new();
            let mut counts = Vec::with_capacity(results.len());
            for batch in results {
                counts.push(batch.len() as i64);
                for (i, d) in batch {
                    all_indices.push(i as i64);
                    all_distances.push(d);
                }
            }
            let total = all_indices.len();
            let n_queries = counts.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(total), all_indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(total), all_distances)) };
            let cnt = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n_queries), counts)) };
            Ok((idx, dst, cnt).into_pyobject(py)?.into_any().unbind())
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_radius(query_slice, radius);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) };
            Ok((idx, dst).into_pyobject(py)?.into_any().unbind())
        }
    }

    fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<(PyArray, PyArray)> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        let n_queries = queries_arr.shape().dims()[0];
        if is_batch {
            let results = self.inner.query_knn_batch(&queries_arr, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .flatten()
                .map(|(i, d)| (i as i64, d))
                .unzip();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(vec![n_queries, k]), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, k]), distances)) },
            ))
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_knn(query_slice, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) },
            ))
        }
    }

    #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        bandwidth: Option<f64>,
        kernel: Option<&str>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let bandwidth = bandwidth.unwrap_or(1.0);
        let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let normalize = normalize.unwrap_or(false);

        let queries_arr = if let Some(q) = queries {
            q.into_spatial_query_ndarray(self.inner.dim)?
        } else {
            NdArray::from_vec(
                Shape::new(vec![self.inner.n_points, self.inner.dim]),
                self.inner.data.clone()
            )
        };

        let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result) }.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Return data points at original indices in shape (n, dim).
    /// Call with no argument to get all points in original index order.
    #[pyo3(signature = (indices=None))]
    fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
        get_tree_data(self.inner.indices(), self.inner.data(), self.inner.n_points, self.inner.dim, indices)
    }
}

#[pyclass(name = "KDTree")]
pub struct PyKDTree {
    inner: KDTree,
}

#[pymethods]
impl PyKDTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean"))]
    fn from_array(array: &PyArray, leaf_size: Option<usize>, metric: Option<&str>) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric_str = metric.unwrap_or("euclidean");
        let metric = parse_metric(metric_str)?;
        let tree = KDTree::from_ndarray(array.as_float()?, leaf_size, metric);
        Ok(PyKDTree { inner: tree })
    }

    fn query_radius(&self, py: Python<'_>, query: ArrayLike, radius: f64) -> PyResult<Py<PyAny>> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        if is_batch {
            let results = self.inner.query_radius_batch(&queries_arr, radius);
            let mut all_indices = Vec::new();
            let mut all_distances = Vec::new();
            let mut counts = Vec::with_capacity(results.len());
            for batch in results {
                counts.push(batch.len() as i64);
                for (i, d) in batch {
                    all_indices.push(i as i64);
                    all_distances.push(d);
                }
            }
            let total = all_indices.len();
            let n_queries = counts.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(total), all_indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(total), all_distances)) };
            let cnt = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n_queries), counts)) };
            Ok((idx, dst, cnt).into_pyobject(py)?.into_any().unbind())
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_radius(query_slice, radius);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) };
            Ok((idx, dst).into_pyobject(py)?.into_any().unbind())
        }
    }

    fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<(PyArray, PyArray)> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        let n_queries = queries_arr.shape().dims()[0];
        if is_batch {
            let results = self.inner.query_knn_batch(&queries_arr, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .flatten()
                .map(|(i, d)| (i as i64, d))
                .unzip();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(vec![n_queries, k]), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, k]), distances)) },
            ))
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_knn(query_slice, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) },
            ))
        }
    }

    #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        bandwidth: Option<f64>,
        kernel: Option<&str>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let bandwidth = bandwidth.unwrap_or(1.0);
        let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let normalize = normalize.unwrap_or(false);

        let queries_arr = if let Some(q) = queries {
            q.into_spatial_query_ndarray(self.inner.dim)?
        } else {
            NdArray::from_vec(
                Shape::new(vec![self.inner.n_points, self.inner.dim]),
                self.inner.data.clone()
            )
        };

        let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result) }.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Return data points at original indices in shape (n, dim).
    /// Call with no argument to get all points in original index order.
    #[pyo3(signature = (indices=None))]
    fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
        get_tree_data(self.inner.indices(), self.inner.data(), self.inner.n_points, self.inner.dim, indices)
    }
}

#[pyclass(name = "VPTree")]
pub struct PyVPTree {
    inner: VPTree,
}

#[pymethods]
impl PyVPTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", selection="first"))]
    fn from_array(array: &PyArray, leaf_size: Option<usize>, metric: Option<&str>, selection: Option<&str>) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric_str = metric.unwrap_or("euclidean");
        let metric = parse_metric(metric_str)?;
        let selection_str = selection.unwrap_or("first");
        let selection_method = parse_vantage_selection(selection_str)?;
        let tree = VPTree::from_ndarray(array.as_float()?, leaf_size, metric, selection_method);
        Ok(PyVPTree { inner: tree })
    }

    fn query_radius(&self, py: Python<'_>, query: ArrayLike, radius: f64) -> PyResult<Py<PyAny>> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        if is_batch {
            let results = self.inner.query_radius_batch(&queries_arr, radius);
            let mut all_indices = Vec::new();
            let mut all_distances = Vec::new();
            let mut counts = Vec::with_capacity(results.len());
            for batch in results {
                counts.push(batch.len() as i64);
                for (i, d) in batch {
                    all_indices.push(i as i64);
                    all_distances.push(d);
                }
            }
            let total = all_indices.len();
            let n_queries = counts.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(total), all_indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(total), all_distances)) };
            let cnt = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n_queries), counts)) };
            Ok((idx, dst, cnt).into_pyobject(py)?.into_any().unbind())
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_radius(query_slice, radius);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) };
            Ok((idx, dst).into_pyobject(py)?.into_any().unbind())
        }
    }

    fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<(PyArray, PyArray)> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        let n_queries = queries_arr.shape().dims()[0];
        if is_batch {
            let results = self.inner.query_knn_batch(&queries_arr, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .flatten()
                .map(|(i, d)| (i as i64, d))
                .unzip();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(vec![n_queries, k]), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, k]), distances)) },
            ))
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_knn(query_slice, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) },
            ))
        }
    }

    #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        bandwidth: Option<f64>,
        kernel: Option<&str>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let bandwidth = bandwidth.unwrap_or(1.0);
        let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let normalize = normalize.unwrap_or(false);

        let queries_arr = if let Some(q) = queries {
            q.into_spatial_query_ndarray(self.inner.dim)?
        } else {
            NdArray::from_vec(
                Shape::new(vec![self.inner.n_points, self.inner.dim]),
                self.inner.data.clone()
            )
        };

        let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result) }.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Return data points at original indices in shape (n, dim).
    /// Call with no argument to get all points in original index order.
    #[pyo3(signature = (indices=None))]
    fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
        get_tree_data(self.inner.indices(), self.inner.data(), self.inner.n_points, self.inner.dim, indices)
    }
}

#[pyclass(name = "AggTree")]
pub struct PyAggTree {
    inner: AggTree,
}

#[pymethods]
impl PyAggTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", kernel="gaussian", bandwidth=1.0, atol=0.01))]
    fn from_array(
        array: &PyArray,
        leaf_size: Option<usize>,
        metric: Option<&str>,
        kernel: Option<&str>,
        bandwidth: Option<f64>,
        atol: Option<f64>,
    ) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let kernel = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let bandwidth = bandwidth.unwrap_or(1.0);
        let atol = atol.unwrap_or(0.01);
        let tree = AggTree::from_ndarray(array.as_float()?, leaf_size, metric, kernel, bandwidth, atol);
        Ok(PyAggTree { inner: tree })
    }

    #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        bandwidth: Option<f64>,
        kernel: Option<&str>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let bandwidth = bandwidth.unwrap_or(1.0);
        let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let normalize = normalize.unwrap_or(false);

        let queries_arr = if let Some(q) = queries {
            q.into_spatial_query_ndarray(self.inner.dim)?
        } else {
            NdArray::from_vec(
                Shape::new(vec![self.inner.n_points, self.inner.dim]),
                self.inner.data.clone()
            )
        };

        let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result) }.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[pyclass(name = "BruteForce")]
pub struct PyBruteForce {
    inner: BruteForce,
}

#[pymethods]
impl PyBruteForce {
    #[staticmethod]
    #[pyo3(signature = (array, metric="euclidean"))]
    fn from_array(array: &PyArray, metric: Option<&str>) -> PyResult<Self> {
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let tree = BruteForce::from_ndarray(array.as_float()?, metric);
        Ok(PyBruteForce { inner: tree })
    }

    fn query_radius(&self, py: Python<'_>, query: ArrayLike, radius: f64) -> PyResult<Py<PyAny>> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        if is_batch {
            let results = self.inner.query_radius_batch(&queries_arr, radius);
            let mut all_indices = Vec::new();
            let mut all_distances = Vec::new();
            let mut counts = Vec::with_capacity(results.len());
            for batch in results {
                counts.push(batch.len() as i64);
                for (i, d) in batch {
                    all_indices.push(i as i64);
                    all_distances.push(d);
                }
            }
            let total = all_indices.len();
            let n_queries = counts.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(total), all_indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(total), all_distances)) };
            let cnt = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n_queries), counts)) };
            Ok((idx, dst, cnt).into_pyobject(py)?.into_any().unbind())
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_radius(query_slice, radius);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            let idx = PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) };
            let dst = PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) };
            Ok((idx, dst).into_pyobject(py)?.into_any().unbind())
        }
    }

    fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<(PyArray, PyArray)> {
        let is_batch = query.ndim() == 2;
        let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
        let n_queries = queries_arr.shape().dims()[0];
        if is_batch {
            let results = self.inner.query_knn_batch(&queries_arr, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .flatten()
                .map(|(i, d)| (i as i64, d))
                .unzip();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(vec![n_queries, k]), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, k]), distances)) },
            ))
        } else {
            let query_slice = &queries_arr.as_slice()[..self.inner.dim];
            let results = self.inner.query_knn(query_slice, k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d)).unzip();
            let n = indices.len();
            Ok((
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) },
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) },
            ))
        }
    }

    #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        bandwidth: Option<f64>,
        kernel: Option<&str>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let bandwidth = bandwidth.unwrap_or(1.0);
        let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let normalize = normalize.unwrap_or(false);

        let queries_arr = if let Some(q) = queries {
            q.into_spatial_query_ndarray(self.inner.dim)?
        } else {
            NdArray::from_vec(
                Shape::new(vec![self.inner.n_points, self.inner.dim]),
                self.inner.data.clone()
            )
        };

        let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result) }.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Return data points at original indices in shape (n, dim).
    /// Call with no argument to get all points in original index order.
    #[pyo3(signature = (indices=None))]
    fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
        get_tree_data(self.inner.indices(), self.inner.data(), self.inner.n_points, self.inner.dim, indices)
    }
}
