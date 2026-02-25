use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use crate::array::{NdArray, Shape};
use crate::spatial::{BallTree, KDTree, VPTree, AggTree, BruteForce, VantagePointSelection, DistanceMetric, KernelType, SpatialQuery};
use super::{PyArray, ArrayData, ArrayLike};

#[pyclass(name = "SpatialResult")]
pub struct PySpatialResult {
    #[pyo3(get)]
    indices: PyArray,
    #[pyo3(get)]
    distances: PyArray,
    #[pyo3(get)]
    counts: Option<PyArray>,
    n_queries: usize,
    k: Option<usize>,
}

//helper for spatial results
fn scalar_or_array(py: Python<'_>, values: Vec<f64>, is_single: bool) -> PyResult<Py<PyAny>> {
    if is_single {
        Ok(values[0].into_pyobject(py)?.into_any().unbind())
    } else {
        let n = values.len();
        Ok(PyArray {
            inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), values)),
        }.into_pyobject(py)?.into_any().unbind())
    }
}

impl PySpatialResult {
    pub fn from_single(indices: Vec<i64>, distances: Vec<f64>) -> Self {
        let n = indices.len();
        PySpatialResult {
            indices: PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)) },
            distances: PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)) },
            counts: None,
            n_queries: 1,
            k: None,
        }
    }

    pub fn from_batch_knn(indices: Vec<i64>, distances: Vec<f64>, n_queries: usize, k: usize) -> Self {
        PySpatialResult {
            indices: PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(vec![n_queries, k]), indices)) },
            distances: PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, k]), distances)) },
            counts: None,
            n_queries,
            k: Some(k),
        }
    }

    pub fn from_batch_radius(indices: Vec<i64>, distances: Vec<f64>, counts: Vec<i64>) -> Self {
        let n_queries = counts.len();
        let total = indices.len();
        PySpatialResult {
            indices: PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(total), indices)) },
            distances: PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(total), distances)) },
            counts: Some(PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n_queries), counts)) }),
            n_queries,
            k: None,
        }
    }

    fn per_query_distances(&self) -> Vec<&[f64]> {
        let dist_slice = self.distances.as_float().unwrap().as_slice();
        if self.n_queries == 1 {
            vec![dist_slice]
        } else if let Some(k) = self.k {
            dist_slice.chunks_exact(k).collect()
        } else {
            let counts = self.counts.as_ref().unwrap();
            let mut offset = 0;
            counts.as_float().unwrap().as_slice().iter().map(|&c| {
                let c = c as usize;
                let slice = &dist_slice[offset..offset + c];
                offset += c;
                slice
            }).collect()
        }
    }

    fn per_query_indices(&self) -> Vec<&[i64]> {
        let idx_slice = self.indices.as_int().unwrap().as_slice();
        if self.n_queries == 1 {
            vec![idx_slice]
        } else if let Some(k) = self.k {
            idx_slice.chunks_exact(k).collect()
        } else {
            let counts = self.counts.as_ref().unwrap();
            let mut offset = 0;
            counts.as_float().unwrap().as_slice().iter().map(|&c| {
                let c = c as usize;
                let slice = &idx_slice[offset..offset + c];
                offset += c;
                slice
            }).collect()
        }
    }
}

#[pymethods]
impl PySpatialResult {
    // fn split(&self, py: Python<'_>) -> PyResult<Py<PyAny>>{

    // }

    fn mean_distance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let means: Vec<f64> = self.per_query_distances().iter().map(|d| {
            if d.is_empty() { f64::NAN } else { d.iter().sum::<f64>() / d.len() as f64 }
        }).collect();
        scalar_or_array(py, means, self.n_queries == 1)
    }

    fn min_distance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mins: Vec<f64> = self.per_query_distances().iter().map(|d| {
            d.iter().copied().fold(f64::NAN, f64::min)
        }).collect();
        scalar_or_array(py, mins, self.n_queries == 1)
    }

    fn max_distance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let maxes: Vec<f64> = self.per_query_distances().iter().map(|d| {
            d.iter().copied().fold(f64::NAN, f64::max)
        }).collect();
        scalar_or_array(py, maxes, self.n_queries == 1)
    }

    fn median_distance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let medians: Vec<f64> = self.per_query_distances().iter().map(|d| {
            if d.is_empty() { return f64::NAN; }
            let mut sorted: Vec<f64> = d.to_vec();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }).collect();
        scalar_or_array(py, medians, self.n_queries == 1)
    }

    fn count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let counts: Vec<f64> = self.per_query_distances().iter()
            .map(|d| d.len() as f64)
            .collect();
        scalar_or_array(py, counts, self.n_queries == 1)
    }

    fn centroid(&self, data: &PyArray) -> PyResult<PyArray> {
        let data_arr = data.as_float()?;
        let data_slice = data_arr.as_slice();
        let dim = data_arr.shape().dims()[1];
        let chunks = self.per_query_indices();

        let mut result = Vec::with_capacity(chunks.len() * dim);
        for indices in &chunks {
            if indices.is_empty() {
                result.extend(std::iter::repeat(f64::NAN).take(dim));
                continue;
            }
            let mut centroid = vec![0.0f64; dim];
            for &idx in *indices {
                let row = &data_slice[idx as usize * dim..(idx as usize + 1) * dim];
                for (c, &v) in centroid.iter_mut().zip(row) {
                    *c += v;
                }
            }
            let n = indices.len() as f64;
            for c in &mut centroid {
                *c /= n;
            }
            result.extend(centroid);
        }

        let n_queries = chunks.len();
        if self.n_queries == 1 {
            Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(dim), result)) })
        } else {
            Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, dim]), result)) })
        }
    }
}


fn get_tree_data(
    tree_indices: &[usize],
    raw_data: &[f64],
    n_points: usize,
    dim: usize,
    indices: Option<ArrayLike>,
) -> PyResult<PyArray> {
    use pyo3::exceptions::PyValueError;

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

macro_rules! impl_spatial_query_methods {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
            fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<PySpatialResult> {
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
                    Ok(PySpatialResult::from_batch_radius(all_indices, all_distances, counts))
                } else {
                    let query_slice = &queries_arr.as_slice()[..self.inner.dim];
                    let results = self.inner.query_radius(query_slice, radius);
                    let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                        .map(|(i, d)| (i as i64, d)).unzip();
                    Ok(PySpatialResult::from_single(indices, distances))
                }
            }

            fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<PySpatialResult> {
                let is_batch = query.ndim() == 2;
                let queries_arr = query.into_spatial_query_ndarray(self.inner.dim)?;
                let n_queries = queries_arr.shape().dims()[0];
                if is_batch {
                    let results = self.inner.query_knn_batch(&queries_arr, k);
                    let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                        .flatten()
                        .map(|(i, d)| (i as i64, d))
                        .unzip();
                    Ok(PySpatialResult::from_batch_knn(indices, distances, n_queries, k))
                } else {
                    let query_slice = &queries_arr.as_slice()[..self.inner.dim];
                    let results = self.inner.query_knn(query_slice, k);
                    let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                        .map(|(i, d)| (i as i64, d)).unzip();
                    Ok(PySpatialResult::from_single(indices, distances))
                }
            }

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

            #[pyo3(signature = (indices=None))]
            fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
                get_tree_data(self.inner.indices(), self.inner.data(), self.inner.n_points, self.inner.dim, indices)
            }
        }
    };
}

impl_spatial_query_methods!(PyBallTree);
impl_spatial_query_methods!(PyKDTree);
impl_spatial_query_methods!(PyVPTree);
impl_spatial_query_methods!(PyBruteForce);


#[pyclass(name = "BallTree")]
pub struct PyBallTree {
    inner: BallTree,
}

#[pymethods] //not an error ide only
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
}

#[pyclass(name = "KDTree")]
pub struct PyKDTree {
    inner: KDTree,
}

#[pymethods] //not an error ide only
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
}

#[pyclass(name = "VPTree")]
pub struct PyVPTree {
    inner: VPTree,
}

#[pymethods] //not an error ide only
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
}

#[pyclass(name = "BruteForce")]
pub struct PyBruteForce {
    inner: BruteForce,
}

#[pymethods] //not an error ide only
impl PyBruteForce {
    #[staticmethod]
    #[pyo3(signature = (array, metric="euclidean"))]
    fn from_array(array: &PyArray, metric: Option<&str>) -> PyResult<Self> {
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let tree = BruteForce::from_ndarray(array.as_float()?, metric);
        Ok(PyBruteForce { inner: tree })
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

    #[pyo3(signature = (queries=None, normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalize = normalize.unwrap_or(false);

        let queries_arr = if let Some(q) = queries {
            q.into_spatial_query_ndarray(self.inner.dim)?
        } else {
            NdArray::from_vec(
                Shape::new(vec![self.inner.n_points, self.inner.dim]),
                self.inner.data.clone()
            )
        };

        let result = self.inner.kernel_density(&queries_arr, normalize);

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result) }.into_pyobject(py)?.into_any().unbind())
        }
    }
}
