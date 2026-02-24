use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError};
use pyo3::types::{PyAny, PyFloat, PyList, PySlice, PyTuple};
use numpy::{PyArrayMethods, PyReadonlyArrayDyn};
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayData, ArrayLike};

fn parse_dtype(dtype: Option<&str>) -> PyResult<bool> {
    match dtype.unwrap_or("float") {
        "float" | "float64" | "f64" => Ok(false),
        "int"   | "int64"   | "i64" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype '{}'; expected 'float' or 'int'",
            other
        ))),
    }
}

#[pymethods]
impl PyArray {
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    pub fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: if parse_dtype(dtype)? {
            ArrayData::Int(NdArray::zeros(Shape::new(shape)))
        } else {
            ArrayData::Float(NdArray::zeros(Shape::new(shape)))
        }})
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    pub fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: if parse_dtype(dtype)? {
            ArrayData::Int(NdArray::<i64>::ones(Shape::new(shape)))
        } else {
            ArrayData::Float(NdArray::<f64>::ones(Shape::new(shape)))
        }})
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype=None))]
    pub fn full(shape: Vec<usize>, fill_value: f64, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: if parse_dtype(dtype)? {
            ArrayData::Int(NdArray::filled(Shape::new(shape), fill_value as i64))
        } else {
            ArrayData::Float(NdArray::full(Shape::new(shape), fill_value))
        }})
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape=None, dtype=None))]
    pub fn asarray(data: ArrayLike, shape: Option<Vec<usize>>, dtype: Option<&str>) -> PyResult<Self> {
        let dtype_is_int = parse_dtype(dtype)?;
        let result_is_int = if dtype.is_none() {
            data.is_int()
        } else {
            dtype_is_int
        };

        if result_is_int {
            let arr = data.into_i64_ndarray()?;
            let arr = if let Some(s) = shape {
                let expected_size: usize = s.iter().product();
                if arr.len() != expected_size {
                    return Err(PyValueError::new_err(format!(
                        "Data length {} doesn't match shape {:?} (expected {})",
                        arr.len(), s, expected_size
                    )));
                }
                NdArray::from_vec(Shape::new(s), arr.as_slice().to_vec())
            } else {
                arr
            };
            Ok(PyArray { inner: ArrayData::Int(arr) })
        } else {
            let arr = data.into_ndarray()?;
            let arr = if let Some(s) = shape {
                let expected_size: usize = s.iter().product();
                if arr.len() != expected_size {
                    return Err(PyValueError::new_err(format!(
                        "Data length {} doesn't match shape {:?} (expected {})",
                        arr.len(), s, expected_size
                    )));
                }
                NdArray::from_vec(Shape::new(s), arr.as_slice().to_vec())
            } else {
                arr
            };
            Ok(PyArray { inner: ArrayData::Float(arr) })
        }
    }

    #[staticmethod]
    #[pyo3(signature = (n, m=None, k=None, dtype=None))]
    pub fn eye(n: usize, m: Option<usize>, k: Option<isize>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: if parse_dtype(dtype)? {
            ArrayData::Int(NdArray::<i64>::eye(n, m, k.unwrap_or(0)))
        } else {
            ArrayData::Float(NdArray::<f64>::eye(n, m, k.unwrap_or(0)))
        }})
    }

    #[staticmethod]
    #[pyo3(signature = (v, k=None))]
    pub fn diag(v: ArrayLike, k: Option<isize>) -> PyResult<Self> {
        let v_arr = v.into_ndarray()?;
        Ok(PyArray { inner: ArrayData::Float(NdArray::from_diag(&v_arr, k.unwrap_or(0))) })
    }

    #[pyo3(signature = (k=None))]
    fn diagonal(&self, k: Option<isize>) -> PyResult<PyArray> {
        let a = self.as_float()?;
        if a.ndim() != 2 {
            return Err(PyValueError::new_err("diagonal requires a 2D array"));
        }
        Ok(PyArray { inner: ArrayData::Float(a.diagonal(k.unwrap_or(0))) })
    }

    #[new]
    #[pyo3(signature = (shape, data, dtype=None))]
    fn new(shape: Vec<usize>, data: Vec<f64>, dtype: Option<&str>) -> PyResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(PyValueError::new_err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_size
            )));
        }
        if parse_dtype(dtype)? {
            let int_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
            Ok(PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(shape), int_data)) })
        } else {
            Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(shape), data)) })
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.dims()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.ndim_val()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            ArrayData::Float(_) => "float64",
            ArrayData::Int(_) => "int64",
        }
    }

    fn get(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ArrayData::Float(a) => a.get(&indices).copied()
                .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
                .and_then(|v| Ok(v.into_pyobject(py)?.into_any().unbind())),
            ArrayData::Int(a) => a.get(&indices).copied()
                .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
                .and_then(|v| Ok(v.into_pyobject(py)?.into_any().unbind())),
        }
    }

    pub fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ArrayData::Float(a) => {
                if a.shape().ndim() == 0 {
                    return Ok(PyFloat::new(py, a.as_slice()[0]).unbind().into_any());
                }
                self.float_to_pylist_recursive(a, py, 0, 0)
            }
            ArrayData::Int(a) => {
                if a.shape().ndim() == 0 {
                    return Ok(a.as_slice()[0].into_pyobject(py)?.into_any().unbind());
                }
                self.int_to_pylist_recursive(a, py, 0, 0)
            }
        }
    }

    fn sin(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.sin()) })
    }

    fn cos(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.cos()) })
    }

    fn exp(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.exp()) })
    }

    fn sqrt(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.sqrt()) })
    }

    fn clip(&self, min: f64, max: f64) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.clip(min, max)) })
    }

    fn tan(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.tan()) })
    }

    fn arcsin(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.asin()) })
    }

    fn arccos(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.acos()) })
    }

    fn arctan(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.atan()) })
    }

    fn log(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.log()) })
    }

    fn abs(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.abs()) })
    }

    fn sign(&self) -> PyResult<Self> {
        Ok(PyArray { inner: ArrayData::Float(self.as_float()?.sign()) })
    }

    fn sum(&self) -> PyResult<f64> {
        Ok(self.as_float()?.sum())
    }

    fn mean(&self) -> PyResult<f64> {
        Ok(self.as_float()?.mean())
    }

    fn var(&self) -> PyResult<f64> {
        Ok(self.as_float()?.var())
    }

    fn std(&self) -> PyResult<f64> {
        Ok(self.as_float()?.std())
    }

    fn median(&self) -> PyResult<f64> {
        Ok(self.as_float()?.median())
    }

    fn max(&self) -> PyResult<f64> {
        Ok(self.as_float()?.max())
    }

    fn min(&self) -> PyResult<f64> {
        Ok(self.as_float()?.min())
    }

    fn quantile(&self, py: Python<'_>, q: ArrayLike) -> PyResult<Py<PyAny>> {
        let a = self.as_float()?;
        match q {
            ArrayLike::Scalar(q_val) => {
                Ok(a.quantile(q_val).into_pyobject(py)?.into_any().unbind())
            }
            ArrayLike::IntScalar(s) => {
                Ok(a.quantile(s as f64).into_pyobject(py)?.into_any().unbind())
            }
            _ => {
                let q_arr = q.into_ndarray()?;
                Ok(PyArray {
                    inner: ArrayData::Float(a.quantiles(q_arr.as_slice())),
                }.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    fn any(&self) -> PyResult<bool> {
        Ok(self.as_float()?.any())
    }

    fn all(&self) -> PyResult<bool> {
        Ok(self.as_float()?.all())
    }

    fn __add__(&self, other: ArrayLike) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs + s) }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs + (s as f64)) }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs + &other.into_ndarray()?) }),
            },
            ArrayData::Int(lhs) => {
                if other.is_int() {
                    Ok(PyArray { inner: ArrayData::Int(lhs + &other.into_i64_ndarray()?) })
                } else {
                    let lhs_f = lhs.map(|x| x as f64);
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f + s) }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f + &other.into_ndarray()?) }),
                    }
                }
            }
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s + a) })
            }
            ArrayData::Int(a) => {
                if let Ok(s) = other.extract::<i64>() {
                    Ok(PyArray { inner: ArrayData::Int(s + a.clone()) })
                } else {
                    let s: f64 = other.extract()?;
                    let af = a.map(|x| x as f64);
                    Ok(PyArray { inner: ArrayData::Float(s + &af) })
                }
            }
        }
    }

    fn __sub__(&self, other: ArrayLike) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs - s) }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs - (s as f64)) }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs - &other.into_ndarray()?) }),
            },
            ArrayData::Int(lhs) => {
                if other.is_int() {
                    Ok(PyArray { inner: ArrayData::Int(lhs - &other.into_i64_ndarray()?) })
                } else {
                    let lhs_f = lhs.map(|x| x as f64);
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f - s) }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f - &other.into_ndarray()?) }),
                    }
                }
            }
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s - a) })
            }
            ArrayData::Int(a) => {
                if let Ok(s) = other.extract::<i64>() {
                    Ok(PyArray { inner: ArrayData::Int(s - a.clone()) })
                } else {
                    let s: f64 = other.extract()?;
                    let af = a.map(|x| x as f64);
                    Ok(PyArray { inner: ArrayData::Float(s - &af) })
                }
            }
        }
    }

    fn __mul__(&self, other: ArrayLike) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs * s) }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs * (s as f64)) }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs * &other.into_ndarray()?) }),
            },
            ArrayData::Int(lhs) => {
                if other.is_int() {
                    Ok(PyArray { inner: ArrayData::Int(lhs * &other.into_i64_ndarray()?) })
                } else {
                    let lhs_f = lhs.map(|x| x as f64);
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f * s) }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f * &other.into_ndarray()?) }),
                    }
                }
            }
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s * a) })
            }
            ArrayData::Int(a) => {
                if let Ok(s) = other.extract::<i64>() {
                    Ok(PyArray { inner: ArrayData::Int(s * a.clone()) })
                } else {
                    let s: f64 = other.extract()?;
                    let af = a.map(|x| x as f64);
                    Ok(PyArray { inner: ArrayData::Float(s * &af) })
                }
            }
        }
    }

    fn __truediv__(&self, other: ArrayLike) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs / s) }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs / (s as f64)) }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs / &other.into_ndarray()?) }),
            },
            ArrayData::Int(lhs) => {
                let lhs_f = lhs.map(|x| x as f64);
                match other {
                    ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f / s) }),
                    ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f / (s as f64)) }),
                    _ => Ok(PyArray { inner: ArrayData::Float(lhs_f / &other.into_ndarray()?) }),
                }
            }
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s / a) })
            }
            ArrayData::Int(a) => {
                let s: f64 = other.extract()?;
                let af = a.map(|x| x as f64);
                Ok(PyArray { inner: ArrayData::Float(s / &af) })
            }
        }
    }

    fn __neg__(&self) -> Self {
        match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(-a) },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(-a) },
        }
    }

    fn __richcmp__(&self, other: ArrayLike, op: pyo3::basic::CompareOp) -> PyResult<PyArray> {
        match op {
            pyo3::basic::CompareOp::Lt => self.apply_cmp(other, |a, b| a < b),
            pyo3::basic::CompareOp::Le => self.apply_cmp(other, |a, b| a <= b),
            pyo3::basic::CompareOp::Gt => self.apply_cmp(other, |a, b| a > b),
            pyo3::basic::CompareOp::Ge => self.apply_cmp(other, |a, b| a >= b),
            pyo3::basic::CompareOp::Eq => self.apply_cmp(other, |a, b| a == b),
            pyo3::basic::CompareOp::Ne => self.apply_cmp(other, |a, b| a != b),
        }
    }

    fn __pow__(&self, exp: ArrayLike, _modulo: Option<i64>) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(a) => {
                let data: Vec<f64> = match exp {
                    ArrayLike::Scalar(e) =>
                        a.as_slice().iter().map(|&x| x.powf(e)).collect(),
                    ArrayLike::IntScalar(e) =>
                        a.as_slice().iter().map(|&x| x.powf(e as f64)).collect(),
                    ArrayLike::Array(arr) => match &arr.inner {
                        ArrayData::Float(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter()).map(|(&x, &e)| x.powf(e)).collect(),
                        ArrayData::Int(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter()).map(|(&x, &e)| x.powf(e as f64)).collect(),
                    },
                    _ => {
                        let rhs = exp.into_ndarray()?;
                        a.as_slice().iter().zip(rhs.as_slice().iter()).map(|(&x, &e)| x.powf(e)).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)) })
            }
            ArrayData::Int(a) => {
                let data: Vec<f64> = match exp {
                    ArrayLike::Scalar(e) =>
                        a.as_slice().iter().map(|&x| (x as f64).powf(e)).collect(),
                    ArrayLike::IntScalar(e) =>
                        a.as_slice().iter().map(|&x| (x as f64).powf(e as f64)).collect(),
                    ArrayLike::Array(arr) => match &arr.inner {
                        ArrayData::Float(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter()).map(|(&x, &e)| (x as f64).powf(e)).collect(),
                        ArrayData::Int(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter()).map(|(&x, &e)| (x as f64).powf(e as f64)).collect(),
                    },
                    _ => {
                        let rhs = exp.into_ndarray()?;
                        a.as_slice().iter().zip(rhs.as_slice().iter()).map(|(&x, &e)| (x as f64).powf(e)).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)) })
            }
        }
    }

    fn __rpow__(&self, base: &Bound<'_, PyAny>, _modulo: Option<i64>) -> PyResult<Self> {
        let b = if let Ok(v) = base.extract::<f64>() { v }
                else { base.extract::<i64>()? as f64 };
        match &self.inner {
            ArrayData::Float(a) =>
                Ok(PyArray { inner: ArrayData::Float(a.map(|x| b.powf(x))) }),
            ArrayData::Int(a) => {
                let data: Vec<f64> = a.as_slice().iter().map(|&x| b.powf(x as f64)).collect();
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)) })
            }
        }
    }

    fn __matmul__(&self, other: &PyArray) -> PyResult<Self> {
        let a = self.as_float()?;
        let b = other.as_float()?;
        Ok(PyArray { inner: ArrayData::Float(a.matmul(b)) })
    }

    fn matmul(&self, other: &PyArray) -> PyResult<Self> {
        let a = self.as_float()?;
        let b = other.as_float()?;
        Ok(PyArray { inner: ArrayData::Float(a.matmul(b)) })
    }

    fn dot(&self, other: &PyArray) -> PyResult<Self> {
        let a = self.as_float()?;
        let b = other.as_float()?;
        Ok(PyArray { inner: ArrayData::Float(a.dot(b)) })
    }

    fn transpose(&self) -> Self {
        match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(a.transpose()) },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(a.transpose()) },
        }
    }

    fn t(&self) -> Self {
        match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(a.t()) },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(a.t()) },
        }
    }

    fn ravel(&self) -> Self {
        match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(a.ravel()) },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(a.ravel()) },
        }
    }

    fn take(&self, indices: Vec<usize>) -> Self {
        match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(a.take(&indices)) },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(a.take(&indices)) },
        }
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let total: usize = shape.iter().product();
        match &self.inner {
            ArrayData::Float(a) => {
                if a.len() != total {
                    return Err(PyValueError::new_err(format!(
                        "Cannot reshape array of size {} into shape {:?}",
                        a.len(), shape
                    )));
                }
                Ok(PyArray { inner: ArrayData::Float(a.reshape(shape)) })
            }
            ArrayData::Int(a) => {
                if a.len() != total {
                    return Err(PyValueError::new_err(format!(
                        "Cannot reshape array of size {} into shape {:?}",
                        a.len(), shape
                    )));
                }
                Ok(PyArray { inner: ArrayData::Int(a.reshape(shape)) })
            }
        }
    }

    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ArrayData::Float(a) => {
                if a.len() != 1 {
                    return Err(PyValueError::new_err(format!(
                        "item() can only be called on arrays with exactly one element, got {} elements",
                        a.len()
                    )));
                }
                Ok(a.item().into_pyobject(py)?.into_any().unbind())
            }
            ArrayData::Int(a) => {
                if a.len() != 1 {
                    return Err(PyValueError::new_err(format!(
                        "item() can only be called on arrays with exactly one element, got {} elements",
                        a.len()
                    )));
                }
                Ok(a.item().into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        use numpy::IntoPyArray;
        match &self.inner {
            ArrayData::Float(a) => {
                let shape = a.shape().dims();
                let data = a.as_slice().to_vec();
                Ok(data.into_pyarray(py).reshape(shape).unwrap().into_any().unbind())
            }
            ArrayData::Int(a) => {
                let shape = a.shape().dims();
                let data = a.as_slice().to_vec();
                Ok(data.into_pyarray(py).reshape(shape).unwrap().into_any().unbind())
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ArrayData::Float(a) => format!(
                "Array(dtype=float64, shape={:?}, data={:?})",
                a.shape().dims(), a.as_slice()
            ),
            ArrayData::Int(a) => format!(
                "Array(dtype=int64, shape={:?}, data={:?})",
                a.shape().dims(), a.as_slice()
            ),
        }
    }

    fn __len__(&self) -> usize {
        self.len_val()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dims = self.dims();
        let ndim = dims.len();

        if let Ok(bool_mask) = key.extract::<Vec<bool>>() {
            return self.apply_bool_mask_py(&bool_mask, py);
        }

        if let Ok(np_bool) = key.extract::<PyReadonlyArrayDyn<bool>>() {
            let mask: Vec<bool> = np_bool.as_slice()?.iter().copied().collect();
            return self.apply_bool_mask_py(&mask, py);
        }

        if let Ok(py_arr) = key.extract::<PyRef<PyArray>>() {
            return match &py_arr.inner {
                ArrayData::Int(idx_arr) => {
                    let indices: Vec<isize> = idx_arr.as_slice().iter().map(|&x| x as isize).collect();
                    self.apply_int_index_array_py(&indices, py)
                }
                ArrayData::Float(_) => Err(PyValueError::new_err(
                    "index arrays must be integer type"
                )),
            };
        }

        if let Ok(tuple) = key.cast::<PyTuple>() {
            let tuple_len = tuple.len();

            if tuple_len > ndim {
                return Err(PyValueError::new_err(format!(
                    "too many indices for array: array is {}-dimensional, but {} were indexed",
                    ndim, tuple_len
                )));
            }

            let mut indices: Vec<usize> = Vec::with_capacity(tuple_len);
            for i in 0..tuple_len {
                let item = tuple.get_item(i)?;
                let idx = item.extract::<isize>()?;
                let dim_size = dims[i] as isize;
                let normalized = if idx < 0 { dim_size + idx } else { idx };
                if normalized < 0 || normalized >= dim_size {
                    return Err(PyValueError::new_err(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, i, dims[i]
                    )));
                }
                indices.push(normalized as usize);
            }

            if tuple_len == ndim {
                return self.scalar_at_indices(&indices, py);
            }

            let result_dims: Vec<usize> = dims[tuple_len..].to_vec();
            let result_size: usize = result_dims.iter().product();

            let strides = self.strides_val();
            let mut start_offset = 0;
            for (i, &idx) in indices.iter().enumerate() {
                start_offset += idx * strides[i];
            }

            return Ok(self.subarray_at(result_dims, start_offset, result_size)
                .into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(idx) = key.extract::<isize>() {
            let dim0 = dims[0] as isize;
            let normalized = if idx < 0 { dim0 + idx } else { idx };
            if normalized < 0 || normalized >= dim0 {
                return Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, dims[0]
                )));
            }

            if ndim == 1 {
                return self.scalar_at_flat(normalized as usize, py);
            }

            let result_dims: Vec<usize> = dims[1..].to_vec();
            let result_size: usize = result_dims.iter().product();
            let start_offset = normalized as usize * self.strides_val()[0];

            return Ok(self.subarray_at(result_dims, start_offset, result_size)
                .into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(slice) = key.cast::<PySlice>() {
            let axis0_len = dims[0] as isize;
            let indices = slice.indices(axis0_len)?;
            let stride0 = self.strides_val()[0];

            if ndim == 1 {
                let mut result_flat = Vec::new();
                let mut i = indices.start;
                while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                    if i >= 0 && i < axis0_len {
                        result_flat.push(i as usize);
                    }
                    i += indices.step;
                }
                let arr = self.gather_flat_indices(result_flat);
                return Ok(arr.into_pyobject(py)?.into_any().unbind());
            }

            let row_size: usize = dims[1..].iter().product();
            let mut row_starts = Vec::new();
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                if i >= 0 && i < axis0_len {
                    row_starts.push(i as usize * stride0);
                }
                i += indices.step;
            }

            let num_rows = row_starts.len();
            let mut result_dims = vec![num_rows];
            result_dims.extend_from_slice(&dims[1..]);

            let arr = self.gather_rows(row_starts, row_size, result_dims);
            return Ok(arr.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err("indices must be integers, slices, or tuples of integers"))
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let dims = self.dims();
        let ndim = dims.len();

        if let Ok(tuple) = key.cast::<PyTuple>() {
            let tuple_len = tuple.len();

            if tuple_len != ndim {
                return Err(PyValueError::new_err(format!(
                    "cannot set item: expected {} indices, got {}",
                    ndim, tuple_len
                )));
            }

            let mut indices: Vec<usize> = Vec::with_capacity(tuple_len);
            for i in 0..tuple_len {
                let item = tuple.get_item(i)?;
                let idx = item.extract::<isize>()?;
                let dim_size = dims[i] as isize;
                let normalized = if idx < 0 { dim_size + idx } else { idx };
                if normalized < 0 || normalized >= dim_size {
                    return Err(PyValueError::new_err(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, i, dims[i]
                    )));
                }
                indices.push(normalized as usize);
            }

            match &mut self.inner {
                ArrayData::Float(a) => {
                    let v: f64 = value.extract()?;
                    *a.get_mut(&indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))? = v;
                }
                ArrayData::Int(a) => {
                    let v: i64 = value.extract()?;
                    *a.get_mut(&indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))? = v;
                }
            }
            return Ok(());
        }

        if let Ok(idx) = key.extract::<isize>() {
            if ndim != 1 {
                return Err(PyValueError::new_err(
                    "single index assignment only supported for 1D arrays; use tuple indexing for nD arrays"
                ));
            }

            let dim0 = dims[0] as isize;
            let normalized = if idx < 0 { dim0 + idx } else { idx };
            if normalized < 0 || normalized >= dim0 {
                return Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, dims[0]
                )));
            }

            match &mut self.inner {
                ArrayData::Float(a) => {
                    let v: f64 = value.extract()?;
                    a.as_mut_slice()[normalized as usize] = v;
                }
                ArrayData::Int(a) => {
                    let v: i64 = value.extract()?;
                    a.as_mut_slice()[normalized as usize] = v;
                }
            }
            return Ok(());
        }

        Err(PyValueError::new_err("indices must be integers or tuples of integers for assignment"))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        match &slf.inner {
            ArrayData::Float(a) => {
                Py::new(py, PyArrayIter { data: a.as_slice().to_vec(), index: 0 })
                    .map(|o| o.into_any())
            }
            ArrayData::Int(a) => {
                Py::new(py, PyIntArrayIter { data: a.as_slice().to_vec(), index: 0 })
                    .map(|o| o.into_any())
            }
        }
    }

    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        match &self.inner {
            ArrayData::Float(a) => {
                let v: f64 = value.extract()?;
                Ok(a.as_slice().contains(&v))
            }
            ArrayData::Int(a) => {
                let v: i64 = value.extract()?;
                Ok(a.as_slice().contains(&v))
            }
        }
    }
}

impl PyArray {
    fn apply_cmp(&self, other: ArrayLike, cmp: impl Fn(f64, f64) -> bool) -> PyResult<Self> {
        match &self.inner {
            ArrayData::Float(a) => {
                let data: Vec<f64> = match other {
                    ArrayLike::Scalar(s) =>
                        a.as_slice().iter().map(|&x| if cmp(x, s) { 1.0 } else { 0.0 }).collect(),
                    ArrayLike::IntScalar(s) => {
                        let s = s as f64;
                        a.as_slice().iter().map(|&x| if cmp(x, s) { 1.0 } else { 0.0 }).collect()
                    }
                    ArrayLike::Array(arr) => match &arr.inner {
                        ArrayData::Float(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter())
                                .map(|(&x, &y)| if cmp(x, y) { 1.0 } else { 0.0 }).collect(),
                        ArrayData::Int(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter())
                                .map(|(&x, &y)| if cmp(x, y as f64) { 1.0 } else { 0.0 }).collect(),
                    },
                    _ => {
                        let rhs = other.into_ndarray()?;
                        a.as_slice().iter().zip(rhs.as_slice().iter())
                            .map(|(&x, &y)| if cmp(x, y) { 1.0 } else { 0.0 }).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)) })
            }
            ArrayData::Int(a) => {
                let data: Vec<i64> = match other {
                    ArrayLike::Scalar(s) =>
                        a.as_slice().iter().map(|&x| if cmp(x as f64, s) { 1 } else { 0 }).collect(),
                    ArrayLike::IntScalar(s) => {
                        let s = s as f64;
                        a.as_slice().iter().map(|&x| if cmp(x as f64, s) { 1 } else { 0 }).collect()
                    }
                    ArrayLike::Array(arr) => match &arr.inner {
                        ArrayData::Float(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter())
                                .map(|(&x, &y)| if cmp(x as f64, y) { 1 } else { 0 }).collect(),
                        ArrayData::Int(b) =>
                            a.as_slice().iter().zip(b.as_slice().iter())
                                .map(|(&x, &y)| if cmp(x as f64, y as f64) { 1 } else { 0 }).collect(),
                    },
                    _ => {
                        let rhs = other.into_i64_ndarray()?;
                        a.as_slice().iter().zip(rhs.as_slice().iter())
                            .map(|(&x, &y)| if cmp(x as f64, y as f64) { 1 } else { 0 }).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Int(NdArray::from_vec(a.shape().clone(), data)) })
            }
        }
    }

    fn apply_bool_mask_py(&self, mask: &[bool], py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dims = self.dims();
        if dims.is_empty() {
            return Err(PyValueError::new_err("Cannot apply boolean mask to scalar"));
        }
        if mask.len() != dims[0] {
            return Err(PyValueError::new_err(format!(
                "Boolean mask length {} doesn't match first dimension {}",
                mask.len(), dims[0]
            )));
        }
        let result = match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(a.boolean_mask(mask)) },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(a.boolean_mask(mask)) },
        };
        Ok(result.into_pyobject(py)?.into_any().unbind())
    }

    fn dims(&self) -> Vec<usize> {
        match &self.inner {
            ArrayData::Float(a) => a.shape().dims().to_vec(),
            ArrayData::Int(a) => a.shape().dims().to_vec(),
        }
    }

    fn ndim_val(&self) -> usize {
        match &self.inner {
            ArrayData::Float(a) => a.ndim(),
            ArrayData::Int(a) => a.ndim(),
        }
    }

    fn len_val(&self) -> usize {
        match &self.inner {
            ArrayData::Float(a) => a.as_slice().len(),
            ArrayData::Int(a) => a.as_slice().len(),
        }
    }

    fn strides_val(&self) -> Vec<usize> {
        match &self.inner {
            ArrayData::Float(a) => a.strides().to_vec(),
            ArrayData::Int(a) => a.strides().to_vec(),
        }
    }

    fn scalar_at_flat(&self, flat_idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ArrayData::Float(a) => Ok(a.as_slice()[flat_idx].into_pyobject(py)?.into_any().unbind()),
            ArrayData::Int(a) => Ok(a.as_slice()[flat_idx].into_pyobject(py)?.into_any().unbind()),
        }
    }

    fn scalar_at_indices(&self, indices: &[usize], py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ArrayData::Float(a) => {
                let v = a.get(indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))?;
                Ok((*v).into_pyobject(py)?.into_any().unbind())
            }
            ArrayData::Int(a) => {
                let v = a.get(indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))?;
                Ok((*v).into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    fn subarray_at(&self, dims: Vec<usize>, start: usize, size: usize) -> PyArray {
        match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(
                NdArray::from_vec(Shape::new(dims), a.as_slice()[start..start + size].to_vec())
            )},
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(
                NdArray::from_vec(Shape::new(dims), a.as_slice()[start..start + size].to_vec())
            )},
        }
    }

    fn gather_flat_indices(&self, flat_indices: Vec<usize>) -> PyArray {
        match &self.inner {
            ArrayData::Float(a) => {
                let data: Vec<f64> = flat_indices.iter().map(|&i| a.as_slice()[i]).collect();
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(data.len()), data)) }
            }
            ArrayData::Int(a) => {
                let data: Vec<i64> = flat_indices.iter().map(|&i| a.as_slice()[i]).collect();
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(data.len()), data)) }
            }
        }
    }

    fn gather_rows(&self, row_starts: Vec<usize>, row_size: usize, result_dims: Vec<usize>) -> PyArray {
        match &self.inner {
            ArrayData::Float(a) => {
                let mut result = Vec::new();
                for start in row_starts {
                    result.extend_from_slice(&a.as_slice()[start..start + row_size]);
                }
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(result_dims), result)) }
            }
            ArrayData::Int(a) => {
                let mut result = Vec::new();
                for start in row_starts {
                    result.extend_from_slice(&a.as_slice()[start..start + row_size]);
                }
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(result_dims), result)) }
            }
        }
    }

    fn float_to_pylist_recursive(&self, a: &NdArray<f64>, py: Python<'_>, dim: usize, offset: usize) -> PyResult<Py<PyAny>> {
        let data = a.as_slice();
        let dim_size = a.shape().dim(dim).unwrap();
        let strides = a.strides();

        if dim == a.ndim() - 1 {
            let list = PyList::new(py, (0..dim_size).map(|i| data[offset + i * strides[dim]]))?;
            return Ok(list.into());
        }

        let items: Vec<Py<PyAny>> = (0..dim_size)
            .map(|i| self.float_to_pylist_recursive(a, py, dim + 1, offset + i * strides[dim]))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into())
    }

    fn int_to_pylist_recursive(&self, a: &NdArray<i64>, py: Python<'_>, dim: usize, offset: usize) -> PyResult<Py<PyAny>> {
        let data = a.as_slice();
        let dim_size = a.shape().dim(dim).unwrap();
        let strides = a.strides();

        if dim == a.ndim() - 1 {
            let list = PyList::new(py, (0..dim_size).map(|i| data[offset + i * strides[dim]]))?;
            return Ok(list.into());
        }

        let items: Vec<Py<PyAny>> = (0..dim_size)
            .map(|i| self.int_to_pylist_recursive(a, py, dim + 1, offset + i * strides[dim]))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into())
    }

    fn apply_int_index_array_py(&self, indices: &[isize], py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dims = self.dims();
        let dim0 = dims[0] as isize;

        let normalized: Vec<usize> = indices.iter().map(|&idx| {
            let n = if idx < 0 { dim0 + idx } else { idx };
            if n < 0 || n >= dim0 {
                Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}", idx, dim0
                )))
            } else {
                Ok(n as usize)
            }
        }).collect::<PyResult<_>>()?;

        if dims.len() == 1 {
            let arr = self.gather_flat_indices(normalized);
            return Ok(arr.into_pyobject(py)?.into_any().unbind());
        }

        let stride0 = self.strides_val()[0];
        let row_starts: Vec<usize> = normalized.iter().map(|&i| i * stride0).collect();
        let row_size: usize = dims[1..].iter().product();
        let mut result_dims = vec![normalized.len()];
        result_dims.extend_from_slice(&dims[1..]);

        let arr = self.gather_rows(row_starts, row_size, result_dims);
        Ok(arr.into_pyobject(py)?.into_any().unbind())
    }
}

#[pyclass]
pub struct PyArrayIter {
    data: Vec<f64>,
    index: usize,
}

#[pymethods]
impl PyArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<f64> {
        if slf.index < slf.data.len() {
            let val = slf.data[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

#[pyclass]
pub struct PyIntArrayIter {
    data: Vec<i64>,
    index: usize,
}

#[pymethods]
impl PyIntArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<i64> {
        if slf.index < slf.data.len() {
            let val = slf.data[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
}