pub mod array;
pub mod ops;
pub mod random;

pub use array::{NdArray, Shape, Storage, BroadcastIter};
pub use random::Generator;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass(name = "Array")]
pub struct PyArray {
    inner: NdArray<f64>,
}

#[pymethods]
impl PyArray {
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyArray {
            inner: NdArray::zeros(Shape::new(shape)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (n, m=None, k=None))]
    fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> Self {
        PyArray {
            inner: NdArray::eye(n, m, k.unwrap_or(0)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (v, k=None))]
    fn diag(v: Vec<f64>, k: Option<isize>) -> Self {
        let v_arr = NdArray::from_vec(Shape::d1(v.len()), v);
        PyArray {
            inner: NdArray::from_diag(&v_arr, k.unwrap_or(0)),
        }
    }

    #[staticmethod]
    fn outer(a: Vec<f64>, b: Vec<f64>) -> Self {
        let a_arr = NdArray::from_vec(Shape::d1(a.len()), a);
        let b_arr = NdArray::from_vec(Shape::d1(b.len()), b);
        PyArray {
            inner: NdArray::outer(&a_arr, &b_arr),
        }
    }

    #[pyo3(signature = (k=None))]
    fn diagonal(&self, k: Option<isize>) -> PyResult<Vec<f64>> {
        if self.inner.ndim() != 2 {
            return Err(PyValueError::new_err("diagonal requires a 2D array"));
        }
        let diag = self.inner.diagonal(k.unwrap_or(0));
        Ok(diag.as_slice().to_vec())
    }

    #[new]
    fn new(shape: Vec<usize>, data: Vec<f64>) -> PyResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(PyValueError::new_err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_size
            )));
        }
        Ok(PyArray {
            inner: NdArray::from_vec(Shape::new(shape), data),
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    fn get(&self, indices: Vec<usize>) -> PyResult<f64> {
        self.inner.get(&indices)
            .copied()
            .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
    }

    fn tolist(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
    }

    fn sin(&self) -> Self {
        PyArray { inner: self.inner.sin() }
    }

    fn cos(&self) -> Self {
        PyArray { inner: self.inner.cos() }
    }

    fn exp(&self) -> Self {
        PyArray { inner: self.inner.exp() }
    }

    fn sqrt(&self) -> Self {
        PyArray { inner: self.inner.sqrt() }
    }

    fn clip(&self, min: f64, max: f64) -> Self {
        PyArray { inner: self.inner.clip(min, max) }
    }

    fn __add__(&self, other: &PyArray) -> Self {
        PyArray { inner: &self.inner + &other.inner }
    }

    fn __sub__(&self, other: &PyArray) -> Self {
        PyArray { inner: &self.inner - &other.inner }
    }

    fn __mul__(&self, other: &PyArray) -> Self {
        PyArray { inner: &self.inner * &other.inner }
    }

    fn __truediv__(&self, other: &PyArray) -> Self {
        PyArray { inner: &self.inner / &other.inner }
    }

    fn __neg__(&self) -> Self {
        PyArray { inner: -&self.inner }
    }

    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, data={:?})",
            self.inner.shape().dims(),
            self.inner.as_slice())
    }
}

#[pyclass(name = "Generator")]
pub struct PyGenerator {
    inner: Generator,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new() -> Self {
        PyGenerator { inner: Generator::new() }
    }

    #[staticmethod]
    fn from_seed(seed: u64) -> Self {
        PyGenerator { inner: Generator::from_seed(seed) }
    }

    fn uniform(&mut self, low: f64, high: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.uniform(low, high, Shape::new(shape)),
        }
    }

    fn standard_normal(&mut self, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.standard_normal(Shape::new(shape)),
        }
    }

    fn normal(&mut self, mu: f64, sigma: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.normal(mu, sigma, Shape::new(shape)),
        }
    }

    fn randint(&mut self, low: i64, high: i64, shape: Vec<usize>) -> Vec<i64> {
        let arr = self.inner.randint(low, high, Shape::new(shape));
        arr.as_slice().to_vec()
    }
}

#[pymodule]
mod substratum {
    use super::*;

    #[pymodule_export]
    use super::PyArray;

    #[pymodule_export]
    use super::PyGenerator;

    #[pyfunction]
    fn zeros(shape: Vec<usize>) -> PyArray {
        PyArray::zeros(shape)
    }

    #[pyfunction]
    #[pyo3(signature = (n, m=None, k=None))]
    fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> PyArray {
        PyArray::eye(n, m, k)
    }

    #[pyfunction]
    #[pyo3(signature = (v, k=None))]
    fn diag(v: Vec<f64>, k: Option<isize>) -> PyArray {
        PyArray::diag(v, k)
    }

    #[pyfunction]
    fn outer(a: Vec<f64>, b: Vec<f64>) -> PyArray {
        PyArray::outer(a, b)
    }
}
