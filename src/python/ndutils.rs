use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::exceptions::PyValueError;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods, PyArrayMethods, IntoPyArray};
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayLike};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(column_stack, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyArray {
    PyArray::zeros(shape)
}

#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyArray {
    PyArray::ones(shape)
}

#[pyfunction]
pub fn full(shape: Vec<usize>, fill_value: f64) -> PyArray {
    PyArray::full(shape, fill_value)
}

#[pyfunction]
#[pyo3(signature = (data, shape=None))]
pub fn asarray(data: ArrayLike, shape: Option<Vec<usize>>) -> PyResult<PyArray> {
    PyArray::asarray(data, shape)
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=None))]
pub fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> PyArray {
    PyArray::eye(n, m, k)
}

#[pyfunction]
#[pyo3(signature = (v, k=None))]
pub fn diag(v: ArrayLike, k: Option<isize>) -> PyArray {
    PyArray::diag(v, k)
}

#[pyfunction]
pub fn column_stack(arrays: Vec<PyArray>) -> PyResult<PyArray> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let array_refs: Vec<&NdArray<f64>> = arrays.iter().map(|a| &a.inner).collect();
    Ok(PyArray {
        inner: NdArray::column_stack(&array_refs),
    })
}

#[pyfunction]
fn outer(a: ArrayLike, b: ArrayLike) -> PyArray {
    let a_arr = a.into_ndarray().unwrap();
    let b_arr = b.into_ndarray().unwrap();
    PyArray {
        inner: NdArray::outer(&a_arr, &b_arr),
    }
}

#[pyfunction]
fn from_numpy(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let numpy_arr: PyReadonlyArrayDyn<f64> = arr.extract()?;
    let shape: Vec<usize> = numpy_arr.shape().to_vec();
    let data: Vec<f64> = numpy_arr.as_slice()?.to_vec();
    Ok(PyArray {
        inner: NdArray::from_vec(Shape::new(shape), data),
    })
}

#[pyfunction]
fn to_numpy<'py>(py: Python<'py>, arr: &PyArray) -> Bound<'py, PyArrayDyn<f64>> {
    let shape = arr.inner.shape().dims();
    let data = arr.inner.as_slice().to_vec();
    data.into_pyarray(py).reshape(shape).unwrap()
}
