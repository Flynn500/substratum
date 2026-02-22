use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayData, ArrayLike};

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
#[pyo3(signature = (shape, dtype=None))]
pub fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::zeros(shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::ones(shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None))]
pub fn full(shape: Vec<usize>, fill_value: f64, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::full(shape, fill_value, dtype)
}

#[pyfunction]
#[pyo3(signature = (data, shape=None, dtype=None))]
pub fn asarray(data: ArrayLike, shape: Option<Vec<usize>>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::asarray(data, shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=None, dtype=None))]
pub fn eye(n: usize, m: Option<usize>, k: Option<isize>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::eye(n, m, k, dtype)
}

#[pyfunction]
#[pyo3(signature = (v, k=None))]
pub fn diag(v: ArrayLike, k: Option<isize>) -> PyResult<PyArray> {
    PyArray::diag(v, k)
}

#[pyfunction]
pub fn column_stack(arrays: Vec<PyArray>) -> PyResult<PyArray> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let float_arrays: Vec<NdArray<f64>> = arrays.iter()
        .map(|a| a.as_float().map(|x| x.clone()))
        .collect::<PyResult<_>>()?;
    let array_refs: Vec<&NdArray<f64>> = float_arrays.iter().collect();
    Ok(PyArray {
        inner: ArrayData::Float(NdArray::column_stack(&array_refs)),
    })
}

#[pyfunction]
fn outer(a: ArrayLike, b: ArrayLike) -> PyResult<PyArray> {
    let a_arr = a.into_ndarray()?;
    let b_arr = b.into_ndarray()?;
    Ok(PyArray {
        inner: ArrayData::Float(NdArray::outer(&a_arr, &b_arr)),
    })
}

#[pyfunction]
fn from_numpy(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let numpy_arr: PyReadonlyArrayDyn<f64> = arr.extract()?;
    let shape: Vec<usize> = numpy_arr.shape().to_vec();
    let data: Vec<f64> = numpy_arr.as_slice()?.to_vec();
    Ok(PyArray {
        inner: ArrayData::Float(NdArray::from_vec(Shape::new(shape), data)),
    })
}

#[pyfunction]
fn to_numpy(py: Python<'_>, arr: &PyArray) -> PyResult<Py<PyAny>> {
    arr.to_numpy(py)
}
