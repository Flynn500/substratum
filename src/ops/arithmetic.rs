use std::ops::{Add, Sub, Mul, Div};

use crate::array::broadcast::BroadcastIter;
use crate::array::ndarray::NdArray;
use crate::array::storage::Storage;

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        //owned + owned
        impl<T> $trait for NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: NdArray<T>) -> Self::Output {
                binary_op_inplace(self, &rhs, |a, b| a $op b)
            }
        }

        //owned + ref
        impl<T> $trait<&NdArray<T>> for NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: &NdArray<T>) -> Self::Output {
                binary_op_inplace(self, rhs, |a, b| a $op b)
            }
        }

        //ref + owned
        impl<T> $trait<NdArray<T>> for &NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: NdArray<T>) -> Self::Output {
                binary_op_inplace_rev(rhs, self, |a, b| a $op b)
            }
        }

        //ref + ref
        impl<T> $trait<&NdArray<T>> for &NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: &NdArray<T>) -> Self::Output {
                binary_op_copy(self, rhs, |a, b| a $op b)
            }
        }
    };
}

fn binary_op_copy<T, F>(a: &NdArray<T>, b: &NdArray<T>, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let iter = BroadcastIter::new(a.shape(), b.shape())
        .expect("Shapes are not broadcast-compatible");

    let output_shape = iter.output_shape().clone();
    let a_data = a.as_slice();
    let b_data = b.as_slice();

    let result_data: Vec<T> = iter
        .map(|(idx_a, idx_b)| op(a_data[idx_a], b_data[idx_b]))
        .collect();

    NdArray::new(output_shape, Storage::from_vec(result_data))
}

fn binary_op_inplace<T, F>(mut a: NdArray<T>, b: &NdArray<T>, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if a.shape() == b.shape() {
        let a_data = a.as_mut_slice();
        let b_data = b.as_slice();
        for i in 0..a_data.len() {
            a_data[i] = op(a_data[i], b_data[i]);
        }
        a
    } else {
        binary_op_copy(&a, b, op)
    }
}

fn binary_op_inplace_rev<T, F>(mut a: NdArray<T>, b: &NdArray<T>, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if a.shape() == b.shape() {
        let a_data = a.as_mut_slice();
        let b_data = b.as_slice();
        for i in 0..a_data.len() {
            a_data[i] = op(b_data[i], a_data[i]);
        }
        a
    } else {
        binary_op_copy(b, &a, op)
    }
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Sub, sub, -);
impl_binary_op!(Mul, mul, *);
impl_binary_op!(Div, div, /);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::Shape;

    #[test]
    fn add_same_shape() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1, 2, 3]);
        let b = NdArray::from_vec(Shape::d1(3), vec![10, 20, 30]);

        let c = &a + &b;
        assert_eq!(c.as_slice(), &[11, 22, 33]);
    }

    #[test]
    fn add_broadcast_scalar() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1, 2, 3]);
        let b = NdArray::from_vec(Shape::scalar(), vec![10]);

        let c = &a + &b;
        assert_eq!(c.as_slice(), &[11, 12, 13]);
    }

    #[test]
    fn add_broadcast_row() {
        // [2, 3] + [3] -> [2, 3]
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        let b = NdArray::from_vec(Shape::d1(3), vec![10, 20, 30]);

        let c = &a + &b;
        assert_eq!(c.as_slice(), &[11, 22, 33, 14, 25, 36]);
    }

    #[test]
    fn sub_same_shape() {
        let a = NdArray::from_vec(Shape::d1(3), vec![10, 20, 30]);
        let b = NdArray::from_vec(Shape::d1(3), vec![1, 2, 3]);

        let c = &a - &b;
        assert_eq!(c.as_slice(), &[9, 18, 27]);
    }

    #[test]
    fn mul_broadcast() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1, 2, 3, 4]);
        let b = NdArray::from_vec(Shape::scalar(), vec![10]);

        let c = &a * &b;
        assert_eq!(c.as_slice(), &[10, 20, 30, 40]);
    }

    #[test]
    fn div_same_shape() {
        let a = NdArray::from_vec(Shape::d1(3), vec![10.0, 20.0, 30.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![2.0, 4.0, 5.0]);

        let c = &a / &b;
        assert_eq!(c.as_slice(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn owned_add() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1, 2]);
        let b = NdArray::from_vec(Shape::d1(2), vec![3, 4]);

        let c = a + b;
        assert_eq!(c.as_slice(), &[4, 6]);
    }
}