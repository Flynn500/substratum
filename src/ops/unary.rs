use std::ops::Neg;

use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;
use crate::array::storage::Storage;

impl<T> Neg for NdArray<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = NdArray<T>;

    fn neg(self) -> Self::Output {
        neg_array(&self)
    }
}

impl<T> Neg for &NdArray<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = NdArray<T>;

    fn neg(self) -> Self::Output {
        neg_array(self)
    }
}

fn neg_array<T>(a: &NdArray<T>) -> NdArray<T>
where
    T: Neg<Output = T> + Copy,
{
    let result_data: Vec<T> = a.as_slice().iter().map(|&x| -x).collect();
    NdArray::new(a.shape().clone(), Storage::from_vec(result_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_integers() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1, -2, 3]);
        let b = -&a;
        assert_eq!(b.as_slice(), &[-1, 2, -3]);
    }

    #[test]
    fn neg_floats() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.5, -2.5]);
        let b = -a;
        assert_eq!(b.as_slice(), &[-1.5, 2.5]);
    }

    #[test]
    fn neg_preserves_shape() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        let b = -&a;
        assert_eq!(b.shape().dims(), &[2, 3]);
    }
}