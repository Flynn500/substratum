use std::ops::{Add, Sub, Mul, Div};

use crate::array::ndarray::NdArray;
use crate::array::storage::Storage;

macro_rules! impl_scalar_ops {
    ($scalar:ty) => {
        // array + scalar
        impl Add<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: $scalar) -> Self::Output {
                scalar_op(&self, rhs, |a, b| a + b)
            }
        }

        impl Add<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: $scalar) -> Self::Output {
                scalar_op(self, rhs, |a, b| a + b)
            }
        }

        // scalar + array
        impl Add<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op(&rhs, self, |a, b| a + b)
            }
        }

        impl Add<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op(rhs, self, |a, b| a + b)
            }
        }

        // array - scalar
        impl Sub<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: $scalar) -> Self::Output {
                scalar_op(&self, rhs, |a, b| a - b)
            }
        }

        impl Sub<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: $scalar) -> Self::Output {
                scalar_op(self, rhs, |a, b| a - b)
            }
        }

        // scalar - array
        impl Sub<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op_rev(&rhs, self, |a, b| b - a)
            }
        }

        impl Sub<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op_rev(rhs, self, |a, b| b - a)
            }
        }

        // array * scalar
        impl Mul<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: $scalar) -> Self::Output {
                scalar_op(&self, rhs, |a, b| a * b)
            }
        }

        impl Mul<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: $scalar) -> Self::Output {
                scalar_op(self, rhs, |a, b| a * b)
            }
        }

        // scalar * array
        impl Mul<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op(&rhs, self, |a, b| a * b)
            }
        }

        impl Mul<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op(rhs, self, |a, b| a * b)
            }
        }

        // array / scalar
        impl Div<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: $scalar) -> Self::Output {
                scalar_op(&self, rhs, |a, b| a / b)
            }
        }

        impl Div<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: $scalar) -> Self::Output {
                scalar_op(self, rhs, |a, b| a / b)
            }
        }

        // scalar / array
        impl Div<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op_rev(&rhs, self, |a, b| b / a)
            }
        }

        impl Div<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op_rev(rhs, self, |a, b| b / a)
            }
        }
    };
}

fn scalar_op<T, F>(arr: &NdArray<T>, scalar: T, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let result: Vec<T> = arr.as_slice().iter().map(|&a| op(a, scalar)).collect();
    NdArray::new(arr.shape().clone(), Storage::from_vec(result))
}

fn scalar_op_rev<T, F>(arr: &NdArray<T>, scalar: T, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let result: Vec<T> = arr.as_slice().iter().map(|&a| op(a, scalar)).collect();
    NdArray::new(arr.shape().clone(), Storage::from_vec(result))
}

impl_scalar_ops!(f32);
impl_scalar_ops!(f64);
impl_scalar_ops!(i32);
impl_scalar_ops!(i64);
impl_scalar_ops!(u32);
impl_scalar_ops!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::Shape;

    #[test]
    fn array_add_scalar() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0_f64, 2.0, 3.0]);
        let b = &a + 10.0;
        assert_eq!(b.as_slice(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn scalar_add_array() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0_f64, 2.0, 3.0]);
        let b = 10.0 + &a;
        assert_eq!(b.as_slice(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn array_sub_scalar() {
        let a = NdArray::from_vec(Shape::d1(3), vec![10_i32, 20, 30]);
        let b = &a - 5;
        assert_eq!(b.as_slice(), &[5, 15, 25]);
    }

    #[test]
    fn scalar_sub_array() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1_i32, 2, 3]);
        let b = 10 - &a;
        assert_eq!(b.as_slice(), &[9, 8, 7]);
    }

    #[test]
    fn array_mul_scalar() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0_f64, 2.0, 3.0]);
        let b = &a * 2.0;
        assert_eq!(b.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn scalar_mul_array() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1_i32, 2, 3]);
        let b = 10 * &a;
        assert_eq!(b.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn array_div_scalar() {
        let a = NdArray::from_vec(Shape::d1(3), vec![10.0_f64, 20.0, 30.0]);
        let b = &a / 2.0;
        assert_eq!(b.as_slice(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn scalar_div_array() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0_f64, 2.0, 4.0]);
        let b = 8.0 / &a;
        assert_eq!(b.as_slice(), &[8.0, 4.0, 2.0]);
    }
}