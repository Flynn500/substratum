use crate::array::shape::Shape;
use crate::array::storage::Storage;

#[derive(Debug, Clone)]
pub struct NdArray<T> {
    shape: Shape,
    strides: Vec<usize>,
    storage: Storage<T>,
}

impl<T> NdArray<T> {
    pub fn new(shape: Shape, storage: Storage<T>) -> Self {
        assert_eq!(
            shape.size(),
            storage.len(),
            "Storage length {} doesn't match shape size {}",
            storage.len(),
            shape.size()
        );

        let strides = shape.strides_row_major();
        
        NdArray {
            shape,
            strides,
            storage,
        }
    }

    pub fn from_vec(shape: Shape, data: Vec<T>) -> Self {
        Self::new(shape, Storage::from_vec(data))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn len(&self) -> usize {
        self.shape.size()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.size() == 0
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }

        let mut offset = 0;
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.dims()).enumerate() {
            if idx >= dim {
                return None;
            }
            offset += idx * self.strides[i];
        }
        
        Some(offset)
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        self.flat_index(indices)
            .and_then(|i| self.storage.get(i))
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.flat_index(indices)
            .and_then(|i| self.storage.get_mut(i))
    }
}

impl<T: Clone> NdArray<T> {
    pub fn filled(shape: Shape, value: T) -> Self {
        let storage = Storage::filled(value, shape.size());
        Self::new(shape, storage)
    }
}

impl<T: Default + Clone> NdArray<T> {
    pub fn zeros(shape: Shape) -> Self {
        let storage = Storage::zeros(shape.size());
        Self::new(shape, storage)
    }
}

impl<T: Copy> NdArray<T> {
    pub fn map<F, U>(&self, f: F) -> NdArray<U>
    where
        F: Fn(T) -> U,
        U: Copy,
    {
        let result_data: Vec<U> = self.as_slice().iter().map(|&x| f(x)).collect();
        NdArray::new(self.shape().clone(), Storage::from_vec(result_data))
    }
}

impl NdArray<f64> {
    pub fn eye(n: usize, m: Option<usize>, k: isize) -> Self {
        let m = m.unwrap_or(n);
        let mut arr = NdArray::zeros(Shape::d2(n, m));

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        let mut row = row_start;
        let mut col = col_start;
        while row < n && col < m {
            *arr.get_mut(&[row, col]).unwrap() = 1.0;
            row += 1;
            col += 1;
        }
        arr
    }

    pub fn from_diag(v: &NdArray<f64>, k: isize) -> Self {
        assert_eq!(v.ndim(), 1, "Input must be 1D");
        let n = v.len();
        let size = n + k.unsigned_abs();
        let mut arr = NdArray::zeros(Shape::d2(size, size));

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        for (i, &val) in v.as_slice().iter().enumerate() {
            *arr.get_mut(&[row_start + i, col_start + i]).unwrap() = val;
        }
        arr
    }

    pub fn diagonal(&self, k: isize) -> Self {
        assert_eq!(self.ndim(), 2, "Input must be 2D");
        let (n, m) = (self.shape().dims()[0], self.shape().dims()[1]);

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        if row_start >= n || col_start >= m {
            return NdArray::from_vec(Shape::d1(0), vec![]);
        }

        let diag_len = (n - row_start).min(m - col_start);
        let mut data = Vec::with_capacity(diag_len);

        for i in 0..diag_len {
            data.push(*self.get(&[row_start + i, col_start + i]).unwrap());
        }

        NdArray::from_vec(Shape::d1(diag_len), data)
    }

    pub fn outer(a: &NdArray<f64>, b: &NdArray<f64>) -> Self {
        assert_eq!(a.ndim(), 1, "First input must be 1D");
        assert_eq!(b.ndim(), 1, "Second input must be 1D");

        let m = a.len();
        let n = b.len();
        let mut data = Vec::with_capacity(m * n);

        for &ai in a.as_slice() {
            for &bi in b.as_slice() {
                data.push(ai * bi);
            }
        }

        NdArray::from_vec(Shape::d2(m, n), data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_basic() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 6);
        assert_eq!(arr.shape().dims(), &[2, 3]);
    }

    #[test]
    fn indexing_2d() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        
        assert_eq!(arr.get(&[0, 0]), Some(&1));
        assert_eq!(arr.get(&[0, 2]), Some(&3));
        assert_eq!(arr.get(&[1, 0]), Some(&4));
        assert_eq!(arr.get(&[1, 2]), Some(&6));
    }

    #[test]
    fn indexing_out_of_bounds() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        
        assert_eq!(arr.get(&[2, 0]), None);
        assert_eq!(arr.get(&[0, 3]), None);
        assert_eq!(arr.get(&[0]), None);
    }

    #[test]
    fn zeros_creates_default_values() {
        let arr: NdArray<f64> = NdArray::zeros(Shape::d2(2, 2));
        assert_eq!(arr.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn filled_creates_repeated_value() {
        let arr = NdArray::filled(Shape::d1(4), 7);
        assert_eq!(arr.as_slice(), &[7, 7, 7, 7]);
    }

    #[test]
    fn get_mut_modifies_element() {
        let mut arr = NdArray::from_vec(Shape::d2(2, 2), vec![1, 2, 3, 4]);
        *arr.get_mut(&[1, 1]).unwrap() = 99;
        assert_eq!(arr.get(&[1, 1]), Some(&99));
    }

    #[test]
    #[should_panic(expected = "Storage length")]
    fn mismatched_shape_panics() {
        NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3]);
    }

    #[test]
    fn eye_square() {
        let arr = NdArray::eye(3, None, 0);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        assert_eq!(arr.as_slice(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn eye_rectangular() {
        let arr = NdArray::eye(2, Some(4), 0);
        assert_eq!(arr.shape().dims(), &[2, 4]);
        assert_eq!(arr.as_slice(), &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn eye_positive_offset() {
        let arr = NdArray::eye(3, Some(4), 1);
        assert_eq!(arr.shape().dims(), &[3, 4]);
        // [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        assert_eq!(arr.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn eye_negative_offset() {
        let arr = NdArray::eye(3, Some(3), -1);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        assert_eq!(arr.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn from_diag_main_diagonal() {
        let v = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let arr = NdArray::from_diag(&v, 0);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        assert_eq!(arr.as_slice(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn from_diag_positive_offset() {
        let v = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let arr = NdArray::from_diag(&v, 1);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
        assert_eq!(arr.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn from_diag_negative_offset() {
        let v = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let arr = NdArray::from_diag(&v, -1);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
        assert_eq!(arr.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn diagonal_main() {
        let arr = NdArray::from_vec(Shape::d2(3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let diag = arr.diagonal(0);
        assert_eq!(diag.shape().dims(), &[3]);
        assert_eq!(diag.as_slice(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diagonal_positive_offset() {
        let arr = NdArray::from_vec(Shape::d2(3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let diag = arr.diagonal(1);
        assert_eq!(diag.shape().dims(), &[2]);
        assert_eq!(diag.as_slice(), &[2.0, 6.0]);
    }

    #[test]
    fn diagonal_negative_offset() {
        let arr = NdArray::from_vec(Shape::d2(3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let diag = arr.diagonal(-1);
        assert_eq!(diag.shape().dims(), &[2]);
        assert_eq!(diag.as_slice(), &[4.0, 8.0]);
    }

    #[test]
    fn diagonal_rectangular() {
        let arr = NdArray::from_vec(Shape::d2(2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let diag = arr.diagonal(0);
        assert_eq!(diag.shape().dims(), &[2]);
        assert_eq!(diag.as_slice(), &[1.0, 6.0]);
    }

    #[test]
    fn outer_product() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![3.0, 4.0, 5.0]);
        let result = NdArray::outer(&a, &b);
        assert_eq!(result.shape().dims(), &[2, 3]);
        // [[3, 4, 5], [6, 8, 10]]
        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn outer_single_elements() {
        let a = NdArray::from_vec(Shape::d1(1), vec![2.0]);
        let b = NdArray::from_vec(Shape::d1(1), vec![3.0]);
        let result = NdArray::outer(&a, &b);
        assert_eq!(result.shape().dims(), &[1, 1]);
        assert_eq!(result.as_slice(), &[6.0]);
    }
}