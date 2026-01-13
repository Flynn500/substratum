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

    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose requires 2D array");
        let (n, m) = (self.shape().dims()[0], self.shape().dims()[1]);
        let mut data = Vec::with_capacity(n * m);

        for j in 0..m {
            for i in 0..n {
                data.push(*self.get(&[i, j]).unwrap());
            }
        }

        NdArray::from_vec(Shape::d2(m, n), data)
    }

    pub fn t(&self) -> Self {
        self.transpose()
    }


    pub fn matmul(&self, other: &NdArray<f64>) -> Self {
        match (self.ndim(), other.ndim()) {
            (1, 1) => {
                assert_eq!(self.len(), other.len(), "Vectors must have same length");
                let sum: f64 = self.as_slice().iter()
                    .zip(other.as_slice().iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                NdArray::from_vec(Shape::d1(1), vec![sum])
            }
            (2, 1) => {
                let (n, k) = (self.shape().dims()[0], self.shape().dims()[1]);
                assert_eq!(k, other.len(), "Inner dimensions must match");
                let mut data = Vec::with_capacity(n);
                for i in 0..n {
                    let sum: f64 = (0..k)
                        .map(|j| self.get(&[i, j]).unwrap() * other.get(&[j]).unwrap())
                        .sum();
                    data.push(sum);
                }
                NdArray::from_vec(Shape::d1(n), data)
            }
            (1, 2) => {
                let (k, m) = (other.shape().dims()[0], other.shape().dims()[1]);
                assert_eq!(self.len(), k, "Inner dimensions must match");
                let mut data = Vec::with_capacity(m);
                for j in 0..m {
                    let sum: f64 = (0..k)
                        .map(|i| self.get(&[i]).unwrap() * other.get(&[i, j]).unwrap())
                        .sum();
                    data.push(sum);
                }
                NdArray::from_vec(Shape::d1(m), data)
            }
            (2, 2) => {
                let (n, k1) = (self.shape().dims()[0], self.shape().dims()[1]);
                let (k2, m) = (other.shape().dims()[0], other.shape().dims()[1]);
                assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);
                let mut data = Vec::with_capacity(n * m);
                for i in 0..n {
                    for j in 0..m {
                        let sum: f64 = (0..k1)
                            .map(|k| self.get(&[i, k]).unwrap() * other.get(&[k, j]).unwrap())
                            .sum();
                        data.push(sum);
                    }
                }
                NdArray::from_vec(Shape::d2(n, m), data)
            }
            _ => panic!("matmul requires 1D or 2D arrays"),
        }
    }

    pub fn dot(&self, other: &NdArray<f64>) -> Self {
        self.matmul(other)
    }

    pub fn cholesky(&self) -> Result<NdArray<f64>, &'static str> {
        if self.ndim() != 2 {
            return Err("Cholesky decomposition requires a 2D matrix");
        }

        let n = self.shape().dims()[0];
        let m = self.shape().dims()[1];

        if n != m {
            return Err("Cholesky decomposition requires a square matrix");
        }

        let mut l = NdArray::zeros(Shape::d2(n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                for k in 0..j {
                    sum += l.get(&[i, k]).unwrap() * l.get(&[j, k]).unwrap();
                }

                if i == j {
                    let diag = self.get(&[i, i]).unwrap() - sum;
                    if diag <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    *l.get_mut(&[i, j]).unwrap() = diag.sqrt();
                } else {
                    let l_jj = *l.get(&[j, j]).unwrap();
                    *l.get_mut(&[i, j]).unwrap() = (self.get(&[i, j]).unwrap() - sum) / l_jj;
                }
            }
        }

        Ok(l)
    }


    pub fn qr(&self) -> Result<(NdArray<f64>, NdArray<f64>), &'static str> {
        if self.ndim() != 2 {
            return Err("QR decomposition requires a 2D matrix");
        }

        let m = self.shape().dims()[0];
        let n = self.shape().dims()[1];

        let mut r = self.clone();
        let mut q = NdArray::eye(m, None, 0);

        let k_max = if m > n { n } else { m };

        for k in 0..k_max {
            let mut x = Vec::with_capacity(m - k);
            for i in k..m {
                x.push(*r.get(&[i, k]).unwrap());
            }

            let norm_x: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-14 {
                continue;
            }

            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            let mut v = x.clone();
            v[0] += sign * norm_x;

            let norm_v: f64 = v.iter().map(|&val| val * val).sum::<f64>().sqrt();
            if norm_v < 1e-14 {
                continue;
            }
            for val in &mut v {
                *val /= norm_v;
            }

            let mut vt_r = vec![0.0; n - k];
            for j in k..n {
                let mut sum = 0.0;
                for i in 0..(m - k) {
                    sum += v[i] * r.get(&[k + i, j]).unwrap();
                }
                vt_r[j - k] = sum;
            }

            for i in 0..(m - k) {
                for j in k..n {
                    let old_val = *r.get(&[k + i, j]).unwrap();
                    *r.get_mut(&[k + i, j]).unwrap() = old_val - 2.0 * v[i] * vt_r[j - k];
                }
            }

            let mut q_v = vec![0.0; m];
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..(m - k) {
                    sum += q.get(&[i, k + j]).unwrap() * v[j];
                }
                q_v[i] = sum;
            }

            for i in 0..m {
                for j in 0..(m - k) {
                    let old_val = *q.get(&[i, k + j]).unwrap();
                    *q.get_mut(&[i, k + j]).unwrap() = old_val - 2.0 * q_v[i] * v[j];
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                if i > j {
                    *r.get_mut(&[i, j]).unwrap() = 0.0;
                }
            }
        }

        Ok((q, r))
    }

    pub fn eig(&self) -> Result<(NdArray<f64>, NdArray<f64>), &'static str> {
        self.eig_with_params(1000, 1e-10)
    }

    pub fn eig_with_params(&self, max_iter: usize, tol: f64) -> Result<(NdArray<f64>, NdArray<f64>), &'static str> {
        if self.ndim() != 2 {
            return Err("Eigendecomposition requires a 2D matrix");
        }

        let n = self.shape().dims()[0];
        let m = self.shape().dims()[1];

        if n != m {
            return Err("Eigendecomposition requires a square matrix");
        }

        if n == 0 {
            return Ok((
                NdArray::from_vec(Shape::d1(0), vec![]),
                NdArray::from_vec(Shape::d2(0, 0), vec![]),
            ));
        }

        if n == 1 {
            let eigenvalue = *self.get(&[0, 0]).unwrap();
            return Ok((
                NdArray::from_vec(Shape::d1(1), vec![eigenvalue]),
                NdArray::from_vec(Shape::d2(1, 1), vec![1.0]),
            ));
        }

        let mut a = self.clone();

        let mut v = NdArray::eye(n, None, 0);

        for _ in 0..max_iter {
            let (q, r) = a.qr()?;

            a = r.matmul(&q);

            v = v.matmul(&q);

            let mut off_diag_sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let val = *a.get(&[i, j]).unwrap();
                        off_diag_sum += val * val;
                    }
                }
            }

            if off_diag_sum.sqrt() < tol {
                break;
            }
        }

        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(*a.get(&[i, i]).unwrap());
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j].abs().partial_cmp(&eigenvalues[i].abs()).unwrap()
        });

        let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

        let mut sorted_eigenvectors = Vec::with_capacity(n * n);
        for i in 0..n {
            for &idx in &indices {
                sorted_eigenvectors.push(*v.get(&[i, idx]).unwrap());
            }
        }

        Ok((
            NdArray::from_vec(Shape::d1(n), sorted_eigenvalues),
            NdArray::from_vec(Shape::d2(n, n), sorted_eigenvectors),
        ))
    }

    pub fn eigvals(&self) -> Result<NdArray<f64>, &'static str> {
        let (eigenvalues, _) = self.eig()?;
        Ok(eigenvalues)
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

    // Transpose tests
    #[test]
    fn transpose_square() {
        let arr = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let transposed = arr.transpose();
        assert_eq!(transposed.shape().dims(), &[2, 2]);
        // [[1, 2], [3, 4]] -> [[1, 3], [2, 4]]
        assert_eq!(transposed.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_rectangular() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = arr.transpose();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        assert_eq!(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_t_alias() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t1 = arr.transpose();
        let t2 = arr.t();
        assert_eq!(t1.as_slice(), t2.as_slice());
        assert_eq!(t1.shape().dims(), t2.shape().dims());
    }

    #[test]
    fn transpose_double_is_identity() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let double_transposed = arr.transpose().transpose();
        assert_eq!(double_transposed.shape().dims(), arr.shape().dims());
        assert_eq!(double_transposed.as_slice(), arr.as_slice());
    }

    // Matmul tests - 1D x 1D (dot product)
    #[test]
    fn matmul_1d_1d() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![4.0, 5.0, 6.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[1]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result.as_slice(), &[32.0]);
    }

    // Matmul tests - 2D x 1D (matrix-vector)
    #[test]
    fn matmul_2d_1d() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[2]);
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert_eq!(result.as_slice(), &[14.0, 32.0]);
    }

    // Matmul tests - 1D x 2D (vector-matrix)
    #[test]
    fn matmul_1d_2d() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let b = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[3]);
        // [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
        assert_eq!(result.as_slice(), &[9.0, 12.0, 15.0]);
    }

    // Matmul tests - 2D x 2D (matrix-matrix)
    #[test]
    fn matmul_2d_2d_square() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(Shape::d2(2, 2), vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[2, 2]);
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_2d_2d_rectangular() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d2(3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[2, 2]);
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]] = [[58, 64], [139, 154]]
        assert_eq!(result.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_identity() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let identity = NdArray::eye(2, None, 0);
        let result = a.matmul(&identity);
        assert_eq!(result.as_slice(), a.as_slice());
    }

    // Dot tests (alias for matmul)
    #[test]
    fn dot_is_matmul_alias() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(Shape::d2(2, 2), vec![5.0, 6.0, 7.0, 8.0]);
        let matmul_result = a.matmul(&b);
        let dot_result = a.dot(&b);
        assert_eq!(matmul_result.as_slice(), dot_result.as_slice());
        assert_eq!(matmul_result.shape().dims(), dot_result.shape().dims());
    }

    #[test]
    fn dot_1d_vectors() {
        let a = NdArray::from_vec(Shape::d1(4), vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(Shape::d1(4), vec![2.0, 3.0, 4.0, 5.0]);
        let result = a.dot(&b);
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_eq!(result.as_slice(), &[40.0]);
    }

    // Error cases
    #[test]
    #[should_panic(expected = "Inner dimensions must match")]
    fn matmul_2d_2d_dimension_mismatch() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "Inner dimensions must match")]
    fn matmul_2d_1d_dimension_mismatch() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "Vectors must have same length")]
    fn matmul_1d_1d_length_mismatch() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "transpose requires 2D array")]
    fn transpose_1d_panics() {
        let arr = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        arr.transpose();
    }

    #[test]
    fn cholesky_basic() {
        // Symmetric positive-definite matrix
        // [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        let a = NdArray::from_vec(
            Shape::d2(3, 3),
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]
        );
        let l = a.cholesky().unwrap();
        
        // Expected L: [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        assert_eq!(l.shape().dims(), &[3, 3]);
        assert_eq!(l.as_slice(), &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // Not positive definite (negative eigenvalue)
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 2.0, 1.0]);
        assert!(a.cholesky().is_err());
    }

    #[test]
    fn cholesky_non_square_fails() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(a.cholesky().is_err());
    }

    // QR decomposition tests
    #[test]
    fn qr_square_matrix() {
        let a = NdArray::from_vec(Shape::d2(3, 3), vec![
            12.0, -51.0, 4.0,
            6.0, 167.0, -68.0,
            -4.0, 24.0, -41.0
        ]);
        let (q, r) = a.qr().unwrap();

        // Verify Q * R ≈ A
        let reconstructed = q.matmul(&r);
        for i in 0..9 {
            assert!((reconstructed.as_slice()[i] - a.as_slice()[i]).abs() < 1e-10);
        }

        // Verify Q is orthogonal: Q^T * Q ≈ I
        let qt_q = q.t().matmul(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((*qt_q.get(&[i, j]).unwrap() - expected).abs() < 1e-10);
            }
        }

        // Verify R is upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(r.get(&[i, j]).unwrap().abs() < 1e-10);
            }
        }
    }

    #[test]
    fn qr_rectangular_matrix() {
        let a = NdArray::from_vec(Shape::d2(4, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0
        ]);
        let (q, r) = a.qr().unwrap();

        // Q should be 4x4, R should be 4x2
        assert_eq!(q.shape().dims(), &[4, 4]);
        assert_eq!(r.shape().dims(), &[4, 2]);

        // Verify Q * R ≈ A
        let reconstructed = q.matmul(&r);
        for i in 0..8 {
            assert!((reconstructed.as_slice()[i] - a.as_slice()[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn qr_identity_matrix() {
        let a = NdArray::eye(3, None, 0);
        let (q, r) = a.qr().unwrap();

        // For identity, Q and R should both be close to identity (or Q=-I, R=-I)
        let reconstructed = q.matmul(&r);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((*reconstructed.get(&[i, j]).unwrap() - expected).abs() < 1e-10);
            }
        }
    }

    // Eigendecomposition tests
    #[test]
    fn eig_diagonal_matrix() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let a = NdArray::from_vec(Shape::d2(3, 3), vec![
            3.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 1.0
        ]);
        let (eigenvalues, _eigenvectors) = a.eig().unwrap();

        // Eigenvalues should be 3, 2, 1 (sorted by absolute value descending)
        assert_eq!(eigenvalues.len(), 3);
        assert!((eigenvalues.as_slice()[0] - 3.0).abs() < 1e-8);
        assert!((eigenvalues.as_slice()[1] - 2.0).abs() < 1e-8);
        assert!((eigenvalues.as_slice()[2] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn eig_symmetric_matrix() {
        // Symmetric matrix [[2, 1], [1, 2]]
        // Eigenvalues: 3 and 1
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![2.0, 1.0, 1.0, 2.0]);
        let (eigenvalues, eigenvectors) = a.eig().unwrap();

        // Check eigenvalues (sorted by absolute value)
        assert_eq!(eigenvalues.len(), 2);
        let ev1 = eigenvalues.as_slice()[0];
        let ev2 = eigenvalues.as_slice()[1];
        assert!((ev1 - 3.0).abs() < 1e-8 || (ev1 - 1.0).abs() < 1e-8);
        assert!((ev2 - 3.0).abs() < 1e-8 || (ev2 - 1.0).abs() < 1e-8);
        assert!((ev1 - ev2).abs() > 1.5); // They should be different

        // Verify A * v = λ * v for each eigenvector
        for col in 0..2 {
            let lambda = eigenvalues.as_slice()[col];
            let v = NdArray::from_vec(Shape::d1(2), vec![
                *eigenvectors.get(&[0, col]).unwrap(),
                *eigenvectors.get(&[1, col]).unwrap()
            ]);
            let av = a.matmul(&v);
            let lambda_v = NdArray::from_vec(Shape::d1(2), vec![
                lambda * v.as_slice()[0],
                lambda * v.as_slice()[1]
            ]);

            for i in 0..2 {
                assert!((av.as_slice()[i] - lambda_v.as_slice()[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn eig_1x1_matrix() {
        let a = NdArray::from_vec(Shape::d2(1, 1), vec![5.0]);
        let (eigenvalues, eigenvectors) = a.eig().unwrap();

        assert_eq!(eigenvalues.len(), 1);
        assert!((eigenvalues.as_slice()[0] - 5.0).abs() < 1e-10);
        assert_eq!(eigenvectors.shape().dims(), &[1, 1]);
        assert!((eigenvectors.as_slice()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eig_identity_matrix() {
        let a = NdArray::eye(3, None, 0);
        let (eigenvalues, _eigenvectors) = a.eig().unwrap();

        // All eigenvalues of identity matrix are 1
        assert_eq!(eigenvalues.len(), 3);
        for &ev in eigenvalues.as_slice() {
            assert!((ev - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn eig_non_square_fails() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(a.eig().is_err());
    }

    #[test]
    fn eig_1d_fails() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        assert!(a.eig().is_err());
    }

    #[test]
    fn eigvals_returns_only_eigenvalues() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![2.0, 1.0, 1.0, 2.0]);
        let eigenvalues = a.eigvals().unwrap();

        assert_eq!(eigenvalues.ndim(), 1);
        assert_eq!(eigenvalues.len(), 2);
    }

    #[test]
    fn eig_larger_symmetric_matrix() {
        // 4x4 symmetric matrix
        let a = NdArray::from_vec(Shape::d2(4, 4), vec![
            4.0, 1.0, 1.0, 1.0,
            1.0, 4.0, 1.0, 1.0,
            1.0, 1.0, 4.0, 1.0,
            1.0, 1.0, 1.0, 4.0
        ]);
        let (eigenvalues, eigenvectors) = a.eig().unwrap();

        assert_eq!(eigenvalues.len(), 4);

        // Verify reconstruction: A * V ≈ V * diag(eigenvalues)
        // For each eigenvector column, A * v should equal lambda * v
        for col in 0..4 {
            let lambda = eigenvalues.as_slice()[col];
            let mut v_data = Vec::with_capacity(4);
            for row in 0..4 {
                v_data.push(*eigenvectors.get(&[row, col]).unwrap());
            }
            let v = NdArray::from_vec(Shape::d1(4), v_data.clone());
            let av = a.matmul(&v);

            for i in 0..4 {
                let expected = lambda * v_data[i];
                assert!((av.as_slice()[i] - expected).abs() < 1e-5,
                    "Mismatch at eigenvector {}, element {}: av={}, λv={}",
                    col, i, av.as_slice()[i], expected);
            }
        }
    }
}