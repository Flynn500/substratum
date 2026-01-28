  
use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;


impl NdArray<f64> {
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