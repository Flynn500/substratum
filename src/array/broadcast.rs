use crate::array::shape::Shape;

pub struct BroadcastIter {
    output_shape: Shape,
    current: Vec<usize>,
    shape_a: Shape,
    shape_b: Shape,
    strides_a: Vec<usize>,
    strides_b: Vec<usize>,
    done: bool,
}

impl BroadcastIter {
    pub fn new(shape_a: &Shape, shape_b: &Shape) -> Option<Self> {
        let output_shape = shape_a.broadcast(shape_b)?;
        let ndim = output_shape.ndim();

        if ndim == 0 {
            return Some(BroadcastIter {
                output_shape,
                current: vec![],
                shape_a: shape_a.clone(),
                shape_b: shape_b.clone(),
                strides_a: vec![],
                strides_b: vec![],
                done: false,
            });
        }

        let strides_a = shape_a.strides_row_major();
        let strides_b = shape_b.strides_row_major();

        Some(BroadcastIter {
            output_shape,
            current: vec![0; ndim],
            shape_a: shape_a.clone(),
            shape_b: shape_b.clone(),
            strides_a,
            strides_b,
            done: false,
        })
    }

    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    fn offset_a(&self) -> usize {
        let ndim_out = self.output_shape.ndim();
        let ndim_a = self.shape_a.ndim();
        
        let mut offset = 0;
        for i in 0..ndim_a {
            let out_idx = ndim_out - ndim_a + i;
            let idx = self.current[out_idx];
            
            let dim_size = self.shape_a.dim(i).unwrap();
            let actual_idx = if dim_size == 1 { 0 } else { idx };
            
            offset += actual_idx * self.strides_a[i];
        }
        offset
    }

    fn offset_b(&self) -> usize {
        let ndim_out = self.output_shape.ndim();
        let ndim_b = self.shape_b.ndim();
        
        let mut offset = 0;
        for i in 0..ndim_b {
            let out_idx = ndim_out - ndim_b + i;
            let idx = self.current[out_idx];
            
            let dim_size = self.shape_b.dim(i).unwrap();
            let actual_idx = if dim_size == 1 { 0 } else { idx };
            
            offset += actual_idx * self.strides_b[i];
        }
        offset
    }

    fn advance(&mut self) -> bool {
        if self.output_shape.ndim() == 0 {
            self.done = true;
            return false;
        }

        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.output_shape.dim(i).unwrap() {
                return true;
            }
            self.current[i] = 0;
        }
        
        self.done = true;
        false
    }
}

impl Iterator for BroadcastIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = (self.offset_a(), self.offset_b());
        self.advance();
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.output_shape.size();
        (size, Some(size))
    }
}

impl ExactSizeIterator for BroadcastIter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_shape_iteration() {
        let a = Shape::d1(3);
        let b = Shape::d1(3);
        
        let offsets: Vec<_> = BroadcastIter::new(&a, &b).unwrap().collect();
        assert_eq!(offsets, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn broadcast_scalar() {
        let a = Shape::d1(3);
        let b = Shape::scalar();
        
        let offsets: Vec<_> = BroadcastIter::new(&a, &b).unwrap().collect();
        assert_eq!(offsets, vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn broadcast_trailing_dim() {
        let a = Shape::d2(2, 3);
        let b = Shape::d1(3);
        
        let offsets: Vec<_> = BroadcastIter::new(&a, &b).unwrap().collect();
        assert_eq!(offsets, vec![
            (0, 0), (1, 1), (2, 2),
            (3, 0), (4, 1), (5, 2),
        ]);
    }

    #[test]
    fn broadcast_size_one_dim() {
        let a = Shape::new(vec![3, 1]);
        let b = Shape::new(vec![1, 4]);
        
        let iter = BroadcastIter::new(&a, &b).unwrap();
        assert_eq!(iter.output_shape().dims(), &[3, 4]);
        
        let offsets: Vec<_> = iter.collect();
        assert_eq!(offsets, vec![
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3), 
        ]);
    }

    #[test]
    fn incompatible_returns_none() {
        let a = Shape::d2(3, 4);
        let b = Shape::d2(2, 4);
        assert!(BroadcastIter::new(&a, &b).is_none());
    }

    #[test]
    fn scalar_scalar() {
        let a = Shape::scalar();
        let b = Shape::scalar();
        
        let offsets: Vec<_> = BroadcastIter::new(&a, &b).unwrap().collect();
        assert_eq!(offsets, vec![(0, 0)]);
    }
}