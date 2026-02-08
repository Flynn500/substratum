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

impl<T: Copy> NdArray<T> {
    pub fn item(&self) -> T {
        assert_eq!(
            self.len(),
            1,
            "item() can only be called on arrays with exactly one element, got {} elements",
            self.len()
        );
        self.as_slice()[0]
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

    pub fn take(&self, indices: &[usize]) -> NdArray<T> {
        let data: Vec<T> = indices.iter()
            .map(|&i| self.as_slice()[i])
            .collect();

        NdArray::from_vec(Shape::d1(data.len()), data)
    }
}

