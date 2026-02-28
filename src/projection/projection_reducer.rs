use crate::array::{NdArray};
use crate::random::Generator;
use super::{RandomProjection, ProjectionType};

pub struct ProjectionReducer {
    projection: RandomProjection,
}

impl ProjectionReducer {
    pub fn fit(
        input_dim: usize,
        output_dim: usize,
        projection_type: ProjectionType,
        rng: &mut Generator,
    ) -> Self {
        let projection = RandomProjection::new(input_dim, output_dim, projection_type, rng);
        Self { projection }
    }

    pub fn transform(&self, data: &NdArray<f64>) -> NdArray<f64> {
        self.projection.project(data)
    }

    pub fn fit_transform(
        data: &NdArray<f64>,
        output_dim: usize,
        projection_type: ProjectionType,
        rng: &mut Generator,
    ) -> (Self, NdArray<f64>) {
        let input_dim = data.shape().dims()[1];
        let reducer = Self::fit(input_dim, output_dim, projection_type, rng);
        let transformed = reducer.transform(data);
        (reducer, transformed)
    }

    pub fn input_dim(&self) -> usize { self.projection.input_dim }
    pub fn output_dim(&self) -> usize { self.projection.output_dim }
    pub fn projection(&self) -> &RandomProjection { &self.projection }
}