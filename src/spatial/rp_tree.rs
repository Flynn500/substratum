use std::collections::BinaryHeap;

use crate::{Generator, array::{NdArray, Shape}, projection::{ProjectionType, RandomProjection, random_projection::ProjectionDirection}, spatial::{HeapItem, common::DistanceMetric}};
use super::spatial_query::{SpatialQuery};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RPNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub direction: ProjectionDirection,
    pub split: f64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPTree {
    pub nodes: Vec<RPNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub projection_type: ProjectionType,
    rng: Generator,
}

impl RPTree {
    pub fn new(
        data: &NdArray<f64>,
        leaf_size: usize,
        metric: DistanceMetric,
        projection_type: ProjectionType,
        seed: u64,
    ) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

        let rng = Generator::from_seed(seed);

        let mut tree = RPTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: data.clone(),
            n_points,
            dim,
            leaf_size,
            metric,
            projection_type,
            rng,
        };

        tree.build_recursive(0, n_points);
        tree.reorder_data();
        tree
    }

    fn reorder_data(&mut self) {
        let mut new_data = vec![0.0; self.data.len()];

        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(self.data.row(old_idx));
        }

        self.data = NdArray::from_vec(Shape::new(vec![self.n_points, self.dim]), new_data);
    }

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let node_idx = self.nodes.len();

        self.nodes.push(RPNode {
            start,
            end,
            left: None,
            right: None,
            direction: ProjectionDirection::Empty,
            split: 0.0,
        });

        let count = end - start;
        if count <= self.leaf_size {
            return node_idx;
        }

        let (direction, split, mid) = RandomProjection::rp_split(
            &self.data,
            &mut self.indices,
            start,
            end,
            self.dim,
            self.projection_type,
            &mut self.rng,
        );

        self.nodes[node_idx].direction = direction;
        self.nodes[node_idx].split = split;

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    fn knn_recursive_inner(
        &self,
        node_idx: usize,
        query: &[f64],
        heap: &mut BinaryHeap<HeapItem>,
        k: usize,
        accumulated_dist_sq: f64,
    ) {
        if heap.len() == k {
            let worst = heap.peek().unwrap().distance;
            if accumulated_dist_sq > worst * worst {
                return;
            }
        }

        let node = &self.nodes[node_idx];

        if node.left.is_none() {
            for i in node.start..node.end {
                let dist = self.metric.distance(query, self.get_point(i));
                if heap.len() < k {
                    heap.push(HeapItem { distance: dist, index: self.indices[i] });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(HeapItem { distance: dist, index: self.indices[i] });
                }
            }
            return;
        }

        let (first, second, margin) = self.node_projection(node_idx, query);

        self.knn_recursive_inner(first, query, heap, k, accumulated_dist_sq);
        self.knn_recursive_inner(second, query, heap, k, accumulated_dist_sq + margin * margin);
    }

    fn radius_recursive_inner(
        &self,
        node_idx: usize,
        query: &[f64],
        radius: f64,
        results: &mut Vec<(usize, f64)>,
        accumulated_dist_sq: f64,
    ) {
        if accumulated_dist_sq > radius * radius {
            return;
        }

        let node = &self.nodes[node_idx];

        if node.left.is_none() {
            for i in node.start..node.end {
                let dist = self.metric.distance(query, self.get_point(i));
                if dist <= radius {
                    results.push((self.indices[i], dist));
                }
            }
            return;
        }

        let proj = node.direction.project(query);
        let margin = (proj - node.split).abs();

        let (first, second) = if proj <= node.split {
            (node.left.unwrap(), node.right.unwrap())
        } else {
            (node.right.unwrap(), node.left.unwrap())
        };

        self.radius_recursive_inner(first, query, radius, results, accumulated_dist_sq);
        self.radius_recursive_inner(second, query, radius, results, accumulated_dist_sq + margin * margin);
    }
}


impl SpatialQuery for RPTree {
    type Node = RPNode;

    fn nodes(&self) -> &[RPNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { self.data.as_slice() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize {self.n_points}

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }


    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node_idx];
        let proj = node.direction.project(query);
        (proj - node.split).abs()
    }

    fn knn_child_order(&self, _node_idx: usize, _query: &[f64]) -> (usize, usize) { (0,0) }

    fn node_projection(&self, node_idx: usize, query: &[f64]) -> (usize, usize, f64) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let proj = node.direction.project(query);
        let dist = (proj - node.split).abs();
        let (first, second) = if proj <= node.split { (l, r) } else { (r, l) };
        (first, second, dist)
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BinaryHeap<HeapItem>, k: usize) {
        self.knn_recursive_inner(node_idx, query, heap, k, 0.0)
    }

    fn query_radius_recursive(&self, node_idx: usize, query: &[f64], radius: f64, results: &mut Vec<(usize, f64)>) {
        self.radius_recursive_inner(node_idx, query, radius, results, 0.0);
    }
}