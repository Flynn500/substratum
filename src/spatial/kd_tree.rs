use crate::{array::NdArray, spatial::common::DistanceMetric};
use super::spatial_query::{SpatialQuery};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KDNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,

    pub axis: usize,
    pub split: f64,

    pub bbox_min: Vec<f64>,
    pub bbox_max: Vec<f64>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KDTree {
    pub nodes: Vec<KDNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl KDTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize, metric: DistanceMetric) -> Self{
        let mut tree = KDTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size,
            metric,
        };

        tree.build_recursive(0, n_points);
        tree.reorder_data();
        tree
    }

    pub fn from_ndarray(array: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, leaf_size, metric)
    }

    fn reorder_data(&mut self) {
        let mut new_data = vec![0.0; self.data.len()];
        
        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let src = old_idx * self.dim;
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(&self.data[src..src + self.dim]);
        }
        
        self.data = new_data;
    }


    fn get_point_from_idx(&self, i: usize) -> &[f64] {
        let original_idx = self.indices[i];
        &self.data[original_idx * self.dim..(original_idx + 1) * self.dim]
    }

    fn init_node(&self, start: usize, end: usize) -> (Vec<f64>, Vec<f64>, usize) {
        let mut min = vec![f64::INFINITY; self.dim];
        let mut max = vec![f64::NEG_INFINITY; self.dim];

        for i in start..end {
            let p = self.get_point_from_idx(i);
            for (j, &x) in p.iter().enumerate() {
                max[j] = max[j].max(x);
                min[j] = min[j].min(x);

            }
        }
        let mut best_dim = 0;
        let mut best_spread = 0.0;
        for d in 0..self.dim {
            let spread = max[d] - min[d];
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        (min, max, best_dim)        
    }

    fn partition(&mut self, start: usize, end: usize, dim: usize) -> usize {
        let mut slots: Vec<(f64, usize)> = (start..end)
            .map(|slot| (self.get_point_from_idx(slot)[dim], slot))
            .collect();

        let mid_offset = (end - start) / 2;

        slots.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        let new_order: Vec<usize> = slots
            .iter()
            .map(|&(_val, slot)| self.indices[slot])
            .collect();

        self.indices[start..end].copy_from_slice(&new_order);

        start + mid_offset
    }


    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (min, max, axis)  = self.init_node(start, end);
        let node_idx = self.nodes.len();

        self.nodes.push(KDNode { 
            start, 
            end, 
            left: None, 
            right: None, 
            axis, 
            split: 0.0, 
            bbox_min: min, 
            bbox_max: max,
        });
        
        let count = end - start;

        if count <= self.leaf_size {
            return node_idx;
        }

        let mut mid = self.partition(start, end, axis);
        let split = self.get_point_from_idx(mid)[axis];
        self.nodes[node_idx].split = split;

        if mid == start {
            mid = start + 1;
        } else if mid == end {
            mid = end - 1;
        }

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);
        
        node_idx
    }
}

impl SpatialQuery for KDTree {
    type Node = KDNode;

    fn nodes(&self) -> &[KDNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { &self.data }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize {self.n_points}

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }


    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node_idx];
        match self.metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0;
                for d in 0..query.len() {
                    if query[d] < node.bbox_min[d] {
                        let diff = node.bbox_min[d] - query[d];
                        sum += diff * diff;
                    } else if query[d] > node.bbox_max[d] {
                        let diff = query[d] - node.bbox_max[d];
                        sum += diff * diff;
                    }
                }
                sum.sqrt()
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for d in 0..query.len() {
                    if query[d] < node.bbox_min[d] {
                        sum += node.bbox_min[d] - query[d];
                    } else if query[d] > node.bbox_max[d] {
                        sum += query[d] - node.bbox_max[d];
                    }
                }
                sum
            }
            DistanceMetric::Chebyshev => {
                let mut max_dist: f64 = 0.0;
                for d in 0..query.len() {
                    let dist = if query[d] < node.bbox_min[d] {
                        node.bbox_min[d] - query[d]
                    } else if query[d] > node.bbox_max[d] {
                        query[d] - node.bbox_max[d]
                    } else {
                        0.0
                    };
                    max_dist = max_dist.max(dist);
                }
                max_dist
            }
        }
    }

    fn knn_child_order(&self, node_idx: usize, query: &[f64]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        if query[node.axis] < node.split { (l, r) } else { (r, l) }
    }
}