use crate::{array::NdArray, spatial::common::DistanceMetric};
use super::spatial_query::{SpatialQuery};


#[derive(Clone, Debug)]
pub struct BallNode {
    pub center: Vec<f64>,
    pub radius: f64,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}


#[derive(Debug, Clone)]
pub struct BallTree {
    pub nodes: Vec<BallNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl BallTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize, metric: DistanceMetric) -> Self{
        let mut tree = BallTree {
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

    fn init_node(&self, start: usize, end: usize) -> (Vec<f64>, f64) {
        let n = (end - start) as f64;
        let mut centroid = vec![0.0; self.dim];

        for i in start..end {
            let p = self.get_point_from_idx(i);
            for (j, &x) in p.iter().enumerate() {
                centroid[j] += x;
            }
        }

        
        for c in &mut centroid {
            *c /= n;
        }

        let mut max_dist: f64 = 0.0;
        for i in start..end {
            let p = self.get_point_from_idx(i);
            let dist = self.metric.distance(p, &centroid);

            if  dist > max_dist {
                max_dist = dist;
            }
        }
        (centroid, max_dist)
    }

    fn select_split_dim(&self, start: usize, end: usize) -> usize {
        let mut best_dim = 0;
        let mut best_spread = 0.0;
        
        for d in 0..self.dim {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            
            for i in start..end {
                let val = self.get_point_from_idx(i)[d];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
            
            let spread = max_val - min_val;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }
        best_dim
    }

    fn partition(&mut self, start: usize, end: usize, dim: usize) -> usize {
        let mut slots_by_key: Vec<(f64, usize)> = (start..end)
            .map(|slot| (self.get_point_from_idx(slot)[dim], slot))
            .collect();

        let mid_offset = (end - start) / 2;

        slots_by_key.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        let new_order: Vec<usize> = slots_by_key
            .iter()
            .map(|&(_key, slot)| self.indices[slot])
            .collect();

        self.indices[start..end].copy_from_slice(&new_order);

        start + mid_offset
    }


    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (center, radius) = self.init_node(start, end);
        
        let node_idx = self.nodes.len();

        self.nodes.push(BallNode {
            center,
            radius,
            start,
            end,
            left: None,
            right: None,
        });
        
        let count = end - start;

        if count <= self.leaf_size {
            return node_idx;
        }

        let dim = self.select_split_dim(start, end);
        let mut mid = self.partition(start, end, dim);

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

impl SpatialQuery for BallTree {
    type Node = BallNode;

    fn nodes(&self) -> &[BallNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { &self.data }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }

    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node_idx];
        let d = self.metric.distance(query, &node.center);
        (d - node.radius).max(0.0)
    }

    fn knn_child_order(&self, node_idx: usize, query: &[f64]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let dl = self.metric.distance(query, &self.nodes[l].center);
        let dr = self.metric.distance(query, &self.nodes[r].center);
        if dl <= dr { (l, r) } else { (r, l) }
    }
}