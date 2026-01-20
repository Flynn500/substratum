use std::{cmp::Ordering, collections::BinaryHeap};
use crate::{array::{NdArray, Shape}, spatial::common::{DistanceMetric, KernelType, HeapItem}};


#[derive(Debug, Clone)]
pub struct VPNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>, 

    pub vantage: usize,
    pub radius: f64,

    pub min_dist: f64,
    pub max_dist: f64,
}

#[derive(Debug, Clone)]
pub struct VPTree {
    pub nodes: Vec<VPNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}


impl VPTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize, metric: DistanceMetric) -> Self{
        let mut tree = VPTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size,
            metric,
        };

        tree.build_recursive(0, n_points);
        tree
    }

    pub fn from_ndarray(array: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, leaf_size, metric)
    }

    fn get_point(&self, i: usize) -> &[f64] {
        let idx = self.indices[i];
        &self.data[idx * self.dim..(idx + 1) * self.dim]
    }

    fn compute_distances_and_partition(&mut self, start: usize, end: usize, vantage_idx: usize) -> (f64, f64, f64, usize) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        let vantage_point = self.get_point(vantage_idx).to_vec();

        let mut idx_dist: Vec<(usize, f64)> = (start..end)
            .map(|i| {
                let p = self.get_point(i);
                let dist = self.metric.distance(p, &vantage_point);
                min = min.min(dist);
                max = max.max(dist);
                (i, dist)
            })
            .collect();

        let n = idx_dist.len();
        let mid = n / 2;

        idx_dist.select_nth_unstable_by(mid, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });

        let median_radius = if n % 2 == 1 {
            idx_dist[mid].1
        } else {
            let lower_max = idx_dist[..mid]
                .iter()
                .map(|(_, d)| *d)
                .fold(f64::NEG_INFINITY, f64::max);
            (lower_max + idx_dist[mid].1) * 0.5
        };

        let reordered: Vec<usize> = idx_dist.iter().map(|(i, _)| self.indices[*i]).collect();
        self.indices[start..end].copy_from_slice(&reordered);

        (median_radius, min, max, start + mid)
    }

    fn min_dist_to_node(&self, query: &[f64], node: &VPNode) -> f64 {
        let vantage_point = self.get_point(node.vantage);
        let d_qv = self.metric.distance(query, vantage_point);

        let l1 = d_qv - node.max_dist;
        let l2 = node.min_dist - d_qv;

        l1.max(l2).max(0.0)
    }

    fn max_dist_to_node(&self, query: &[f64], node: &VPNode) -> f64 {
        let vantage_point = self.get_point(node.vantage);
        let d_qv = self.metric.distance(query, vantage_point);

        d_qv + node.max_dist
    }

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let vantage_idx = start; 
        let (radius, min_dist, max_dist, mid) = self.compute_distances_and_partition(start, end, vantage_idx);
    
        let node_idx = self.nodes.len();

        self.nodes.push(VPNode { 
            start, 
            end, 
            left: None, 
            right: None, 
            vantage: vantage_idx,
            radius, 
            min_dist, 
            max_dist, 
        });
        
        let count = end - start;

        if count <= self.leaf_size {
            return node_idx;
        }

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);
        
        node_idx
    }
}