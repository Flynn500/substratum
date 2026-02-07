#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub enum Node {
    Internal {
        feature: usize,
        threshold: f64,
        left: NodeId,
        right: NodeId,
    },

    Leaf {
        value: f64,
        n_samples: usize,
    },
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }
}