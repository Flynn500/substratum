pub fn gini(class_counts: &[usize]) -> f64 {
    let total: usize = class_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let sum_sq: f64 = class_counts
        .iter()
        .map(|&c| {
            let p = c as f64 / total_f;
            p * p
        })
        .sum();
    1.0 - sum_sq
}

pub fn entropy(class_counts: &[usize]) -> f64 {
    let total: usize = class_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let ent: f64 = class_counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.log2()
        })
        .sum();
    ent
}

pub fn mse_stats(values: impl Iterator<Item = f64>) -> MseResult {
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count: usize = 0;

    for v in values {
        sum += v;
        sum_sq += v * v;
        count += 1;
    }

    if count == 0 {
        return MseResult {
            mse: 0.0,
            mean: 0.0,
            count: 0,
        };
    }

    let mean = sum / count as f64;
    let mse = (sum_sq / count as f64) - mean * mean;

    MseResult {
        mse: mse.max(0.0),
        mean,
        count,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MseResult {
    pub mse: f64,
    pub mean: f64,
    pub count: usize,
}

pub fn weighted_impurity(
    n_left: usize,
    impurity_left: f64,
    n_right: usize,
    impurity_right: f64,
) -> f64 {
    let total = (n_left + n_right) as f64;
    if total == 0.0 {
        return 0.0;
    }
    (n_left as f64 * impurity_left + n_right as f64 * impurity_right) / total
}
