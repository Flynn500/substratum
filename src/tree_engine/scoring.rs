pub fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }
    let n_f = n as f64;
    let harmonic = (n_f - 1.0).ln() + 0.5772156649;
    2.0 * harmonic - 2.0 * (n_f - 1.0) / n_f
}

pub fn anomaly_score(depth: usize, leaf_n_samples: usize, n_training: usize) -> f64 {
    let c_train = average_path_length(n_training);
    if c_train == 0.0 {
        return 0.5;
    }
    let adjusted_path = depth as f64 + average_path_length(leaf_n_samples);
    2.0_f64.powf(-adjusted_path / c_train)
}
