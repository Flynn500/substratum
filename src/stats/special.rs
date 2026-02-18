pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        panic!("gamma undefined for non-positive integers");
    }

    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        lanczos_gamma(x)
    }
}

fn lanczos_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let x = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}