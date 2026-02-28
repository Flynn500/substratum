//Xoshiro256** Generator
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generator {
    state: [u64; 4],
}

impl Generator {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos() as u64;
        
        Self::from_seed(seed)
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut state = [0u64; 4];
        let mut s = seed;
        
        for i in 0..4 {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            state[i] = z ^ (z >> 31);
        }
        
        Self { state }
    }

    pub fn usize_below(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }

    pub fn partial_shuffle(&mut self, indices: &mut [usize], k: usize) {
        let n = indices.len();
        for i in 0..k.min(n) {
            let j = i + self.usize_below(n - i);
            indices.swap(i, j);
        }
    }
    
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        
        let t = self.state[1] << 17;
        
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        
        result
    }
    pub fn next_f64(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            bits as f64 * (1.0 / (1u64 << 53) as f64)
        }

        pub fn next_gaussian(&mut self) -> f64 {
        let pi = std::f64::consts::PI;
        let mut r1 = self.next_f64();
        while r1 == 0.0 {
            r1 = self.next_f64();
        }
        let r2 = self.next_f64();
        f64::sqrt(-2.0 * f64::ln(r1)) * f64::sin(2.0 * pi * r2)
    }

    pub fn randint(&mut self, low: i64, high: i64, shape: Shape) -> NdArray<i64> {
        assert!(low < high, "low must be less than high");
        
        let range: u64 = (high - low) as u64;
        let data: Vec<i64> = (0..shape.size())
            .map(|_| low + (self.next_u64() % range) as i64)
            .collect();
        
        NdArray::from_vec(shape, data)
    }

    pub fn uniform(&mut self, low: f64, high: f64, shape: Shape) -> NdArray<f64>{
        assert!(low < high, "low must be less than high");

        let range: f64 = high - low;
        let data: Vec<f64> = (0..shape.size())
            .map(|_| low + (self.next_f64() * range))
            .collect();

        NdArray::from_vec(shape, data)
    }

    pub fn standard_uniform(&mut self, shape: Shape) -> NdArray<f64> {
        let data: Vec<f64> = (0..shape.size())
            .map(|_| self.next_f64())
            .collect();
        
        NdArray::from_vec(shape, data)
    }
 
    pub fn standard_normal(&mut self, shape: Shape) -> NdArray<f64> {
        let pi= std::f64::consts::PI;
        let mut data= Vec::with_capacity(shape.size());

        for _ in 0..(shape.size() / 2){
            let mut rand1 = self.next_f64();
            while rand1 == 0.0 {
                rand1 = self.next_f64();
            }
            let mut rand2 = self.next_f64();
            while rand2 == 0.0 {
                rand2 = self.next_f64();
            }

            let r = f64::sqrt(-2.0 * f64::ln(rand1));
            let z0 = r * f64::sin(2.0 * pi * rand2);
            let z1 = r * f64::cos(2.0 * pi * rand2);
            data.push(z0);
            data.push(z1);
        }
        if shape.size() % 2 == 1 {
            let rand1 = self.next_f64();
            let rand2 = self.next_f64();
            let z0 = f64::sqrt(-2.0 * f64::ln(rand1)) * f64::sin(2.0 * pi * rand2);
            data.push(z0);
        }

        NdArray::from_vec(shape, data)
    }

    fn sample_standard_normal_single(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn normal(&mut self, mu: f64, sigma: f64, shape: Shape) -> NdArray<f64> {
        let z = self.standard_normal(shape);
        mu + sigma * z
    }

    pub fn lognormal(&mut self, mu: f64, sigma: f64, shape: Shape) -> NdArray<f64> {
        let z = self.normal(mu, sigma, shape);
        z.exp()
    }

    fn sample_gamma_single(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            return self.sample_gamma_single(alpha + 1.0) * self.next_f64().powf(1.0 / alpha);
        }
        
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        
        loop {
            let mut x;
            let mut v;

            loop {
                x = self.sample_standard_normal_single();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            
            v = v * v * v;
            let u = self.next_f64();

            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v;
            }
            
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: Shape) -> NdArray<f64> {
        assert!(shape_param > 0.0, "shape parameter must be positive");
        assert!(scale > 0.0, "scale must be positive");

        let data: Vec<f64> = (0..shape.size())
            .map(|_| self.sample_gamma_single(shape_param) * scale)
            .collect();

        NdArray::from_vec(shape, data)
    }

    pub fn beta(&mut self, alpha: f64, beta_param: f64, shape: Shape) -> NdArray<f64> {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(beta_param > 0.0, "beta must be positive");

        let data: Vec<f64> = (0..shape.size())
            .map(|_| {
                let x = self.sample_gamma_single(alpha);
                let y = self.sample_gamma_single(beta_param);
                x / (x + y)
            })
            .collect();

        NdArray::from_vec(shape, data)
    }
}
