//! This module contains functions for testing the accuracy of continuous probability distributions.
//! 
//! # Examples
//! 
//! ```
//! use ko::continuous_testing::validate_continuous_cdf;
//! use ko::continuous_distribution::NormalDistribution;
//! 
//! let samples: Vec<f64> = vec![0.1, 0.2, 0.1, 0.15, 0.3, 0.4, 0.2, 0.5];
//! let dist = NormalDistribution::new(0.0, 1.0);
//! validate_continuous_cdf(&dist, &samples);
//! ```

use crate::continuous_distribution::{ContinuousProbabilityDistribution};


const DELTA: f64 = 0.001;
const EPSILON: f64 = 0.05;


fn continuous_set_reduce(set: &Vec<f64>) -> Vec<f64> {
    //! returns a sorted, reduced set
    let mut reduced_set: Vec<f64> = set.clone();
    reduced_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    reduced_set.dedup();
    reduced_set.to_vec()
}

pub fn estimate_continuous_cdf(samples: &Vec<f64>) -> Vec<(f64, f64)> {
    //! returns a vector of (x, y) pairs, where x is a value in the sample set and y is the estimated CDF at x
    let n: usize = samples.len();
    let cdf_x: Vec<f64> = continuous_set_reduce(&samples);
    let cdf_y: Vec<f64> = cdf_x.clone()
        .into_iter().map(|x| samples.iter().filter(|&y| *y <= x).count() as f64 / n as f64)
        .collect::<Vec<f64>>();
    cdf_x.iter().zip(cdf_y.iter()).map(|(&x, &y)| (x, y)).collect()
}


pub fn est_cdf(est_dist: &Vec<(f64, f64)>, x: &f64) -> f64 {
    //! evaluates the estimated CDF at x
    est_dist.iter().filter(|(x_i, _)| x_i <= x).last().unwrap().1
}


pub fn validate_continuous_cdf(dist: &impl ContinuousProbabilityDistribution, samples: &Vec<f64>) -> bool {
    //! performs a Kolmogorov-Smirnov test
    let cdf: Vec<(f64, f64)> = estimate_continuous_cdf(samples);
    let max_x: f64 = cdf.iter().map(|(x, _)| *x).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_x: f64 = cdf.iter().map(|(x, _)| *x).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let xs: Vec<f64> = (0..((max_x - min_x) / DELTA).floor() as i32)
        .into_iter().map(|i| min_x + i as f64 * DELTA)
        .collect::<Vec<f64>>();

    let max_diff: f64 = xs.iter().map(|&x| (dist.cdf(x) - est_cdf(&cdf, &x)).abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    max_diff < EPSILON
}