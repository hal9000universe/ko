//! # Sample
//! 
//! This module contains functions for sampling from probability distributions.
//! 
//! ## Example Discrete Distribution
//! 
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//! use ko::sample::discrete_sample;
//! 
//! let probabilities: Vec<f64> = vec![0.5, 0.25, 0.125, 0.125];
//! let dist: DiscreteProbabilityDistribution<i32> =
//!    DiscreteProbabilityDistribution::multinomial(probabilities);
//! println!("Distribution: {:?}", dist);
//! 
//! // discrete sample
//! let disc_samples: Vec<i32> = discrete_sample(1000, &dist);
//! ```
//! 
//! ## Example Continuous Distribution
//! 
//! ```
//! use ko::continuous_distribution::NormalDistribution;
//! use ko::sample::continuous_sample;
//! 
//! // continuous probability distribution
//! let cont_dist: NormalDistribution = NormalDistribution::new(0., 1.);
//! println!("Continuous Distribution: {:?}", cont_dist);
//! 
//! // continuous sample
//! let cont_samples: Vec<f64> = continuous_sample(1000, &cont_dist);
//! println!("Continuous Sample: {:?}", cont_samples);

use crate::discrete_distribution::DiscreteProbabilityDistribution;
use crate::continuous_distribution::ContinuousProbabilityDistribution;

pub fn discrete_sample(n: usize, dist: &DiscreteProbabilityDistribution<i32>) -> Vec<i32> {
    let mut samples: Vec<i32> = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(dist.sample());
    }
    samples
}

pub fn continuous_sample(n: usize, dist: &impl ContinuousProbabilityDistribution) -> Vec<f64> {
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(dist.sample());
    }
    samples
}