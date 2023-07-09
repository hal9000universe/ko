//! # Binomial Distinction Test
//! 
//! The binomial distinction test is a statistical test for distinguishing between two binomial distributions.
//! 
//! ## Example
//! 
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//! use ko::sample::discrete_sample;
//! use ko::binomial_testing::validate_binomial;
//! 
//! // Construct the test & sample distributions.
//! let test_p: f64 = 0.5;
//! let sample_p: f64 = 0.75;
//! let test_dist: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(test_p);
//! let sample_dist: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(sample_p);
//! 
//! // Sample from the sample distribution.
//! let n_samples: usize = 100;
//! let samples: Vec<i32> = discrete_sample(n_samples, &sample_dist);
//! 
//! // Run the test.
//! let distinction: bool = validate_binomial(&test_dist, &samples);
//! println!("Binomial distinction test: {}", distinction);
//! ```

use crate::discrete_distribution::DiscreteProbabilityDistribution;
use crate::sample::discrete_sample;


// const BETA: f64 = 0.95;
const Z: f64 = 1.96;


pub fn estimate_binomial(samples: &Vec<i32>) -> DiscreteProbabilityDistribution<i32> {
    //! Construct a binomial distribution from a sample.
    let p: f64 = samples.iter().sum::<i32>() as f64 / samples.len() as f64;
    DiscreteProbabilityDistribution::binomial(p)
}


pub fn validate_binomial(test_dist: &DiscreteProbabilityDistribution<i32>, samples: &Vec<i32>) -> bool {
    //! Run the binomial distinction test.
    let binom_dist: DiscreteProbabilityDistribution<i32> = estimate_binomial(samples);
    let h: f64 = binom_dist.probabilities()[1];  // relative frequency of 1
    let n: f64 = samples.len() as f64;
    let min_p: f64 = (2. * h * n + Z.powi(2) - (Z.powi(4) + 4. * h * n * Z.powi(2) - 4. * h.powi(2) * n * Z.powi(2)).sqrt()) / (2. * (n + Z.powi(2)));
    let max_p: f64 = (2. * h * n + Z.powi(2) + (Z.powi(4) + 4. * h * n * Z.powi(2) - 4. * h.powi(2) * n * Z.powi(2)).sqrt()) / (2. * (n + Z.powi(2)));
    min_p <= test_dist.probabilities()[1] && test_dist.probabilities()[1] <= max_p
}

pub fn run_binomial_distinction() {
    //! Run the binomial distinction test.
    
    // Construct the test & sample distributions.
    let test_p: f64 = 0.5;
    let sample_p: f64 = 0.75;
    let test_dist: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(test_p);
    let sample_dist: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(sample_p);

    let max_n: usize = 1000;
    for n_samples in 10..max_n {
        // Sample from the sample distribution.
        let samples: Vec<i32> = discrete_sample(n_samples, &sample_dist);

        // Run the test.
        let distinction: bool = validate_binomial(&test_dist, &samples);
        println!("n: {} - Binomial distinction test: {}", n_samples, distinction); 
    }
}