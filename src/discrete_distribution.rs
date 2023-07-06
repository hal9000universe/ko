//! Discrete Probability distributions.
//!
//! This module contains the `DiscreteProbabilityDistribution` struct, which
//! represents a discrete probability distribution. It is parameterized by the
//! type of the outcomes. The outcomes must be `Eq` and `Clone`, and the
//! probabilities must be non-negative and sum to 1.
//!
//! The `DiscreteProbabilityDistribution` struct has the following methods:
//!
//! - `new(outcomes: Vec<T>, probabilities: Vec<f64>) -> Self`: creates a new
//!  `DiscreteProbabilityDistribution` from a vector of outcomes and a vector
//! of probabilities.
//!
//! - `probabilities(&self) -> Vec<f64>`: returns the probabilities of the
//! outcomes.
//!
//! - `outcomes(&self) -> Vec<T>`: returns the outcomes.
//!
//! - `sample(&self) -> T`: returns a random outcome.
//!
//! - `pmf(&self, x: T) -> f64`: returns the probability mass function of the
//! outcome `x`.
//!
//! - `measure(&self, domain: &[T]) -> f64`: returns the measure of the
//! distribution over the set `domain`.
//!
//! - `multinomial(probabilities: Vec<f64>) -> Self`: creates a new
//! `DiscreteProbabilityDistribution` from a vector of probabilities. The
//! outcomes are the integers from 0 to `probabilities.len() - 1`.
//!
//! - 'binomial(n: usize, p: f64) -> Self': creates a new
//! `DiscreteProbabilityDistribution` from a number of trials `n` and a
//! probability of success `p`. The outcomes are the integers from 0 to `n`.
//!
//! # Examples
//!
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//!
//! // define outcomes
//! let outcomes = vec!["a", "b", "c"];
//!
//! // define probabilities
//! let probabilities = vec![0.1, 0.2, 0.7];
//!
//! // create discrete probability distribution
//! let d = DiscreteProbabilityDistribution::new(outcomes, probabilities);
//!
//! // check probabilities
//! assert_eq!(d.pmf(&"a"), 0.1);
//! assert_eq!(d.pmf(&"b"), 0.2);
//! assert_eq!(d.pmf(&"c"), 0.7);
//! assert_eq!(d.pmf(&"d"), 0.);
//! ```

use rand::{rngs::ThreadRng, Rng};
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct DiscreteProbabilityDistribution<T> {
    pub outcomes: Vec<T>,
    pub probabilities: Vec<f64>,
}

impl<T> DiscreteProbabilityDistribution<T> {
    pub fn new(outcomes: Vec<T>, probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! outcomes and a vector of probabilities.
        //!
        //! # Panics
        //!
        //! Panics if the lengths of the vectors are not equal, if the
        //! probabilities are not non-negative, or if the probabilities do not
        //! sum to 1.
        assert_eq!(
            outcomes.len(),
            probabilities.len(),
            "outcomes and probabilities must have the same length"
        );
        assert!(
            probabilities.iter().all(|&p| p >= -1e-10),
            "probabilities must be non-negative"
        );
        assert!(
            (probabilities.iter().sum::<f64>() - 1.).abs() < 1e-10,
            "probabilities must sum to 1"
        );
        Self {
            outcomes,
            probabilities,
        }
    }

    pub fn probabilities(&self) -> Vec<f64> {
        //! Returns a clone of the probabilities of the outcomes.
        self.probabilities.clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Clone,
{
    pub fn outcomes(&self) -> Vec<T> {
        //! Returns a clone of the outcomes.
        self.outcomes.clone()
    }

    pub fn sample(&self) -> T {
        //! Returns a random outcome.
        let mut rng: ThreadRng = rand::thread_rng();
        let mut u: f64 = rng.gen::<f64>();
        let mut i: usize = 0;
        while u > 0. {
            u -= self.probabilities[i];
            i += 1;
        }
        self.outcomes[i - 1].clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Eq,
{
    pub fn pmf(&self, x: &T) -> f64 {
        //! Returns the probability mass function of the outcome `x`.
        match self.outcomes.iter().position(|y| y == x) {
            Some(i) => self.probabilities[i],
            None => 0.,
        }
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Eq + Hash,
{
    pub fn measure(&self, domain: &Vec<T>) -> f64 {
        //! Returns the measure of the probability mass function over the set `domain`.
        assert!(
            domain.len()
                == domain
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
        );
        domain.iter().map(|x| self.pmf(x)).sum()
    }
}

fn factorial(n: i32) -> i32 {
    //! Returns the factorial of `n`.
    if n == 0 || n == 1 {
        1
    } else {
        (1..n + 1).fold(1, |acc, x| acc * x)
    }
}

fn binomial_coeff(n: i32, k: i32) -> i32 {
    //! Returns the binomial coefficient `n` over `k`.
    factorial(n) / (factorial(k) * factorial(n - k))
}

impl DiscreteProbabilityDistribution<i32> {
    pub fn multinomial(probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! probabilities. The outcomes are the integers from 0 to
        //! `probabilities.len() - 1`.
        let outcomes: Vec<i32> = (0..probabilities.len() as i32).collect();
        Self::new(outcomes, probabilities)
    }

    pub fn binomial(n: i32, p: f64) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from the parameters
        //! of a binomial distribution.
        //!
        //! Panics
        //!
        //! Panics if `p` is not in the interval [0, 1], if `n` is not positive, or if 'n' is larger than 12.
        assert!(p >= 0. && p <= 1., "p must be in the interval [0, 1]");
        assert!(n > 0, "n must be positive");
        assert!(n <= 12, "n must be smaller than 13");
        let outcomes: Vec<i32> = (0..=n + 1).collect();
        let probabilities: Vec<f64> = outcomes
            .iter()
            .map(|k| binomial_coeff(n, *k) as f64 * p.powi(*k) * (1. - p).powi(n - *k))
            .collect();
        Self::new(outcomes, probabilities)
    }
}

pub fn discrete_distribution_metric<T>(
    dist_x: &DiscreteProbabilityDistribution<T>,
    dist_y: &DiscreteProbabilityDistribution<T>,
) -> f64
where
    T: Eq + Hash + Copy,
{
    //! Returns the metric between two discrete probability distributions.
    //!
    //! # Examples
    //!
    //! ```
    //! use ko::discrete_distribution::DiscreteProbabilityDistribution;
    //!
    //! // define outcomes
    //! let outcomes = vec!["a", "b", "c"];
    //!
    //! // define probabilities
    //! let probabilities = vec![0.1, 0.2, 0.7];
    //!
    //! // create discrete probability distribution
    //! let d = DiscreteProbabilityDistribution::new(outcomes, probabilities);
    //!
    //! // check probabilities
    //! assert_eq!(d.pmf(&"a"), 0.1);
    //! assert_eq!(d.pmf(&"b"), 0.2);
    //! assert_eq!(d.pmf(&"c"), 0.7);
    //! assert_eq!(d.pmf(&"d"), 0.);
    //! ```

    // define domain
    let mut domain: Vec<T> = dist_x.outcomes();
    domain.append(&mut dist_y.outcomes());
    // remove duplicates
    let domain: Vec<T> = domain
        .iter()
        .collect::<std::collections::HashSet<&T>>()
        .iter()
        .map(|&&x| x)
        .collect::<Vec<T>>();
    // calculate metric
    let mut metric: f64 = 0.;
    for x in domain {
        metric += (dist_x.pmf(&x) - dist_y.pmf(&x)).powi(2);
    }
    metric.sqrt()
}

impl DiscreteProbabilityDistribution<i32> {
    pub fn construct(samples: &Vec<i32>) -> DiscreteProbabilityDistribution<i32> {
        //! Constructs a discrete probability distribution from a vector of samples.
        //! The outcomes are the unique values in the vector of samples.
        //! The probabilities are the relative frequencies of the outcomes.
        //! The outcomes are sorted in ascending order.
        
        // define outcomes
        let mut outcomes: Vec<i32> = samples
            .iter()
            .collect::<std::collections::HashSet<&i32>>()
            .iter()
            .map(|&&x| x)
            .collect::<Vec<i32>>();
        outcomes.sort();
        // define probabilities
        let probabilities: Vec<f64> = outcomes
            .iter()
            .map(|&x| samples.iter().filter(|&&y| y == x).count() as f64 / samples.len() as f64)
            .collect();
        // create discrete probability distribution
        DiscreteProbabilityDistribution::new(outcomes, probabilities)
    }

    pub fn construct_binomial(samples: &Vec<i32>) -> DiscreteProbabilityDistribution<i32> {
        //! Constructs a binomial distribution from a vector of samples.
        let n: i32 = samples.len() as i32;
        let mean: f64 = samples.iter().sum::<i32>() as f64 / n as f64;
        let p: f64 = mean / n as f64;
        DiscreteProbabilityDistribution::binomial(n, p)
    }
}
