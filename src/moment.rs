//! # Moment
//!
//! This module contains functions for calculating moments of discrete probability distributions.
//!
//! # Example
//!
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//!
//! // create a distribution
//! let dist: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![1, 2, 3], vec![0.5, 0.25, 0.25]);
//!
//! // compute the first moment
//! let moment1: f64 = ko::moment::moment(1, &dist);
//!
//! // compute the second moment
//! let moment2: f64 = ko::moment::moment(2, &dist);
//!
//! // compute the third central moment
//! let central_moment3: f64 = ko::moment::central_moment(3, &dist);
//! ```

use crate::discrete_distribution::DiscreteProbabilityDistribution;

pub fn moment<T>(n: i32, dist: &DiscreteProbabilityDistribution<T>) -> f64
where
    T: Copy + Into<f64>,
{
    //! returns the nth moment of a discrete probability distribution
    dist.outcomes
        .iter()
        .zip(dist.probabilities.iter())
        .fold(0., |sum, (x, p)| sum + (*x).into().powi(n) * p)
}

pub fn central_moment<T>(n: i32, dist: &DiscreteProbabilityDistribution<T>) -> f64
where
    T: Copy + Into<f64>,
{
    //! returns the nth central moment of a discrete probability distribution
    let mean: f64 = moment(1, &dist);
    dist.outcomes
        .iter()
        .zip(dist.probabilities.iter())
        .fold(0., |sum, (x, p)| sum + ((*x).into() - mean).powi(n) * p)
}
