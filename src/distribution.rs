//! Probability distributions.
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
//! - `pmf(&self, x: T) -> f64`: returns the probability mass function of the
//! outcome `x`.
//! 
//! - `multinomial(probabilities: Vec<f64>) -> Self`: creates a new
//! `DiscreteProbabilityDistribution` from a vector of probabilities. The
//! outcomes are the integers from 0 to `probabilities.len() - 1`.
//! 
//! # Examples
//! 
//! ```
//! use ko::distribution::DiscreteProbabilityDistribution;
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
//! assert_eq!(d.pmf("a"), 0.1);
//! assert_eq!(d.pmf("b"), 0.2);
//! assert_eq!(d.pmf("c"), 0.7);
//! assert_eq!(d.pmf("d"), 0.);
//! ```
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
            probabilities.iter().all(|&p| p >= 0.),
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
        //! Returns the probabilities of the outcomes.
        self.probabilities.clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Clone,
{
    pub fn outcomes(&self) -> Vec<T> {
        //! Returns the outcomes.
        self.outcomes.clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Eq,
{
    pub fn pmf(&self, x: T) -> f64 {
        //! Returns the probability mass function of the outcome `x`.
        match self.outcomes.iter().position(|y| y == &x) {
            Some(i) => self.probabilities[i],
            None => 0.,
        }
    }
}

impl DiscreteProbabilityDistribution<i32> {
    pub fn multinomial(probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! probabilities. The outcomes are the integers from 0 to
        //! `probabilities.len() - 1`.
        let outcomes: Vec<i32> = (0..probabilities.len() as i32).collect();
        Self::new(outcomes, probabilities)
    }
}
