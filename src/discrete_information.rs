//! This module contains functions for calculating information theoretic quantities.
//!
//! # Example
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//! use ko::discrete_information::{entropy, InformationUnit};
//!
//! // create two distributions
//! let dist_x: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![1, 2], vec![0.5, 0.5]);
//! let dist_y: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![3, 6], vec![0.5, 0.5]);
//!
//! // compute entropy
//! let entropy_x: InformationUnit = entropy(&dist_x);
//! let entropy_y: InformationUnit = entropy(&dist_y);
//!
//!
//! // check outcomes and probabilities
//! assert_eq!(entropy_x.to_float(), 1.);
//! assert_eq!(entropy_y.to_float(), 1.);
//! ```

use crate::discrete_distribution::DiscreteProbabilityDistribution;

use std::{
    f64::consts::E,
    ops::{Add, Sub},
};

#[derive(Debug, Copy, Clone)]
pub enum InformationUnit {
    Bit(f64),
    Nat(f64),
}

impl InformationUnit {
    pub fn to_bits(&self) -> InformationUnit {
        match self {
            InformationUnit::Bit(x) => InformationUnit::Bit(*x),
            InformationUnit::Nat(x) => InformationUnit::Bit(x * E.log2()),
        }
    }

    pub fn to_nats(&self) -> InformationUnit {
        match self {
            InformationUnit::Bit(x) => InformationUnit::Nat(x * 2f64.ln()),
            InformationUnit::Nat(x) => InformationUnit::Nat(*x),
        }
    }

    pub fn to_float(&self) -> f64 {
        match self {
            InformationUnit::Bit(x) => *x,
            InformationUnit::Nat(x) => *x,
        }
    }

    pub fn apply(&self, func: impl Fn(f64) -> f64) -> InformationUnit {
        match self {
            InformationUnit::Bit(x) => InformationUnit::Bit(func(*x)),
            InformationUnit::Nat(x) => InformationUnit::Nat(func(*x)),
        }
    }
}

impl Add for InformationUnit {
    type Output = InformationUnit;

    fn add(self, other: InformationUnit) -> InformationUnit {
        match (self, other) {
            (InformationUnit::Bit(x), InformationUnit::Bit(y)) => InformationUnit::Bit(x + y),
            (InformationUnit::Nat(x), InformationUnit::Nat(y)) => InformationUnit::Nat(x + y),
            (InformationUnit::Bit(x), InformationUnit::Nat(y)) => {
                InformationUnit::Bit(x + y * E.log2())
            }
            (InformationUnit::Nat(x), InformationUnit::Bit(y)) => {
                InformationUnit::Bit(x * E.log2() + y)
            }
        }
    }
}

impl Sub for InformationUnit {
    type Output = InformationUnit;

    fn sub(self, other: InformationUnit) -> InformationUnit {
        match (self, other) {
            (InformationUnit::Bit(x), InformationUnit::Bit(y)) => InformationUnit::Bit(x - y),
            (InformationUnit::Nat(x), InformationUnit::Nat(y)) => InformationUnit::Nat(x - y),
            (InformationUnit::Bit(x), InformationUnit::Nat(y)) => {
                InformationUnit::Bit(x - y / 2f64.ln())
            }
            (InformationUnit::Nat(x), InformationUnit::Bit(y)) => {
                InformationUnit::Bit(x / 2f64.ln() - y)
            }
        }
    }
}

pub fn entropy<T>(dist: &DiscreteProbabilityDistribution<T>) -> InformationUnit {
    //! returns the entropy of a discrete probability distribution in bits
    InformationUnit::Bit(
        -dist
            .probabilities()
            .into_iter()
            .fold(0., |sum, p| sum + p * p.log2()),
    )
}

pub fn kullback_leibler_divergence(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> InformationUnit {
    //! returns the Kullback-Leibler divergence of two discrete probability distributions in bits
    let mut outcomes: Vec<i32> = dist_x.outcomes();
    outcomes.append(&mut dist_y.outcomes());
    InformationUnit::Bit(
        outcomes
            .iter()
            .map(|&x| x)
            .collect::<std::collections::HashSet<i32>>()
            .iter()
            .map(|x| {
                let p_x: f64 = dist_x.pmf(x);
                let p_y: f64 = dist_y.pmf(x);
                p_x * (p_x / p_y).log2()
            })
            .sum(),
    )
}

fn average_discrete_distributions(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> DiscreteProbabilityDistribution<i32> {
    let mut outcomes: Vec<i32> = dist_x.outcomes();
    outcomes.append(&mut dist_y.outcomes());
    let outcomes: Vec<i32> = outcomes
        .iter()
        .map(|&x| x)
        .collect::<std::collections::HashSet<i32>>()
        .iter()
        .map(|&x| x)
        .collect();
    let probabilities: Vec<f64> = outcomes
        .iter()
        .map(|x| (dist_x.pmf(x) + dist_y.pmf(x)) / 2.)
        .collect();
    DiscreteProbabilityDistribution::new(outcomes, probabilities)
}

pub fn jensen_shannon_divergence(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> InformationUnit {
    //! returns the Jensen-Shannon divergence of two discrete probability distributions in bits
    let m: DiscreteProbabilityDistribution<i32> = average_discrete_distributions(dist_x, dist_y);
    (kullback_leibler_divergence(dist_x, &m) + kullback_leibler_divergence(dist_y, &m))
        .apply(|x| x / 2.)
}
