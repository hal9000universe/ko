//! This module contains functions for calculating information theoretic quantities.
//!
//! # Example
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//! use ko::information::{entropy, joint_entropy, InformationUnit};
//! use ko::{joint_distribution, cartesian_product};
//!
//! // create two distributions
//! let dist_x: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![1, 2], vec![0.5, 0.5]);
//! let dist_y: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![3, 6], vec![0.5, 0.5]);
//!
//! // compute entropy
//! let entropy_x: InformationUnit = entropy(&dist_x);
//! let entropy_y: InformationUnit = entropy(&dist_y);
//!
//! // compute joint entropy
//! let joint_entropy: InformationUnit = joint_entropy(&dist_x, &dist_y);
//!
//!
//! // check outcomes and probabilities
//! assert_eq!(entropy_x.to_float(), 1.);
//! assert_eq!(entropy_y.to_float(), 1.);
//! assert_eq!(joint_entropy.to_float(), 2.);
//! ```

use crate::cartesian_product;
use crate::discrete_distribution::DiscreteProbabilityDistribution;
use crate::joint_distribution;

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

pub fn joint_entropy(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> InformationUnit {
    //! returns the joint entropy of two discrete probability distributions in bits
    entropy(&joint_distribution!(dist_x, dist_y))
}
