//! This module contains functions for calculating information theoretic quantities.
//!
//! # Example
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//! use ko::information::{entropy, mutual_information, joint_entropy, InformationUnit};
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
//! // compute mutual information
//! let mutual_information: InformationUnit = mutual_information(&dist_x, &dist_y, &joint_distribution!(dist_x, dist_y));
//!
//! // check outcomes and probabilities
//! assert_eq!(entropy_x.to_float(), 1.);
//! assert_eq!(entropy_y.to_float(), 1.);
//! assert_eq!(joint_entropy.to_float(), 2.);
//! assert_eq!(mutual_information.to_float(), 0.);
//! ```

use crate::cartesian_product;
use crate::discrete_distribution::DiscreteProbabilityDistribution;
use crate::joint_distribution;

#[derive(Debug)]
pub enum InformationUnit {
    Bit(f64),
    Nat(f64),
}

impl InformationUnit {
    pub fn to_bits(&self) -> InformationUnit {
        match self {
            InformationUnit::Bit(x) => InformationUnit::Bit(*x),
            InformationUnit::Nat(x) => InformationUnit::Bit(x / 1.4426950408889634),
        }
    }

    pub fn to_nats(&self) -> InformationUnit {
        match self {
            InformationUnit::Bit(x) => InformationUnit::Nat(x * 1.4426950408889634),
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

pub fn entropy<T>(dist: &DiscreteProbabilityDistribution<T>) -> InformationUnit {
    //! returns the entropy of a discrete probability distribution in bits
    InformationUnit::Bit(
        -dist
            .probabilities()
            .into_iter()
            .fold(0., |sum, p| sum + p * p.log2()),
    )
}

pub fn mutual_information<T>(
    dist_x: &DiscreteProbabilityDistribution<T>,
    dist_y: &DiscreteProbabilityDistribution<T>,
    joint_dist: &DiscreteProbabilityDistribution<Vec<T>>,
) -> InformationUnit
where
    T: Copy,
{
    //! returns the mutual information of two discrete probability distributions in bits
    InformationUnit::Bit(
        entropy(dist_x).to_float() + entropy(dist_y).to_float() - entropy(&joint_dist).to_float(),
    )
}

pub fn joint_entropy(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> InformationUnit {
    //! returns the joint entropy of two discrete probability distributions in bits
    entropy(&joint_distribution!(dist_x, dist_y))
}
