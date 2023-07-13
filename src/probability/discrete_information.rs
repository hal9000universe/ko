use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;

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
        //! Converts `self` to bits
        match self {
            InformationUnit::Bit(x) => InformationUnit::Bit(*x),
            InformationUnit::Nat(x) => InformationUnit::Bit(x * E.log2()),
        }
    }

    pub fn to_nats(&self) -> InformationUnit {
        //! Converts `self` to nats
        match self {
            InformationUnit::Bit(x) => InformationUnit::Nat(x * 2f64.ln()),
            InformationUnit::Nat(x) => InformationUnit::Nat(*x),
        }
    }

    pub fn to_float(&self) -> f64 {
        //! Returns the value assigned to `self` as f64
        match self {
            InformationUnit::Bit(x) => *x,
            InformationUnit::Nat(x) => *x,
        }
    }

    pub fn apply(&self, func: impl Fn(f64) -> f64) -> InformationUnit {
        //! Applies a transformation to the value assigned to `self`
        //!
        //! ## Arguments:
        //! * `func`: `impl Fn(f64) -> f64`
        //!
        //! ## Returns:
        //! * an `InformationUnit` the value of which is the transformed value of `self`
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
    //! Computes the entropy of a `DiscreteProbabilityDistribution` in bits
    //!
    //! ## Arguments:
    //! * `dist`: `&DiscreteProbabilityDistribution<T>`
    //!
    //! ## Returns:
    //! * `InformationUnit` corresponding to the entropy of the given `DiscreteProbabilityDistribution`
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
    //! Computes the Kullback-Leibler Divergence of two `DiscreteProbabilityDistribution`s in bits
    //!
    //! ## Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution<i32`
    //! * `dist_y`: `&DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Returns:
    //! * `InformationUnit` corresponding to the Kullback-Leibler Divergence of the given
    //! `DiscreteProbabilityDistribution`s
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
    //! Averages two `DiscreteProbabilityDistribution`s
    //!
    //! ## Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution<i32>`
    //! * `dist_y`: `&DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Returns:
    //! * `DiscreteProbabilityDistribution` corresponding to the average distribution of the given two
    //! `DiscreteProbabilityDistribution`s
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
    //! Computes the Jensen-Shannon Divergence of two `DiscreteProbabilityDistribution`s in bits
    //!
    //! Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution`
    //! * `dist_y`: `&DiscreteProbabilityDistribution`
    //!
    //! Returns:
    //! * `InformationUnit` corresponding to the Jensen-Shannon Divergence of the given
    //! `DiscreteProbabilityDistribution`s
    let m: DiscreteProbabilityDistribution<i32> = average_discrete_distributions(dist_x, dist_y);
    (kullback_leibler_divergence(dist_x, &m) + kullback_leibler_divergence(dist_y, &m))
        .apply(|x| x / 2.)
}
