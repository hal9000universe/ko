//! Joint probability distributions.
//!
//! # Example
//!
//! ```
//! use ko::distribution::DiscreteProbabilityDistribution;
//! use ko::joint_distribution;
//! use ko::cartesian_product;
//!
//! // define probaility distribution
//! let dist: DiscreteProbabilityDistribution<i32> =
//!    DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
//!
//! // compute joint distribution
//! let joint_dist: DiscreteProbabilityDistribution<Vec<i32>> = joint_distribution!(dist, dist);
//!
//! // check outcomes and probabilities
//! assert_eq!(joint_dist.probabilities, vec![0.25, 0.25, 0.25, 0.25]);
//! assert_eq!(joint_dist.outcomes, vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]);
//! ```
#[macro_export]
macro_rules! joint_distribution {
    // joins n independent discrete probability distributions with integer outcomes
    ( $( $x:ident ),* ) => {
        {
            // assemble outcomes and probabilities
            let outcomes = vec![$($x.outcomes()),*];
            let probabilities = vec![$($x.probabilities()),*];
            // compute joint outcomes and probabilities
            let joint_outcomes = cartesian_product!(outcomes);
            let joint_probabilities = cartesian_product!(probabilities).into_iter().map(|x| { x.into_iter().fold(1., |prod, y| prod * y) }).collect();
            // return joint distribution
            DiscreteProbabilityDistribution::new(joint_outcomes, joint_probabilities)
        }
    };
}
