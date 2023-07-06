//! # Convolution
//!
//! ## Discrete Convolution
//!
//! The discrete convolution is used for computing the convolution between two independent integer-valued random variables.
//!
//! # Example Discrete Convolution
//!
//! ```
//! use ko::discrete_distribution::DiscreteProbabilityDistribution;
//! use ko::convolution::{discrete_convolution};
//!
//! // create two distributions
//! let dist1: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![1, 2], vec![0.5, 0.5]);
//! let dist2: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::new(vec![3, 6], vec![0.5, 0.5]);
//!
//! // compute convolution
//! let conv_dist: DiscreteProbabilityDistribution<i32> = discrete_convolution(&dist1, &dist2);
//!
//! // check outcomes and probabilities
//! assert_eq!(conv_dist.outcomes(), vec![4, 5, 7, 8]);
//! assert_eq!(conv_dist.probabilities(), vec![0.25, 0.25, 0.25, 0.25]);
//! ```

use crate::discrete_distribution::DiscreteProbabilityDistribution;

pub fn discrete_convolution(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> DiscreteProbabilityDistribution<i32> {
    //! computes a discrete convolution between two discrete probability distributions, the random variables of which are independent and integer-valued.
    //! The convolution is computed by summing the probabilities of the cartesian product of the outcomes of the two distributions.
    //!
    //! # Example
    //! ```
    //! use ko::discrete_distribution::DiscreteProbabilityDistribution;
    //! use ko::convolution::discrete_convolution;
    //!
    //! // create multinomial distribution
    //! let probabilities: Vec<f64> = vec![0.5, 0.5];
    //! let dist: DiscreteProbabilityDistribution<i32> =
    //!    DiscreteProbabilityDistribution::multinomial(probabilities);
    //!
    //! // compute convolution
    //! let conv_dist: DiscreteProbabilityDistribution<i32> =
    //!   discrete_convolution(&dist, &dist);
    //!
    //! // check result
    //! assert_eq!(conv_dist.outcomes(), vec![0, 1, 2]);
    //! assert_eq!(conv_dist.probabilities(), vec![0.25, 0.5, 0.25]);
    //! ```
    let min: i32 = dist_x.outcomes().iter().min().unwrap() + dist_y.outcomes().iter().min().unwrap();
    let max: i32 = dist_x.outcomes().iter().max().unwrap() + dist_y.outcomes().iter().max().unwrap();
    let outcomes: Vec<i32> = (min..max + 1).collect();
    // compute probabilities
    let probabilities: Vec<f64> = outcomes
        .iter()
        .map(|&z| {
            dist_x
                .outcomes()
                .iter()
                .map(|&k| dist_x.pmf(&k) * dist_y.pmf(&(z - k)))
                .sum()
        })
        .collect();
    // filter out outcomes with zero probability
    let outcomes: Vec<i32> = outcomes
        .iter()
        .zip(probabilities.iter())
        .filter(|(_, &p)| p > 0.)
        .map(|(&z, _)| z)
        .collect();
    // filter out probabilities with zero probability
    let probabilities: Vec<f64> = probabilities
        .iter()
        .filter(|&p| p > &0.)
        .map(|&p| p)
        .collect();
    println!("{:?}", outcomes);
    println!("{:?}", probabilities);
    DiscreteProbabilityDistribution::new(outcomes, probabilities)
}
