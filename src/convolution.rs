use crate::cartesian_product;
use crate::distribution::DiscreteProbabilityDistribution;

pub fn special_convolution(
    dist_x: &DiscreteProbabilityDistribution<f64>,
    dist_y: &DiscreteProbabilityDistribution<f64>,
) -> DiscreteProbabilityDistribution<f64> {
    //! computes a discrete convolution between two discrete probability distributions, the random variables of which are independent and real-valued.
    //!
    //! # Example
    //! ```
    //! use ko::distribution::DiscreteProbabilityDistribution;
    //! use ko::convolution::special_convolution;
    //!
    //! // create two distributions
    //! let dist1: DiscreteProbabilityDistribution<f64> = DiscreteProbabilityDistribution::new(vec![1., 2.], vec![0.5, 0.5]);
    //! let dist2: DiscreteProbabilityDistribution<f64> = DiscreteProbabilityDistribution::new(vec![3., 6.], vec![0.5, 0.5]);
    //!
    //! // compute convolution
    //! let conv_dist: DiscreteProbabilityDistribution<f64> = special_convolution(&dist1, &dist2);
    //!
    //! // check outcomes and probabilities
    //! assert_eq!(conv_dist.outcomes(), vec![4., 5., 7., 8.]);
    //! assert_eq!(conv_dist.probabilities(), vec![0.25, 0.25, 0.25, 0.25]);
    //! ```
    let dist_x_outcomes: Vec<f64> = dist_x.outcomes();
    let dist_y_outcomes: Vec<f64> = dist_y.outcomes();
    let prob_x: Vec<f64> = dist_x.probabilities();
    let prob_y: Vec<f64> = dist_y.probabilities();
    let sums: Vec<f64> = cartesian_product!(dist_x_outcomes, dist_y_outcomes)
        .into_iter()
        .map(|x| x.into_iter().fold(0., |sum, y| sum + y))
        .collect();
    let probabilities: Vec<f64> = cartesian_product!(prob_x, prob_y)
        .into_iter()
        .map(|x| x.into_iter().fold(1., |prod, y| prod * y))
        .collect();
    DiscreteProbabilityDistribution::new(sums, probabilities)
}

pub fn discrete_convolution(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> DiscreteProbabilityDistribution<i32> {
    //! computes a discrete convolution between two discrete probability distributions, the random variables of which are independent and integer-valued.
    //! The convolution is computed by summing the probabilities of the cartesian product of the outcomes of the two distributions.
    //!
    //! # Example
    //! ```
    //! use ko::distribution::DiscreteProbabilityDistribution;
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
    let min = dist_x.outcomes().iter().min().unwrap() + dist_y.outcomes().iter().min().unwrap();
    let max = dist_x.outcomes().iter().max().unwrap() + dist_y.outcomes().iter().max().unwrap();
    let outcomes: Vec<i32> = (min..max + 1).collect();
    let probabilities: Vec<f64> = outcomes
        .iter()
        .map(|&z| {
            dist_x
                .outcomes()
                .iter()
                .map(|&k| dist_x.pmf(k) * dist_y.pmf(z - k))
                .sum()
        })
        .collect();
    DiscreteProbabilityDistribution::new(outcomes, probabilities)
}
