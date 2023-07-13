use crate::probability::continuous_distribution::ContinuousProbabilityDistribution;
use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;

pub fn discrete_sample(n: usize, dist: &DiscreteProbabilityDistribution<i32>) -> Vec<i32> {
    //! Samples `n` times from a `DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Arguments:
    //! * `n`: `usize`, number of samples
    //! * `dist`: `&DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Returns:
    //! * `samples`: `Vec<i32>`
    let mut samples: Vec<i32> = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(dist.sample());
    }
    samples
}

pub fn continuous_sample(n: usize, dist: &impl ContinuousProbabilityDistribution) -> Vec<f64> {
    //! Samples `n` times from a `impl ContinuousProbabilityDistribution`
    //!
    //! ## Arguments:
    //! * `n`: `usize`, number of samples
    //! * `dist`: `&impl ContinuousProbabilityDistribution`
    //!
    //! ## Returns:
    //! * `samples`: `Vec<f64>`
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(dist.sample());
    }
    samples
}
