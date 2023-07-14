#[cfg(test)]
use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;
#[cfg(test)]
use crate::probability::discrete_moment::{discrete_central_moment, discrete_moment};

#[test]
fn test_moments() {
    let tolerance: f64 = 1e-10;
    let p: f64 = 0.5;
    let binomial_distribution: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::binomial(p);
    assert!((discrete_moment(1, &binomial_distribution) - p).abs() < tolerance);
    assert!((discrete_central_moment(1, &binomial_distribution) - 0.).abs() < tolerance);
    assert!((discrete_central_moment(2, &binomial_distribution) - p * (1. - p)).abs() < tolerance);
}
