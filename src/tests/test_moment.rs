#[cfg(test)]
use crate::discrete_distribution::{DiscreteProbabilityDistribution};
#[cfg(test)]
use crate::moment::{moment, central_moment};


#[test]
fn test_moments() {
    let tolerance: f64 = 1e-10;
    let n: i32 = 5;
    let p: f64 = 0.5;
    let binomial_distribution: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(n, p);
    assert!((moment(1, &binomial_distribution) - n as f64 * p).abs() < tolerance);
    assert!((central_moment(1, &binomial_distribution) - 0.).abs() < tolerance);
    assert!((central_moment(2, &binomial_distribution) - n as f64 * p * (1. - p)).abs() < tolerance);
}