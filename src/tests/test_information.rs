#[cfg(test)]
use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;
#[cfg(test)]
use crate::probability::information::{
    discrete_entropy, discrete_jensen_shannon_divergence, discrete_kullback_leibler_divergence,
};

#[test]
fn test_discrete_entropy() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let entropy: f64 = discrete_entropy(&multinomial).to_float();
    assert!(entropy > 0.);
    assert!((entropy - 1.).abs() < tolerance);
}

#[test]
fn test_discrete_kullback_leibler_divergence() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let divergence_value: f64 =
        discrete_kullback_leibler_divergence(&multinomial, &multinomial).to_float();
    assert!(divergence_value >= 0.);
    assert!((divergence_value - 0.).abs() < tolerance);
}

#[test]
fn test_discrete_jensen_shannon_divergence() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let divergence_value: f64 =
        discrete_jensen_shannon_divergence(&multinomial, &multinomial).to_float();
    assert!(divergence_value >= 0.);
    assert!((divergence_value - 0.).abs() < tolerance);
}
