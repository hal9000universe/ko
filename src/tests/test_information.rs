#[cfg(test)]
use crate::discrete_distribution::DiscreteProbabilityDistribution;
#[cfg(test)]
use crate::discrete_information::{entropy, kullback_leibler_divergence, jensen_shannon_divergence, InformationUnit};

#[test]
fn test_information_unit() {
    let bit: InformationUnit = InformationUnit::Bit(1.);
    let nat: InformationUnit = InformationUnit::Nat(1.);
    assert_eq!(bit.to_bits().to_float(), bit.to_float());
    assert_eq!(nat.to_nats().to_float(), nat.to_float());
    assert_eq!(bit.to_float() + bit.to_float(), (bit + bit).to_float());
    assert_eq!(nat.to_float() + nat.to_float(), (nat + nat).to_float());
    assert_eq!(bit.to_nats().to_bits().to_float(), bit.to_float());
    assert_eq!(nat.to_bits().to_nats().to_float(), nat.to_float());
    assert_eq!(nat.to_float(), 2f64.ln() * nat.to_bits().to_float());
}

#[test]
fn test_entropy() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let entropy: f64 = entropy(&multinomial).to_float();
    assert!(entropy > 0.);
    assert!((entropy - 1.).abs() < tolerance);
}

#[test]
fn test_kullback_leibler_divergence() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let divergence_value: f64 =
        kullback_leibler_divergence(&multinomial, &multinomial).to_float();
    assert!(divergence_value >= 0.);
    assert!((divergence_value - 0.).abs() < tolerance);
}

#[test]
fn test_jensen_shannon_divergence() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let divergence_value: f64 =
        jensen_shannon_divergence(&multinomial, &multinomial).to_float();
    assert!(divergence_value >= 0.);
    assert!((divergence_value - 0.).abs() < tolerance);
}
