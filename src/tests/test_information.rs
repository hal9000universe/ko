#[cfg(test)]
use crate::discrete_distribution::DiscreteProbabilityDistribution;
#[cfg(test)]
use crate::information::{entropy, joint_entropy, mutual_information, InformationUnit};
#[cfg(test)]
use crate::{cartesian_product, joint_distribution};

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
fn test_mutual_information() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let joint_distribution: DiscreteProbabilityDistribution<Vec<i32>> =
        joint_distribution!(multinomial, multinomial);
    assert!(
        (mutual_information(&multinomial, &multinomial, &joint_distribution).to_float() - 0.).abs()
            < tolerance
    );
}

#[test]
fn test_joint_entropy() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let joint_entropy: f64 = joint_entropy(&multinomial, &multinomial).to_float();
    assert!(joint_entropy > 0.);
    assert!(
        (joint_entropy - entropy(&multinomial).to_float() - entropy(&multinomial).to_float()).abs()
            < tolerance
    );
}
