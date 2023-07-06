#[cfg(test)] 
use crate::information::{entropy, mutual_information, joint_entropy};
#[cfg(test)]
use crate::discrete_distribution::{DiscreteProbabilityDistribution};
#[cfg(test)]
use crate::{joint_distribution, cartesian_product};

#[test]
fn test_mutual_information() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let joint_distribution: DiscreteProbabilityDistribution<Vec<i32>> = joint_distribution!(multinomial, multinomial);
    assert!((mutual_information(&multinomial, &multinomial, &joint_distribution).to_float() - 0.).abs() < tolerance);
}

#[test]
fn test_joint_entropy() {
    let tolerance: f64 = 1e-10;
    let multinomial: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let joint_entropy: f64 = joint_entropy(&multinomial, &multinomial).to_float();
    assert!(joint_entropy > 0.);
    assert!((joint_entropy - entropy(&multinomial).to_float() - entropy(&multinomial).to_float()).abs() < tolerance);
}