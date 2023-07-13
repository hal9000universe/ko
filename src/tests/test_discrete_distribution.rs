#[cfg(test)]
use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;

#[test]
#[should_panic]
fn test_incompat_out_prob_dist() {
    let outcomes: Vec<i32> = vec![1, 2, 3];
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.125, 0.125];
    DiscreteProbabilityDistribution::new(outcomes, probabilities);
}

#[test]
#[should_panic]
fn test_ill_def_prob_dist() {
    let outcomes: Vec<i32> = vec![1, 2, 3];
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.125];
    DiscreteProbabilityDistribution::new(outcomes, probabilities);
}

#[test]
fn test_pmf() {
    let tolerance: f64 = 1e-10;
    let outcomes: Vec<i32> = vec![1, 2, 3];
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.25];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::new(outcomes, probabilities);
    assert!((dist.pmf(&1) - 0.5).abs() < tolerance);
    assert!((dist.pmf(&2) - 0.25).abs() < tolerance);
    assert!((dist.pmf(&3) - 0.25).abs() < tolerance);
    assert!((dist.pmf(&4) - 0.).abs() < tolerance);
}

#[test]
fn test_discrete_measure() {
    let tolerance: f64 = 1e-10;
    let binomial_distribution: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.25, 0.125, 0.125]);
    assert!(
        (binomial_distribution.measure(&binomial_distribution.outcomes) - 1.).abs() < tolerance
    );
    assert!((binomial_distribution.measure(&vec![0, 1]) - 0.75).abs() < tolerance);
    assert!((binomial_distribution.measure(&vec![0, 1, 2]) - 0.875).abs() < tolerance);
}
