#[cfg(test)]
use crate::distribution::DiscreteProbabilityDistribution;

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
    let outcomes: Vec<i32> = vec![1, 2, 3];
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.25];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::new(outcomes, probabilities);
    assert_eq!(dist.pmf(1), 0.5);
    assert_eq!(dist.pmf(2), 0.25);
    assert_eq!(dist.pmf(3), 0.25);
    assert_eq!(dist.pmf(4), 0.);
}
