#[cfg(test)]
use crate::discrete_distribution::DiscreteProbabilityDistribution;

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
    let binomial_distribution: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(5, 0.5);
    assert!((binomial_distribution.measure(&binomial_distribution.outcomes) - 1.).abs() < tolerance);
    assert!((binomial_distribution.measure(&vec![0, 1, 2]) - 0.5).abs() < tolerance);
    assert!((binomial_distribution.measure(&vec![0, 1, 2, 3]) - 0.8125).abs() < tolerance);
}


#[test]
fn test_binomial_distributions() {
    let tolerance: f64 = 1e-10;
    let max_n: i32 = 12;
    let p: f64 = 0.5;
    for n in 1..max_n {
        let binomial_distribution: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(n, p);
        assert!((binomial_distribution.measure(&binomial_distribution.outcomes()) - 1.).abs() < tolerance);
    }
}
