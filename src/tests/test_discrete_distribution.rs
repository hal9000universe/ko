#[cfg(test)]
use crate::probability::discrete_distribution::{
    discrete_convolution, DiscreteProbabilityDistribution,
};

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

#[test]
fn test_discrete_convolution() {
    let tolerance: f64 = 1e-10;
    let probabilities: Vec<f64> = vec![0.5, 0.5];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(probabilities);
    let conv_dist: DiscreteProbabilityDistribution<i32> = discrete_convolution(&dist, &dist);
    assert_eq!(conv_dist.outcomes(), vec![0, 1, 2]);
    for (x, y) in conv_dist
        .probabilities()
        .iter()
        .zip(vec![0.25, 0.5, 0.25].iter())
    {
        assert!((x - y).abs() < tolerance);
    }
}

#[test]
fn test_distributions() {
    let tolerance: f64 = 1e-10;
    // test convoluted binomial
    let p: f64 = 0.5;
    let binom: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::convoluted_binomial(3, p);
    let probabilities: Vec<f64> = vec![0.125, 0.375, 0.375, 0.125];
    for idx in 0..4 {
        assert!((binom.probabilities()[idx] - probabilities[idx]).abs() < tolerance);
    }
    // test convoluted multinomial
    let probabilities: Vec<f64> = vec![0.5, 0.5];
    let multinom: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::convoluted_multinomial(3, probabilities);
    let probabilities: Vec<f64> = vec![0.125, 0.375, 0.375, 0.125];
    for idx in 0..4 {
        assert!((multinom.probabilities()[idx] - probabilities[idx]).abs() < tolerance);
    }
    // test convoluted distributions of arbitrary size
    let conv: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::convoluted_binomial(100, 0.5);
    assert!((0.5f64.powi(100) - conv.probabilities()[0]).abs() < tolerance);
}
