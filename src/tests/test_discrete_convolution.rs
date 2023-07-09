#[test]
fn test_discrete_convolution() {
    use crate::discrete_distribution::DiscreteProbabilityDistribution;
    use crate::discrete_convolution::discrete_convolution;
    // create multinomial distribution
    let probabilities: Vec<f64> = vec![0.5, 0.5];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(probabilities);
    // compute convolution
    let conv_dist: DiscreteProbabilityDistribution<i32> = discrete_convolution(&dist, &dist);
    // check result
    assert_eq!(conv_dist.outcomes(), vec![0, 1, 2]);
    assert_eq!(conv_dist.probabilities(), vec![0.25, 0.5, 0.25]);
}
