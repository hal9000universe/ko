pub mod cartesian_product;
pub mod continuous_distribution;
pub mod discrete_convolution;
pub mod discrete_distribution;
pub mod discrete_information;
pub mod moment;
pub mod sample;
pub mod binomial_testing;
pub mod tests;

use continuous_distribution::{ContinuousProbabilityDistribution, NormalDistribution};
use discrete_convolution::discrete_convolution;
use discrete_distribution::DiscreteProbabilityDistribution;
use discrete_information::{InformationUnit, entropy, kullback_leibler_divergence, jensen_shannon_divergence};
use moment::{central_moment, moment};
use sample::{continuous_sample, discrete_sample};
use binomial_testing::{estimate_binomial, validate_binomial};


pub fn all() {
    // discrete probability distribution
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.125, 0.125];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(probabilities);
    println!("Distribution: {:?}", dist);

    // entropy
    let dist_entropy: InformationUnit = entropy(&dist);
    println!("Entropy: {:?}", dist_entropy);

    // Kullback-Leibler divergence
    let dist_x: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let dist_y: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::multinomial(vec![0.75, 0.25]);
    println!("Kullback-Leibler Divergence: {:?}", kullback_leibler_divergence(&dist_x, &dist_y));

    // Jensen-Shannon divergence
    println!("Jensen-Shannon Divergence: {:?}", jensen_shannon_divergence(&dist_x, &dist_y));

    // moment
    println!("Mean: {:?}", moment(1, &dist));

    // central moment
    println!("Variance: {:?}", central_moment(2, &dist));

    // discrete convolution
    println!(
        "Discrete Convolution: {:?}",
        discrete_convolution(&dist, &dist)
    );

    // probability sum of discrete convolution
    let disc_conv_dist: DiscreteProbabilityDistribution<i32> = discrete_convolution(&dist, &dist);
    println!(
        "Probability Sum of Discrete Convolution: {}",
        disc_conv_dist.measure(&disc_conv_dist.outcomes())
    );

    // continuous probability distribution
    let cont_dist: NormalDistribution = NormalDistribution::new(0., 1.);
    println!("Continuous Distribution: {:?}", cont_dist);
    println!("Probability Density at 0: {:?}", cont_dist.pdf(0.));
    println!("Cumulative Density at 0: {:?}", cont_dist.cdf(0.));
    println!("Sample: {:?}", cont_dist.sample());
    
    // discrete sample
    let disc_samples: Vec<i32> = discrete_sample(1000, &dist);

    // continuous sample
    let cont_samples: Vec<f64> = continuous_sample(1000, &cont_dist);
    println!("Continuous Sample: {:?}", cont_samples);


    // construct binomial distribution
    let binom_dist: DiscreteProbabilityDistribution<i32> = estimate_binomial(&disc_samples);
    println!("Binomial Distribution: {:?}", binom_dist);

    // binomial distinction
    let test_dist: DiscreteProbabilityDistribution<i32> = DiscreteProbabilityDistribution::binomial(0.5);
    let distinction: bool = validate_binomial(&test_dist, &disc_samples);
    println!("Binomial distinction test: {}", distinction);
}
