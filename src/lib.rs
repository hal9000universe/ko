pub mod cartesian_product;
pub mod continuous_distribution;
pub mod convolution;
pub mod discrete_distribution;
pub mod information;
pub mod joint_distribution;
pub mod moment;
pub mod tests;

use continuous_distribution::{ContinuousProbabilityDistribution, NormalDistribution};
use convolution::discrete_convolution;
use discrete_distribution::DiscreteProbabilityDistribution;
use information::{entropy, joint_entropy};
use moment::{central_moment, moment};

pub fn all() {
    // discrete probability distribution
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.125, 0.125];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(probabilities);
    println!("Distribution: {:?}", dist);

    // joint distribution
    let joint_dist: DiscreteProbabilityDistribution<Vec<i32>> = joint_distribution!(dist, dist);
    println!("Joint Distribution: {:?}", joint_dist);

    // entropy
    println!("Entropy: {:?}", entropy(&dist));

    // joint entropy
    println!("Joint Entropy: {:?}", joint_entropy(&dist, &dist));

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
}
