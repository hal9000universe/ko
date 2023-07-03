mod cartesian_product;
mod convolution;
mod distribution;
mod information;
mod joint_distribution;
mod moment;
mod tests;

use convolution::{discrete_convolution, special_convolution};
use distribution::DiscreteProbabilityDistribution;
use information::{entropy, joint_entropy, mutual_information};
use moment::{central_moment, moment};

fn main() {
    let probabilities: Vec<f64> = vec![0.5, 0.25, 0.125, 0.125];
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(probabilities);
    println!("Distribution: {:?}", dist);
    let joint_dist: DiscreteProbabilityDistribution<Vec<i32>> = joint_distribution!(dist, dist);
    println!("Joint Distribution: {:?}", joint_dist);
    let special_dist: DiscreteProbabilityDistribution<f64> =
        DiscreteProbabilityDistribution::new(vec![1., 2.], vec![0.5, 0.5]);
    println!("Special Distribution: {:?}", special_dist);
    let conv_dist: DiscreteProbabilityDistribution<f64> =
        special_convolution(&special_dist, &special_dist);
    println!("Special Convolution: {:?}", conv_dist);
    println!("Entropy: {:?}", entropy(&dist));
    println!(
        "Mutual Information: {:?}",
        mutual_information(&dist, &dist, &joint_dist)
    );
    println!("Joint Entropy: {:?}", joint_entropy(&dist, &dist));
    println!("Mean: {:?}", moment(1, &dist));
    println!("Variance: {:?}", central_moment(2, &dist));
    println!(
        "Discrete Convolution: {:?}",
        discrete_convolution(&dist, &dist)
    );
    println!(
        "Probability Sum of Discrete Convolution: {}",
        discrete_convolution(&dist, &dist)
            .probabilities
            .iter()
            .fold(0., |sum, x| sum + x)
    );
}
