pub mod cartesian_product;
pub mod convolution;
pub mod distribution;
pub mod function;
pub mod information;
pub mod joint_distribution;
pub mod moment;
pub mod tests;

use convolution::{discrete_convolution, special_convolution};
use distribution::DiscreteProbabilityDistribution;
use function::{ContinuousFunction, DiscreteFunction, Function};
use information::{entropy, joint_entropy, mutual_information};
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

    // special distribution
    let special_dist: DiscreteProbabilityDistribution<f64> =
        DiscreteProbabilityDistribution::new(vec![1., 2.], vec![0.5, 0.5]);
    println!("Special Distribution: {:?}", special_dist);

    // special convolution
    let conv_dist: DiscreteProbabilityDistribution<f64> =
        special_convolution(&special_dist, &special_dist);
    println!("Special Convolution: {:?}", conv_dist);

    // entropy
    println!("Entropy: {:?}", entropy(&dist));

    // mutual information
    println!(
        "Mutual Information: {:?}",
        mutual_information(&dist, &dist, &joint_dist)
    );

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
    println!(
        "Probability Sum of Discrete Convolution: {}",
        discrete_convolution(&dist, &dist)
            .probabilities
            .iter()
            .fold(0., |sum, x| sum + x)
    );

    // continuous function
    let f: ContinuousFunction = ContinuousFunction::new(vec![0., 1.], vec![0., 1.], |x| x * x);
    println!("Continuous Function at 0.5: {}", f.eval(&0.5));

    // discrete function
    let f: DiscreteFunction<i32, i32> = DiscreteFunction::new(vec![0, 1], |x| x * x);
    println!("Discrete Function at 0.5: {}", f.eval(&1));
}
