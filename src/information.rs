use crate::cartesian_product;
use crate::joint_distribution;
use crate::distribution::DiscreteProbabilityDistribution;

pub fn entropy<T>(dist: &DiscreteProbabilityDistribution<T>) -> f64 {
    //! returns the entropy of a discrete probability distribution in shannons
    -dist
        .probabilities()
        .into_iter()
        .fold(0., |sum, p| sum + p * p.log2())
}

pub fn mutual_information<T>(
    dist_x: &DiscreteProbabilityDistribution<T>,
    dist_y: &DiscreteProbabilityDistribution<T>,
    joint_dist: &DiscreteProbabilityDistribution<Vec<T>>,
) -> f64
where
    T: Copy,
{
    //! returns the mutual information of two discrete probability distributions in shannons
    entropy(dist_x) + entropy(dist_y) - entropy(&joint_dist)
}

pub fn joint_entropy(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> f64 {
    //! returns the joint entropy of two discrete probability distributions in shannons
    entropy(&joint_distribution!(dist_x, dist_y))
}
