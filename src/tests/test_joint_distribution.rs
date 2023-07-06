#[cfg(test)]
use crate::discrete_distribution::DiscreteProbabilityDistribution;
#[cfg(test)]
use crate::{cartesian_product, joint_distribution};

#[test]
fn test_joint_distribution() {
    let dist: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(vec![0.5, 0.5]);
    let test_joint_dist: DiscreteProbabilityDistribution<Vec<i32>> =
        joint_distribution!(dist, dist);
    let correct_joint_dist: DiscreteProbabilityDistribution<Vec<i32>> =
        DiscreteProbabilityDistribution::new(
            vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]],
            vec![0.25, 0.25, 0.25, 0.25],
        );
    for test_elem in test_joint_dist.outcomes.iter() {
        assert!(correct_joint_dist.outcomes.contains(test_elem));
    }
    for correct_elem in correct_joint_dist.outcomes.iter() {
        assert!(test_joint_dist.outcomes.contains(correct_elem));
    }

    let x: Vec<i32> = vec![1, 2, 3];
    let y: Vec<i32> = vec![4, 5];
    let dist_x: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::new(x, vec![0.5, 0.5, 0.0]);
    let dist_y: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::new(y, vec![0.5, 0.5]);
    let test_joint_distribution: DiscreteProbabilityDistribution<Vec<i32>> =
        joint_distribution!(dist_x, dist_y);
    let correct_joint_distribution: DiscreteProbabilityDistribution<Vec<i32>> =
        DiscreteProbabilityDistribution::new(
            vec![
                vec![1, 5],
                vec![1, 4],
                vec![2, 4],
                vec![2, 5],
                vec![3, 4],
                vec![3, 5],
            ],
            vec![0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
        );
    for test_elem in test_joint_distribution.outcomes.iter() {
        assert!(correct_joint_distribution.outcomes.contains(test_elem));
    }
    for correct_elem in correct_joint_distribution.outcomes.iter() {
        assert!(test_joint_distribution.outcomes.contains(correct_elem));
    }
}
