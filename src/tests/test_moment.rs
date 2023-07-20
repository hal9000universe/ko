#[cfg(test)]
use crate::probability::utils::empirical_moment::{
    empirical_central_moment, empirical_moment, empirical_standardized_moment,
};

#[test]
fn test_empirical_moment() {
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5];
    let eps: f64 = 1e-6;
    assert!(empirical_moment(1, &samples) - 3. < eps);
    assert!(empirical_moment(2, &samples) - 11. < eps);
    assert!(empirical_moment(3, &samples) - 45. < eps);
    assert!(empirical_moment(4, &samples) - 195.8 < eps);

    let samples: Vec<f64> = vec![1., 2., 3., 4., 5.];
    assert!(empirical_moment(1, &samples) - 3. < eps);
    assert!(empirical_moment(2, &samples) - 11. < eps);
    assert!(empirical_moment(3, &samples) - 45. < eps);
    assert!(empirical_moment(4, &samples) - 195.8 < eps);
}

#[test]
fn test_empirical_central_moment() {
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5];
    let eps: f64 = 1e-6;
    assert!(empirical_central_moment(1, &samples) - 0. < eps);
    assert!(empirical_central_moment(2, &samples) - 2. < eps);
    assert!(empirical_central_moment(3, &samples) - 0. < eps);
    assert!(empirical_central_moment(4, &samples) - 6.8 < eps);

    let samples: Vec<f64> = vec![1., 2., 3., 4., 5.];
    assert!(empirical_central_moment(1, &samples) - 0. < eps);
    assert!(empirical_central_moment(2, &samples) - 2. < eps);
    assert!(empirical_central_moment(3, &samples) - 0. < eps);
    assert!(empirical_central_moment(4, &samples) - 6.8 < eps);
}

#[test]
fn test_standardized_moment() {
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5];
    let eps: f64 = 1e-6;
    assert!(empirical_standardized_moment(1, &samples) - 0. < eps);
    assert!(empirical_standardized_moment(2, &samples) - 1. < eps);
    assert!(empirical_standardized_moment(3, &samples) - 0. < eps);
    assert!(empirical_standardized_moment(4, &samples) - 1.6999999999 < eps);

    let samples: Vec<f64> = vec![1., 2., 3., 4., 5.];
    assert!(empirical_standardized_moment(1, &samples) - 0. < eps);
    assert!(empirical_standardized_moment(2, &samples) - 1. < eps);
    assert!(empirical_standardized_moment(3, &samples) - 0. < eps);
    assert!(empirical_standardized_moment(4, &samples) - 1.6999999999 < eps);
}
