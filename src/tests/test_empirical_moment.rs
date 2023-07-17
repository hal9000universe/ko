#[cfg(test)]
use crate::probability::empirical_moment::{empirical_central_moment, empirical_moment};

#[test]
fn test_empirical_moment() {
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5];
    assert_eq!(empirical_moment(1, &samples), 3.);
    assert_eq!(empirical_moment(2, &samples), 11.);
    assert_eq!(empirical_moment(3, &samples), 45.);
    assert_eq!(empirical_moment(4, &samples), 195.8);

    let samples: Vec<f64> = vec![1., 2., 3., 4., 5.];
    assert_eq!(empirical_moment(1, &samples), 3.);
    assert_eq!(empirical_moment(2, &samples), 11.);
    assert_eq!(empirical_moment(3, &samples), 45.);
    assert_eq!(empirical_moment(4, &samples), 195.8);
}

#[test]
fn test_empirical_central_moment() {
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5];
    assert_eq!(empirical_central_moment(1, &samples), 0.);
    assert_eq!(empirical_central_moment(2, &samples), 2.);
    assert_eq!(empirical_central_moment(3, &samples), 0.);
    assert_eq!(empirical_central_moment(4, &samples), 6.8);

    let samples: Vec<f64> = vec![1., 2., 3., 4., 5.];
    assert_eq!(empirical_central_moment(1, &samples), 0.);
    assert_eq!(empirical_central_moment(2, &samples), 2.);
    assert_eq!(empirical_central_moment(3, &samples), 0.);
    assert_eq!(empirical_central_moment(4, &samples), 6.8);
}
