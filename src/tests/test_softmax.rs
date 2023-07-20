#[cfg(test)]
use crate::probability::utils::softmax::softmax;

#[test]
fn test_softmax() {
    let x = vec![0., 0., 0.];
    let y = softmax(&x);
    assert_eq!(y, vec![1. / 3., 1. / 3., 1. / 3.]);

    let x = vec![1., -10f64.powi(1000), -10f64.powi(1000)];
    let y = softmax(&x);
    assert_eq!(y, vec![1., 0., 0.]);
}
