#[cfg(test)]
use crate::probability::utils::softmax::softmax;

#[test]
fn test_softmax() {
    let tolerance: f64 = 1e-10;
    let x: Vec<f64> = vec![0., 0., 0.];
    let y: Vec<f64> = softmax(&x);
    let z: Vec<f64> = vec![1. / 3., 1. / 3., 1. / 3.];
    for (a, b) in y.iter().zip(z.iter()) {
        assert!((a - b).abs() < tolerance);
    }

    let x: Vec<f64> = vec![1., -10f64.powi(1000), -10f64.powi(1000)];
    let y: Vec<f64> = softmax(&x);
    let z: Vec<f64> = vec![1., 0., 0.];
    for (a, b) in y.iter().zip(z.iter()) {
        assert!((a - b).abs() < tolerance);
    }
}
