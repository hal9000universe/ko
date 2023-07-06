#[cfg(test)]
use crate::continuous_distribution::{ContinuousProbabilityDistribution, NormalDistribution};

#[test]
fn test_normal_cdf() {
    let tolerance: f64 = 1e-2;
    let normal_distribution: NormalDistribution = NormalDistribution::new(0., 1.);
    assert!((normal_distribution.cdf(0.) - 0.5).abs() < tolerance);
    assert!((normal_distribution.cdf(1.) - 0.8413447460685429).abs() < tolerance);
    assert!((normal_distribution.cdf(-1.) - 0.15865525393145707).abs() < tolerance);
    assert!((normal_distribution.cdf(2.) - 0.9772498680518208).abs() < tolerance);
    assert!((normal_distribution.cdf(-2.) - 0.022750131948179195).abs() < tolerance);
}

#[test]
fn test_normal_pdf() {
    let tolerance: f64 = 1e-6;
    let normal_distribution: NormalDistribution = NormalDistribution::new(0., 1.);
    assert!((normal_distribution.pdf(0.) - 0.3989422804014327).abs() < tolerance);
    assert!((normal_distribution.pdf(1.) - 0.24197072451914337).abs() < tolerance);
    assert!((normal_distribution.pdf(-1.) - 0.24197072451914337).abs() < tolerance);
    assert!((normal_distribution.pdf(2.) - 0.05399096651318806).abs() < tolerance);
    assert!((normal_distribution.pdf(-2.) - 0.05399096651318806).abs() < tolerance);
}
