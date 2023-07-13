#[cfg(test)]
use crate::probability::continuous_distribution::{
    ContinuousProbabilityDistribution, NormalDistribution, PowerLawDistribution,
};

#[test]
fn test_normal_cdf() {
    let tolerance: f64 = 1e-2;

    let normal_distribution: NormalDistribution = NormalDistribution::new(0., 1.);
    assert!((normal_distribution.cdf(0.) - 0.5).abs() < tolerance);
    assert!((normal_distribution.cdf(1.) - 0.8413447460685429).abs() < tolerance);
    assert!((normal_distribution.cdf(-1.) - 0.15865525393145707).abs() < tolerance);
    assert!((normal_distribution.cdf(2.) - 0.9772498680518208).abs() < tolerance);
    assert!((normal_distribution.cdf(-2.) - 0.022750131948179195).abs() < tolerance);

    let normal_distribution: NormalDistribution = NormalDistribution::new(10., 23.);
    assert!((normal_distribution.cdf(10.) - 0.5).abs() < tolerance);
    assert!((normal_distribution.measure(&(9., 15.)) - 0.434029613381541).abs() < tolerance);

    let normal_distribution: NormalDistribution = NormalDistribution::new(100., 100.);
    assert!((normal_distribution.cdf(100.) - 0.5).abs() < tolerance);
    assert!((normal_distribution.measure(&(90., 110.)) - 0.68).abs() < tolerance);
    assert!((normal_distribution.measure(&(80., 120.)) - 0.954).abs() < tolerance);
    assert!((normal_distribution.measure(&(70., 130.)) - 0.997).abs() < tolerance);
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

#[test]
fn test_power_law_cdf() {
    let tolerance: f64 = 1e-2;

    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    assert!((power_law.cdf(10f64.powi(10)) - 1.).abs() < tolerance);
    assert!((power_law.measure(&(10., 100.)) - 0.09).abs() < tolerance);
}
