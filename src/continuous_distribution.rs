//! Contains the `ContinuousProbabilityDistribution` trait and implementations.
//!
//! # Examples
//!
//! ```
//! use ko::continuous_distribution::{NormalDistribution, ContinuousProbabilityDistribution};
//!
//! let normal = NormalDistribution::new(0., 1.);
//! let x = 0.5;
//! println!("pdf({}) = {}", x, normal.pdf(x));
//! println!("cdf({}) = {}", x, normal.cdf(x));
//! println!("sample = {}", normal.sample());
//! ```

use rand::distributions::Distribution;
use statrs::distribution::Normal;

const EPSILON: f64 = 0.001; // for numerical integration

pub trait ContinuousProbabilityDistribution {
    fn domain(&self) -> (f64, f64);
    fn range(&self) -> (f64, f64);
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn sample(&self) -> f64;
    fn measure(&self, domain: &(f64, f64)) -> f64 {
        //! Returns the measure of the distribution over the set `domain`.
        assert!(domain.0 < domain.1);
        // measure function over interval
        let mut measure: f64 = 0.0;
        let mut x: f64 = domain.0;
        while x < domain.1 {
            if x + EPSILON < domain.1 {
                measure += EPSILON * self.pdf(x);
                x += EPSILON;
            } else {
                measure += (domain.1 - x) * self.pdf(x);
                break;
            }
        }
        measure
    }
}

#[derive(Debug)]
pub struct NormalDistribution {
    mean: f64,
    variance: f64,
}

impl NormalDistribution {
    pub fn new(mean: f64, variance: f64) -> Self {
        //! Creates a new `NormalDistribution` from a mean and a variance.
        assert!(variance > 0., "variance must be positive");
        Self { mean, variance }
    }
}

impl ContinuousProbabilityDistribution for NormalDistribution {
    fn domain(&self) -> (f64, f64) {
        //! Returns the domain of the pdf.
        (-f64::INFINITY, f64::INFINITY)
    }

    fn range(&self) -> (f64, f64) {
        //! Returns the range of the pdf.
        (0., f64::INFINITY)
    }

    fn pdf(&self, x: f64) -> f64 {
        //! Returns the probability density function of the outcome `x`.
        let variance = self.variance;
        let mean = self.mean;
        let coefficient = 1. / (2f64 * std::f64::consts::PI * variance).sqrt();
        let exponent = -(x - mean).powi(2) / (2. * variance);
        coefficient * exponent.exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        //! Returns the cumulative distribution function of the outcome `x`.
        self.measure(&(self.mean - 5. * self.variance.sqrt(), x))
    }

    fn sample(&self) -> f64 {
        //! Returns a random outcome sampled from the distribution.
        let normal = Normal::new(self.mean, self.variance.sqrt()).unwrap();
        normal.sample(&mut rand::thread_rng())
    }
}
