//! Contains the `ContinuousProbabilityDistribution` trait and implementations.
//!
//! # Example Normal Distribution
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
//! 
//! # Example Power Law Distribution
//! 
//! ```
//! use ko::continuous_distribution::{ContinuousProbabilityDistribution, PowerLawDistribution};
//! 
//! let power_law = PowerLawDistribution::new(0., 2., 1.);
//! let x = 2.;
//! println!("pdf({}) = {}", x, power_law.pdf(x));
//! println!("cdf({} = {}", x, power_law.cdf(x));
//! println!("sample = {}", power_law.sample());
//! ```

use rand::distributions::Distribution;
use statrs::distribution::{Normal, Uniform};

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

    fn measure(&self, domain: &(f64, f64)) -> f64 {
        //! Returns the measure of the distribution over the set `domain`.
        assert!(domain.0 < domain.1);
        
        // define transformed domain
        let domain_length: f64 = domain.1 - domain.0;
        let g_a: f64 = (domain.0 - self.mean) / domain_length;
        let g_b: f64 = (domain.1 - self.mean) / domain_length;

        // measure transformed function over interval
        let mut measure: f64 = 0.0;
        // start at g_a and increment by epsilon until g_b
        let mut x: f64 = g_a;
        while x < g_b {
            if x + EPSILON < g_b {
                measure += EPSILON * (-0.5 * domain_length.powi(2) / self.variance * x.powi(2)).exp();
                x += EPSILON;
            } else {
                measure += (g_b - x) * (-0.5 * domain_length.powi(2) / self.variance * x.powi(2)).exp();
                break;
            }
        }
        // multiply by domain length and divide by sqrt(2pi*variance)
        measure *= domain_length / (2. * std::f64::consts::PI * self.variance).sqrt();
        // return measure
        measure
    }
}

pub fn normal_distribution_metric(
    normal_x: &NormalDistribution,
    normal_y: &NormalDistribution,
) -> f64 {
    //! Returns the metric between two normal distributions.

    // define domain of metric
    let min: f64 = match (normal_x.mean - 5. * normal_x.variance.sqrt())
        .partial_cmp(&(normal_y.mean - 5. * normal_y.variance.sqrt()))
    {
        Some(std::cmp::Ordering::Less) => normal_x.mean - 5. * normal_x.variance.sqrt(),
        _ => normal_y.mean - 5. * normal_y.variance.sqrt(),
    };
    let max: f64 = match (normal_x.mean + 5. * normal_x.variance.sqrt())
        .partial_cmp(&(normal_y.mean + 5. * normal_y.variance.sqrt()))
    {
        Some(std::cmp::Ordering::Greater) => normal_x.mean + 5. * normal_x.variance.sqrt(),
        _ => normal_y.mean + 5. * normal_y.variance.sqrt(),
    };
    let domain: (f64, f64) = (min, max);

    // metric function over interval
    let mut metric: f64 = 0.0;
    let mut x: f64 = domain.0;
    while x < domain.1 {
        if x + EPSILON < domain.1 {
            metric += EPSILON * (normal_x.pdf(x) - normal_y.pdf(x)).powi(2);
            x += EPSILON;
        } else {
            metric += (domain.1 - x) * (normal_x.pdf(x) - normal_y.pdf(x)).powi(2);
            break;
        }
    }
    metric.sqrt()
}

pub struct PowerLawDistribution {
    factor: f64,
    shift: f64,
    exponent: f64,
    min_x: f64, 
}


impl PowerLawDistribution {
    pub fn new(shift: f64, exponent: f64, min_x: f64) -> Self {
        //! Creates a new `FractalDistribution` from a minimum x value and an exponent.
        assert!(exponent > 1., "exponent must be bigger than 1.");
        assert!(shift < min_x, "min_x must be bigger than shift");
        let factor: f64 = (exponent - 1.) / (min_x - shift).powf(1. - exponent);
        Self { factor, shift, exponent, min_x }
    }
}

impl ContinuousProbabilityDistribution for PowerLawDistribution {
    fn domain(&self) -> (f64, f64) {
        //! Returns the domain of the pdf.
        (self.min_x, f64::INFINITY)
    }

    fn range(&self) -> (f64, f64) {
        //! Returns the range of the pdf.
        (0., f64::INFINITY)
    }

    fn pdf(&self, x: f64) -> f64 {
        //! Returns the probability density function of the outcome `x`.
        self.factor * x.powf(-self.exponent)
    }

    fn cdf(&self, x: f64) -> f64 {
        //! Returns the cumulative distribution function of the outcome `x`.
        self.measure(&(self.min_x, x))
    }

    fn sample(&self) -> f64 {
        //! Returns a random outcome sampled from the distribution.
        let uniform = Uniform::new(0., 1.).unwrap();
        let uniform_sample = uniform.sample(&mut rand::thread_rng());
        self.min_x * (1. - uniform_sample).powf(-1. / (1. - self.exponent))
    }

    fn measure(&self, domain: &(f64, f64)) -> f64 {
        //! Returns the measure of the distribution over the set `domain`.
        assert!(domain.0 < domain.1);
        self.factor * ((domain.0 - self.shift).powf(1. - self.exponent) - (domain.1 - self.shift).powf(1. - self.exponent))
    }
}
