use rand::distributions::Distribution;
use statrs::distribution::{Normal, Uniform};

const NUM_STEPS: usize = 10000; // for numerical integration

pub trait ContinuousProbabilityDistribution {
    fn domain(&self) -> (f64, f64);
    fn range(&self) -> (f64, f64);
    fn pdf(&self, x: f64) -> f64;
    fn measure(&self, domain: &(f64, f64)) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn sample(&self) -> f64;
}

#[derive(Debug, Clone)]
pub struct NormalDistribution {
    pub mean: f64,
    pub variance: f64,
}

impl NormalDistribution {
    pub fn new(mean: f64, variance: f64) -> Self {
        //! ## Arguments:
        //! * `mean`: `f64`, mean of the distribution
        //! * `variance`: `f64`, variance of the distribution
        //!
        //! ## Returns:
        //! * `NormalDistribution`: a new normal distribution with the given `mean` and `variance`
        //!
        //! ## Panics:
        //! * if `variance` is not positive
        assert!(variance > 0., "variance must be positive");
        Self { mean, variance }
    }

    pub fn estimate(samples: &Vec<f64>) -> Self {
        //! Estimates the parameters of a normal distribution from samples.
        //!
        //! ## Arguments:
        //! * `samples`: `&Vec<f64>`, samples from which to estimate the distribution
        //!
        //! ## Returns:
        //! * `NormalDistribution`: a new normal distribution with the estimated parameters
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        Self::new(mean, variance)
    }
}

impl ContinuousProbabilityDistribution for NormalDistribution {
    fn domain(&self) -> (f64, f64) {
        //! ## Returns
        //! * `domain`: `(f64, f64)`, domain of the pdf
        (-f64::INFINITY, f64::INFINITY)
    }

    fn range(&self) -> (f64, f64) {
        //! ## Returns
        //! * `range`: `(f64, f64)`, range of the pdf
        (0., f64::INFINITY)
    }

    fn pdf(&self, x: f64) -> f64 {
        //! Computes the probability density function of the outcome `x`.
        //!
        //! ## Arguments:
        //! * `x`: `f64`, outcome at which to evaluate the pdf
        //!
        //! ## Returns:
        //! * pdf(`x`): `f64`
        let variance = self.variance;
        let mean = self.mean;
        let coefficient = 1. / (2f64 * std::f64::consts::PI * variance).sqrt();
        let exponent = -(x - mean).powi(2) / (2. * variance);
        coefficient * exponent.exp()
    }

    fn measure(&self, domain: &(f64, f64)) -> f64 {
        //! Computes the measure of the pdf over the interval `domain`.
        //!
        //! ## Arguments:
        //! * `domain`: `&(f64, f64)`, interval over which to measure the pdf
        //!
        //! ## Returns:
        //! * measure of the pdf over `domain`: `f64`
        assert!(domain.0 < domain.1);

        // define transformed domain
        let domain_length: f64 = domain.1 - domain.0;
        let g_a: f64 = (domain.0 - self.mean) / domain_length;
        let g_b: f64 = (domain.1 - self.mean) / domain_length;

        // measure transformed function over interval
        let mut measure: f64 = 0.0;
        let delta: f64 = (g_b - g_a) / NUM_STEPS as f64;
        let mut x: f64 = g_a;
        for _ in 0..NUM_STEPS {
            measure += delta * (-0.5 * domain_length.powi(2) / self.variance * x.powi(2)).exp();
            x += delta;
        }
        // multiply by domain length and divide by sqrt(2pi*variance)
        measure *= domain_length / (2. * std::f64::consts::PI * self.variance).sqrt();
        // return measure
        measure
    }

    fn cdf(&self, x: f64) -> f64 {
        //! Computes the cumulative density function of the outcome `x`.
        //!
        //! ## Arguments:
        //! * `x`: `f64`, outcome at which to evaluate the cdf
        //!
        //! ## Returns:
        //! * cdf(`x`): `f64`
        self.measure(&(self.mean - 5. * self.variance.sqrt(), x))
    }

    fn sample(&self) -> f64 {
        //! Samples from the distribution.
        //!
        //! ## Returns:
        //! * `sample`: `f64`
        let normal: Normal = Normal::new(self.mean, self.variance.sqrt()).unwrap();
        normal.sample(&mut rand::thread_rng())
    }
}

#[derive(Debug, Clone)]
pub struct PowerLawDistribution {
    factor: f64,
    shift: f64,
    exponent: f64,
    min_x: f64,
}

impl PowerLawDistribution {
    pub fn new(shift: f64, exponent: f64, min_x: f64) -> Self {
        //! ## Arguments:
        //! * `shift`: `f64`, shift of the distribution
        //! * `exponent`: `f64`, exponent of the distribution
        //! * `min_x`: `f64`, minimum value of the distribution
        //!
        //! ## Returns:
        //! * `PowerLawDistribution` with the given parameters
        assert!(exponent > 1., "exponent must be bigger than 1.");
        assert!(shift < min_x, "min_x must be bigger than shift");
        let factor: f64 = (exponent - 1.) / (min_x - shift).powf(1. - exponent);
        Self {
            factor,
            shift,
            exponent,
            min_x,
        }
    }

    pub fn estimate(samples: &Vec<f64>) -> Self {
        //! Estimates the parameters of the power law distribution from samples.
        //!
        //! ## Arguments:
        //! * `samples`: `&Vec<f64>`, samples from the distribution
        //!
        //! ## Returns:
        //! * `PowerLawDistribution` with the estimated parameters
        let min_x: f64 = samples
            .clone()
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap()
            .clone();
        let shift: f64 = min_x - 1.;
        let exponent: f64 =
            1. + samples.len() as f64 / samples.iter().map(|x| (x / min_x).ln()).sum::<f64>();
        Self::new(shift, exponent, min_x)
    }
}

impl ContinuousProbabilityDistribution for PowerLawDistribution {
    fn domain(&self) -> (f64, f64) {
        //! ## Returns:
        //! * `domain`: `(f64, f64)`, domain of the pdf
        (self.min_x, f64::INFINITY)
    }

    fn range(&self) -> (f64, f64) {
        //! ## Returns:
        //! * `range`: `(f64, f64)`, range of the pdf
        (0., f64::INFINITY)
    }

    fn pdf(&self, x: f64) -> f64 {
        //! Computes the probability density function of the outcome `x`.
        //!
        //! ## Arguments:
        //! * `x`: `f64`, outcome at which to evaluate the pdf
        //!
        //! ## Returns:
        //! * pdf(`x`): `f64`
        self.factor * x.powf(-self.exponent)
    }

    fn measure(&self, domain: &(f64, f64)) -> f64 {
        //! Computes the measure of the interval `domain`.
        //!
        //! ## Arguments:
        //! * `domain`: `&(f64, f64)`, interval at which to evaluate the measure
        //!
        //! ## Returns:
        //! * measure of the pdf over `domain`: `f64`
        assert!(domain.0 < domain.1);
        self.factor
            * ((domain.0 - self.shift).powf(1. - self.exponent)
                - (domain.1 - self.shift).powf(1. - self.exponent))
    }

    fn cdf(&self, x: f64) -> f64 {
        //! Computes the cumulative density function of the outcome `x`.
        //!
        //! ## Arguments:
        //! * `x`: `f64`, outcome at which to evaluate the cdf
        //!
        //! Returns:
        //! * cdf(`x`): `f64`
        self.measure(&(self.min_x, x))
    }

    fn sample(&self) -> f64 {
        //! Samples from the distribution.
        //!
        //! ## Returns:
        //! * `sample`: `f64`
        let uniform: Uniform = Uniform::new(0., 1.).unwrap();
        let uniform_sample: f64 = uniform.sample(&mut rand::thread_rng());
        (self.min_x - self.shift) * (1. - uniform_sample).powf(1. / (1. - self.exponent))
            + self.shift
    }
}
