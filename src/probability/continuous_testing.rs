use crate::probability::continuous_distribution::ContinuousProbabilityDistribution;

pub const NUM_STEPS: usize = 1000;
const EPSILON: f64 = 0.05;

fn continuous_set_reduce(set: &Vec<f64>) -> Vec<f64> {
    //! Reduces a Vec<f64> to a set of unique values
    //!
    //! ## Arguments:
    //! * `set`: `&Vec<f64>`
    //!
    //! ## Returns:
    //! * reduced_set: `Vec<f64>`, a vector of unique values
    let mut reduced_set: Vec<f64> = set.clone();
    reduced_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    reduced_set.dedup();
    reduced_set.to_vec()
}

pub fn ks_estimate_continuous_cdf(samples: &Vec<f64>) -> Vec<(f64, f64)> {
    //! Estimates the cumulative distribution function given a set of samples
    //!
    //! ## Arguments:
    //! * `samples`: `&Vec<f64>`
    //!
    //! ## Returns:
    //! * cdf: `Vec<(f64, f64)>`, a vector of (x, y) pairs where x is an outcome and y is relative frequency of that outcome
    let n: usize = samples.len();
    let cdf_x: Vec<f64> = continuous_set_reduce(&samples);
    let cdf_y: Vec<f64> = cdf_x
        .clone()
        .into_iter()
        .map(|x| samples.iter().filter(|&y| *y <= x).count() as f64 / n as f64)
        .collect::<Vec<f64>>();
    cdf_x
        .iter()
        .zip(cdf_y.iter())
        .map(|(&x, &y)| (x, y))
        .collect()
}

pub fn ks_evaluate_estimated_cdf(est_dist: &Vec<(f64, f64)>, x: &f64) -> f64 {
    //! Evaluates the estimated CDF at a given outcome
    //!
    //! ## Arguments:
    //! * `est_dist`: `&Vec<(f64, f64)>`, a vector of (x, y) pairs where x is an outcome and y is relative frequency of that outcome
    //! * `x`: &f64, an outcome
    //!
    //! ## Returns:
    //! * the estimated CDF at `x`: `f64`
    est_dist
        .iter()
        .filter(|(x_i, _)| x_i <= x)
        .last()
        .unwrap()
        .1
}

pub fn ks_distance(dist: &impl ContinuousProbabilityDistribution, samples: &Vec<f64>) -> f64 {
    //! Computes the Kolmogorov-Smirnov distance between the estimated CDF and the true CDF
    //!
    //! ## Arguments:
    //! * `dist`: `&impl ContinuousProbabilityDistribution`, a continuous probability distribution
    //! * `samples`: `&Vec<f64>`, a vector of samples
    //!
    //! ## Returns:
    //! * `f64`, the Kolmogorov-Smirnov distance between the estimated CDF and the true CDF
    let cdf: Vec<(f64, f64)> = ks_estimate_continuous_cdf(samples);
    let max_x: f64 = cdf
        .iter()
        .map(|(x, _)| *x)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min_x: f64 = cdf
        .iter()
        .map(|(x, _)| *x)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let delta: f64 = (max_x - min_x) / NUM_STEPS as f64;
    let xs: Vec<f64> = (1..NUM_STEPS)
        .map(|i| min_x + i as f64 * delta)
        .collect::<Vec<f64>>();
    // compute supremum metric
    let max_diff: f64 = xs
        .iter()
        .map(|&x| (dist.cdf(x) - ks_evaluate_estimated_cdf(&cdf, &x)).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    max_diff
}

pub fn ks_validate_continuous_cdf(
    dist: &impl ContinuousProbabilityDistribution,
    samples: &Vec<f64>,
) -> bool {
    //! Performs a Kolmogorov-Smirnov test to determine if the estimated CDF is a good fit to the samples
    //!
    //! ## Arguments:
    //! * `dist`: `&impl ContinuousProbabilityDistribution`, a continuous probability distribution
    //!
    //! ## Returns:
    //! * `bool`, true if the estimated CDF is a good fit for the true CDF
    let max_diff: f64 = ks_distance(dist, samples);
    max_diff < EPSILON
}
