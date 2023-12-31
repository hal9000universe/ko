use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{
    ContinuousProbabilityDistribution, PowerLawDistribution,
};
use crate::probability::induction::continuous_testing::{
    ks_estimate_continuous_cdf, ks_evaluate_estimated_cdf, NUM_STEPS,
};
use crate::probability::utils::sample::continuous_sample;

pub fn plot_ks_true_power_law_estimation_fidelity() -> Result<(), Box<dyn std::error::Error>> {
    //! Plots the fidelity of the Kolmogorov-Smirnov test for the true power law distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`
    let mut samples: Vec<f64> = Vec::new();
    let mut cdf_diffs: Vec<(f64, f64)> = Vec::new();
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    for _ in (0..100).map(|x| 10 * x) {
        samples.extend(continuous_sample(10, &power_law));
        let cdf: Vec<(f64, f64)> = ks_estimate_continuous_cdf(&samples);
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

        let max_diff: f64 = xs
            .iter()
            .map(|&x| (power_law.cdf(x) - ks_evaluate_estimated_cdf(&cdf, &x)).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        cdf_diffs.push((samples.len() as f64, max_diff));
    }
    plot_data(
        cdf_diffs,
        "True Power Law Distribution KS Test Fidelity",
        "Number of Samples",
        "Fidelity",
        "plots/probability_estimation/ks_true_power_law_estimation.png",
    )
}

pub fn plot_ks_false_power_law_estimation_fidelity() -> Result<(), Box<dyn std::error::Error>> {
    //! Plots the fidelity of the Kolmogorov-Smirnov test for a false power law distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`
    let mut samples: Vec<f64> = Vec::new();
    let mut cdf_diffs: Vec<(f64, f64)> = Vec::new();
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let sample_dist: PowerLawDistribution = PowerLawDistribution::new(1., 4., 2.);
    for _ in (0..100).map(|x| 10 * x) {
        samples.extend(continuous_sample(10, &sample_dist));
        let cdf: Vec<(f64, f64)> = ks_estimate_continuous_cdf(&samples);
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

        let max_diff: f64 = xs
            .iter()
            .map(|&x| (power_law.cdf(x) - ks_evaluate_estimated_cdf(&cdf, &x)).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        cdf_diffs.push((samples.len() as f64, max_diff));
    }
    plot_data(
        cdf_diffs,
        "False Power Law Distribution KS Test Fidelity",
        "Number of Samples",
        "Fidelity",
        "plots/probability_estimation/ks_false_power_law_estimation.png",
    )
}
