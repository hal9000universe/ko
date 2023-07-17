use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{
    ContinuousProbabilityDistribution, PowerLawDistribution,
};
use crate::probability::empirical_moment::empirical_central_moment;

pub fn plot_power_law_pdf() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot pdf of power law distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting pdf of power law distribution
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let num_points: usize = 100000;
    let min_x: f64 = 1.;
    let max_x: f64 = 10.;
    // collect data points
    let data: Vec<(f64, f64)> = (0..num_points)
        .map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x))
        .map(|x| (x, power_law.pdf(x)))
        .collect();
    let caption: &str = "Power Law PDF";
    let x_desc: &str = "x";
    let y_desc: &str = "pdf(x)";
    let save_file: &str = "plots/distributions/power_law/power_law_pdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_power_law_cdf() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot cdf of power law distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting cdf of power law distribution
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let num_points: usize = 100000;
    let min_x: f64 = 1. + 1e-6;
    let max_x: f64 = 10.;
    // collect data points
    let data: Vec<(f64, f64)> = (0..num_points)
        .map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x))
        .map(|x| (x, power_law.cdf(x)))
        .collect();
    let caption: &str = "Power Law CDF";
    let x_desc: &str = "x";
    let y_desc: &str = "cdf(x)";
    let save_file: &str = "plots/distributions/power_law/power_law_cdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_power_law_empirical_variance() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot empirical variance of power law distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting empirical variance of power law distribution
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let max_num_samples: usize = 100000;
    let mut samples: Vec<f64> = vec![];
    // collect data points
    let data: Vec<(f64, f64)> = (0..max_num_samples)
        .map(|num_samples| {
            // add sample
            samples.push(power_law.sample());
            // compute empirical variance
            let variance: f64 = empirical_central_moment(2, &samples);
            (num_samples as f64, variance)
        })
        .collect();
    let caption: &str = "Power Law Empirical Variance";
    let x_desc: &str = "Number of Samples";
    let y_desc: &str = "Variance of Samples";
    let save_file: &str = "plots/distributions/power_law/power_law_empirical_variance.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}
