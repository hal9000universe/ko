use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{ContinuousProbabilityDistribution, PowerLawDistribution};

pub fn plot_power_law_pdf() -> Result<(), Box<dyn std::error::Error>> {
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let num_points: usize = 100000;
    let min_x: f64 = 1.;
    let max_x: f64 = 10.;
    let data: Vec<(f64, f64)> = (0..num_points).map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x)).map(|x| (x, power_law.pdf(x))).collect();
    let caption: &str = "Power Law PDF";
    let x_desc: &str = "x";
    let y_desc: &str = "pdf(x)";
    let save_file: &str = "plots/distributions/power_law/power_law_pdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_power_law_cdf() -> Result<(), Box<dyn std::error::Error>> {
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let num_points: usize = 100000;
    let min_x: f64 = 1. + 1e-6;
    let max_x: f64 = 10.;
    let data: Vec<(f64, f64)> = (0..num_points).map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x)).map(|x| (x, power_law.cdf(x))).collect();
    let caption: &str = "Power Law CDF";
    let x_desc: &str = "x";
    let y_desc: &str = "cdf(x)";
    let save_file: &str = "plots/distributions/power_law/power_law_cdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_power_law_empirical_variance() -> Result<(), Box<dyn std::error::Error>> {
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let max_num_samples: usize = 100000;
    let mut samples: Vec<f64> = vec![];
    let data: Vec<(f64, f64)> = (0..max_num_samples).map(|num_samples| {
        samples.push(power_law.sample());
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        (num_samples as f64, variance)
    }).collect();
    let caption: &str = "Power Law Empirical Variance";
    let x_desc: &str = "Number of Samples";
    let y_desc: &str = "Variance of Samples";
    let save_file: &str = "plots/distributions/power_law/power_law_empirical_variance.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}