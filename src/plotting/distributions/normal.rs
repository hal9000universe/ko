use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{ContinuousProbabilityDistribution, NormalDistribution};

pub fn plot_normal_pdf() -> Result<(), Box<dyn std::error::Error>> {
    let normal: NormalDistribution = NormalDistribution::new(0.0, 1.0);
    let num_points: usize = 100000;
    let min_x: f64 = -4.0;
    let max_x: f64 = 4.0;
    let data: Vec<(f64, f64)> = (0..num_points).map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x)).map(|x| (x, normal.pdf(x))).collect();
    let caption: &str = "Normal Distribution PDF";
    let x_desc: &str = "x";
    let y_desc: &str = "pdf(x)";
    let save_file: &str = "plots/distributions/normal/normal_pdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_normal_cdf() -> Result<(), Box<dyn std::error::Error>> {
    let normal: NormalDistribution = NormalDistribution::new(0.0, 1.0);
    let num_points: usize = 100000;
    let min_x: f64 = -4.0;
    let max_x: f64 = 4.0;
    let data: Vec<(f64, f64)> = (0..num_points).map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x)).map(|x| (x, normal.cdf(x))).collect();
    let caption: &str = "Normal Distribution CDF";
    let x_desc: &str = "x";
    let y_desc: &str = "cdf(x)";
    let save_file: &str = "plots/distributions/normal/normal_cdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_normal_empirical_variance() -> Result<(), Box<dyn std::error::Error>> {
    let normal: NormalDistribution = NormalDistribution::new(0.0, 1.0);
    let max_num_samples: usize = 100000;
    let mut samples: Vec<f64> = vec![];
    let data: Vec<(f64, f64)> = (0..max_num_samples).map(|num_samples| {
        samples.push(normal.sample());
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        (num_samples as f64, variance)
    }).collect();
    let caption: &str = "Normal Distribution Empirical Variance";
    let x_desc: &str = "Number of Samples";
    let y_desc: &str = "Variance of Samples";
    let save_file: &str = "plots/distributions/normal/normal_empirical_variance.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}