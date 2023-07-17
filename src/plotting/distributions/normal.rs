use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{
    ContinuousProbabilityDistribution, NormalDistribution,
};
use crate::probability::empirical_moment::empirical_central_moment;

pub fn plot_normal_pdf() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot pdf of normal distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting pdf of normal distribution
    let normal: NormalDistribution = NormalDistribution::new(0.0, 1.0);
    let num_points: usize = 100000;
    let min_x: f64 = -4.0;
    let max_x: f64 = 4.0;
    // collect data points
    let data: Vec<(f64, f64)> = (0..num_points)
        .map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x))
        .map(|x| (x, normal.pdf(x)))
        .collect();
    let caption: &str = "Normal Distribution PDF";
    let x_desc: &str = "x";
    let y_desc: &str = "pdf(x)";
    let save_file: &str = "plots/distributions/normal/normal_pdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_normal_cdf() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot cdf of normal distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting cdf of normal distribution
    let normal: NormalDistribution = NormalDistribution::new(0.0, 1.0);
    let num_points: usize = 100000;
    let min_x: f64 = -4.0;
    let max_x: f64 = 4.0;
    // collect data points
    let data: Vec<(f64, f64)> = (0..num_points)
        .map(|i| min_x + (i as f64 / num_points as f64) * (max_x - min_x))
        .map(|x| (x, normal.cdf(x)))
        .collect();
    let caption: &str = "Normal Distribution CDF";
    let x_desc: &str = "x";
    let y_desc: &str = "cdf(x)";
    let save_file: &str = "plots/distributions/normal/normal_cdf.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}

pub fn plot_normal_empirical_variance() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot empirical variance of normal distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting empirical variance of normal distribution
    let normal: NormalDistribution = NormalDistribution::new(0.0, 1.0);
    let max_num_samples: usize = 100000;
    let mut samples: Vec<f64> = vec![];
    // collect data points
    let data: Vec<(f64, f64)> = (0..max_num_samples)
        .map(|num_samples| {
            // add sample
            samples.push(normal.sample());
            // calculate empirical variance
            let variance: f64 = empirical_central_moment(2, &samples);
            (num_samples as f64, variance)
        })
        .collect();
    let caption: &str = "Normal Distribution Empirical Variance";
    let x_desc: &str = "Number of Samples";
    let y_desc: &str = "Variance of Samples";
    let save_file: &str = "plots/distributions/normal/normal_empirical_variance.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}
