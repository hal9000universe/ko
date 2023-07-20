use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{
    ContinuousProbabilityDistribution, PowerLawDistribution,
};
use crate::probability::utils::empirical_moment::empirical_moment;

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

pub fn plot_power_law_moments() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot four empirical moments of power law distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting four empirical moments of power law distribution
    let power_law: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);
    let max_num_samples: usize = 10000;
    let mut samples: Vec<f64> = vec![];
    // collect initial samples
    let num_initial_samples: usize = 10;
    // collect initial samples
    for _ in 0..num_initial_samples {
        samples.push(power_law.sample());
    }
    // allocate space for data points
    let mut first_moment_data: Vec<(f64, f64)> = Vec::with_capacity(max_num_samples);
    let mut second_moment_data: Vec<(f64, f64)> = Vec::with_capacity(max_num_samples);
    let mut third_moment_data: Vec<(f64, f64)> = Vec::with_capacity(max_num_samples);
    let mut fourth_moment_data: Vec<(f64, f64)> = Vec::with_capacity(max_num_samples);
    // collect data points
    for num_samples in num_initial_samples..max_num_samples {
        // add sample
        samples.push(power_law.sample());
        // calculate empirical moments
        let first_moment: f64 = empirical_moment(1, &samples);
        let second_moment: f64 = empirical_moment(2, &samples);
        let third_moment: f64 = empirical_moment(3, &samples);
        let fourth_moment: f64 = empirical_moment(4, &samples);
        // add data points
        first_moment_data.push((num_samples as f64, first_moment));
        second_moment_data.push((num_samples as f64, second_moment));
        third_moment_data.push((num_samples as f64, third_moment));
        fourth_moment_data.push((num_samples as f64, fourth_moment));
    }
    println!("Collecting data points complete");
    // plot data
    for nth_moment in 1..5 {
        let caption: &str = &format!("Power Law Empirical {}th Moment", nth_moment);
        let x_desc: &str = "Number of Samples";
        let y_desc: &str = &format!("{}th Moment of Samples", nth_moment);
        let save_file: &str = &format!(
            "plots/distributions/power_law/moments/power_law_empirical_{}th_moment.png",
            nth_moment
        );
        let data: Vec<(f64, f64)> = match nth_moment {
            1 => first_moment_data.clone(),
            2 => second_moment_data.clone(),
            3 => third_moment_data.clone(),
            4 => fourth_moment_data.clone(),
            _ => panic!("Invalid moment"),
        };
        plot_data(data, caption, x_desc, y_desc, save_file)?;
    }
    Ok(())
}
