// TODO: plot multinomial distribution (cdf, pdf, entropy, moments, etc.)

use crate::plotting::plot::plot_data;
use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;
use crate::probability::discrete_information::discrete_entropy;

pub fn plot_binomial_entropy() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot entropy of binomial distribution
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting entropy of binomial distribution

    // create entropy data
    let num_data_points: usize = 100000;
    let mut data: Vec<(f64, f64)> = Vec::with_capacity(num_data_points);
    for data_point_index in 0..num_data_points {
        // create binomial distribution with probability p ranging from 0 to 1
        let p: f64 = data_point_index as f64 / num_data_points as f64;
        let binomial = DiscreteProbabilityDistribution::binomial(p);
        // calculate entropy of binomial distribution
        let entropy: f64 = discrete_entropy(&binomial).to_float();
        data.push((p, entropy));
    }

    // plot entropy data
    let caption: &str = "Entropy of Binomial Distribution";
    let x_desc: &str = "Probability of Success";
    let y_desc: &str = "Entropy";
    let save_file: &str = "plots/distributions/binomial/binomial_entropy.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}
