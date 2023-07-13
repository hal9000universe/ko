use crate::plotting::plot::plot_data;
use crate::probability::discrete_distribution::{
    discrete_distribution_metric, DiscreteProbabilityDistribution,
};
use crate::probability::discrete_testing::estimate_binomial;
use crate::probability::sample::discrete_sample;

pub fn binomial_estimation_fidelity_plot() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot the fidelity of binomial estimation
    //!
    //! Generates binomial distributions, samples from them,
    //! fits a binomial distribution to the samples, and
    //! computes the fidelity between the original and fitted
    //! distributions. This is repeated for different numbers
    //! of samples, and the fidelity is plotted against the
    //! number of samples.
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting data

    // number of samples to average over
    let num_samples: usize = 100000;

    // generate data
    let mut data_samples: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        // generate data sample
        let data_sample: Vec<(f64, f64)> = {
            // randomly sample p from the interval (0, 1)
            let p: f64 = rand::random::<f64>();
            let binomial_dist: DiscreteProbabilityDistribution<i32> =
                DiscreteProbabilityDistribution::binomial(p);
            (2..100)
                .map(|num_samples| {
                    let samples: Vec<i32> = discrete_sample(num_samples, &binomial_dist);
                    let estimate_dist: DiscreteProbabilityDistribution<i32> =
                        estimate_binomial(&samples);
                    let fidelity: f64 =
                        discrete_distribution_metric(&binomial_dist, &estimate_dist);
                    (num_samples as f64, fidelity)
                })
                .collect()
        };
        data_samples.push(data_sample);
    }

    // average over samples
    let mut data: Vec<(f64, f64)> = Vec::with_capacity(data_samples[0].len());
    for coord_idx in 0..data_samples[0].len() {
        let mut sum: f64 = 0.0;
        for sample_idx in 0..data_samples.len() {
            sum += data_samples[sample_idx][coord_idx].1;
        }
        data.push((
            data_samples[0][coord_idx].0,
            sum / data_samples.len() as f64,
        ));
    }

    // plot data
    let caption: &str = "Binomial Estimation Fidelity";
    let x_desc: &str = "Number of Samples";
    let y_desc: &str = "Fidelity";
    let save_file: &str = "plots/probability_estimation/binomial_estimation_fidelity.png";
    plot_data(data, caption, x_desc, y_desc, save_file)
}
