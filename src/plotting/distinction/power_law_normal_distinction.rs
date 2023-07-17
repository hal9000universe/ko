use crate::plotting::plot::plot_data;
use crate::probability::continuous_distribution::{
    ContinuousProbabilityDistribution, NormalDistribution, PowerLawDistribution,
};
use crate::probability::continuous_testing::ks_distance;
use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;
use crate::probability::discrete_information::discrete_entropy;
use crate::probability::empirical_moment::empirical_central_moment;
use crate::probability::sample::continuous_sample;
use crate::probability::softmax::softmax;

const EPSILON: f64 = 1e-10;
const PLOT_DIR: &str = "plots/distinction/";

fn collect_power_law_distinction_data(
    sample_dist: &impl ContinuousProbabilityDistribution,
) -> (
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
) {
    // collect data
    let mut normal_ks_dist_data: Vec<(f64, f64)> = Vec::new(); // (num_samples, ks_distance_normal)
    let mut power_law_ks_dist_data: Vec<(f64, f64)> = Vec::new(); // (num_samples, ks_distance_power_law)
    let mut variance_data: Vec<(f64, f64)> = Vec::new(); // (num_samples, variance)
    let mut decision_entropy_data: Vec<(f64, f64)> = Vec::new(); // (num_samples, decision_confidence)

    // define number of samples
    let num_samples: usize = 80;
    let num_start_samples: usize = 20;
    let mut samples: Vec<f64> = continuous_sample(num_start_samples, &*sample_dist);

    // iterate over samples
    for sample_idx in num_start_samples..num_samples {
        // sample from normal distribution
        samples.push(sample_dist.sample());
        // estimate normal distribution
        let est_normal_dist: NormalDistribution = NormalDistribution::estimate(&samples);
        // estimate power law distribution
        let est_power_law_dist: PowerLawDistribution = PowerLawDistribution::estimate(&samples);
        // calculate ks distances
        let ks_distance_normal: f64 = ks_distance(&est_normal_dist, &samples);
        let ks_distance_power_law: f64 = ks_distance(&est_power_law_dist, &samples);
        // calculate variance
        let variance: f64 = empirical_central_moment(2, &samples);
        // calculate decision probabilities
        let decision_probabilities: Vec<f64> = softmax(&vec![
            1. / ks_distance_normal + EPSILON,
            1. / ks_distance_power_law + EPSILON,
        ]);
        let decision_distribution: DiscreteProbabilityDistribution<i32> =
            DiscreteProbabilityDistribution::multinomial(decision_probabilities);
        let decision_entropy: f64 = discrete_entropy(&decision_distribution).to_float();
        // add data to vectors
        normal_ks_dist_data.push((sample_idx as f64, ks_distance_normal));
        power_law_ks_dist_data.push((sample_idx as f64, ks_distance_power_law));
        variance_data.push((sample_idx as f64, variance));
        decision_entropy_data.push((sample_idx as f64, decision_entropy));
    }

    // return data
    (
        normal_ks_dist_data,
        power_law_ks_dist_data,
        variance_data,
        decision_entropy_data,
    )
}

pub fn average_data_collection(collection: Vec<Vec<(f64, f64)>>) -> Vec<(f64, f64)> {
    // average data over samples
    let data_start: Vec<(f64, f64)> = collection[0].clone();
    let data: Vec<(f64, f64)> = collection
        .iter()
        .fold(data_start.clone(), |acc, x| {
            acc.iter()
                .zip(x.iter())
                .map(|(a, b)| (a.0 + b.0, a.1 + b.1))
                .collect()
        })
        .iter()
        .enumerate()
        .map(|(idx, (x, y))| (x - data_start[idx].0, y - data_start[idx].1))
        .map(|(x, y)| (x / collection.len() as f64, y / collection.len() as f64))
        .collect();
    data
}

pub fn plot_normal_power_law_distinction() -> Result<(), Box<dyn std::error::Error>> {
    //! Plot the distinction between a normal and power law distribution
    //!
    //! ## Arguments:
    //! * `samples`: `&Vec<f64>`, samples to plot
    //! * `save_file`: `&str`, file to save plot to
    //!
    //! ## Returns:
    //! * `Result<(), Box<dyn std::error::Error>>`: Result of plotting data

    // create normal distribution to sample from
    let power_law_sample_dist: PowerLawDistribution = PowerLawDistribution::new(0., 2., 1.);

    // collect data
    let mut normal_ks_dist_data_collections: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut power_law_ks_dist_data_collections: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut variance_data_collections: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut decision_entropy_data_collections: Vec<Vec<(f64, f64)>> = Vec::new();

    // define number of samples
    let num_samples: usize = 100;
    for idx in 0..num_samples {
        let (normal_ks_dist_data, power_law_ks_dist_data, variance_data, decision_entropy_data) =
            collect_power_law_distinction_data(&power_law_sample_dist);
        // plot data
        plot_data(
            normal_ks_dist_data.clone(),
            "KS Distance Normal Distribution",
            "Number of Samples",
            "KS Distance",
            &format!(
                "{}{}{}{}.png",
                PLOT_DIR, "random/", idx, "-power_law_normal_ks_dist"
            ),
        )?;
        plot_data(
            power_law_ks_dist_data.clone(),
            "KS Distance Power Law Distribution",
            "Number of Samples",
            "KS Distance",
            &format!(
                "{}{}{}{}.png",
                PLOT_DIR, "random/", idx, "-power_law_power_law_ks_dist"
            ),
        )?;
        plot_data(
            variance_data.clone(),
            "Variance",
            "Number of Samples",
            "Variance",
            &format!(
                "{}{}{}{}.png",
                PLOT_DIR, "random/", idx, "-power_law_variance"
            ),
        )?;
        plot_data(
            decision_entropy_data.clone(),
            "Decision Entropy",
            "Number of Samples",
            "Decision Entropy",
            &format!(
                "{}{}{}{}.png",
                PLOT_DIR, "random/", idx, "-power_law_decision_entropy"
            ),
        )?;
        normal_ks_dist_data_collections.push(normal_ks_dist_data);
        power_law_ks_dist_data_collections.push(power_law_ks_dist_data);
        variance_data_collections.push(variance_data);
        decision_entropy_data_collections.push(decision_entropy_data);
    }
    println!("Collected Data");

    // average data over samples
    let normal_ks_dist_data: Vec<(f64, f64)> =
        average_data_collection(normal_ks_dist_data_collections);
    let power_law_ks_dist_data: Vec<(f64, f64)> =
        average_data_collection(power_law_ks_dist_data_collections);
    let variance_data: Vec<(f64, f64)> = average_data_collection(variance_data_collections);
    let decision_entropy_data: Vec<(f64, f64)> =
        average_data_collection(decision_entropy_data_collections);
    println!("Averaged Data");

    // plot data
    plot_data(
        normal_ks_dist_data,
        "Average KS Distance",
        "Number of Samples",
        "Normal Distribution KS Distance (Power Law Samples)",
        "plots/distinction/power_law_normal_ks_dist.png",
    )?;
    plot_data(
        power_law_ks_dist_data,
        "Average KS Distance",
        "Number of Samples",
        "Power Law Distribution KS Distance (Power Law Samples)",
        "plots/distinction/power_law_power_law_ks_dist.png",
    )?;
    plot_data(
        variance_data,
        "Average Variance",
        "Number of Samples",
        "Variance of Power Law Samples",
        "plots/distinction/power_law_variance.png",
    )?;
    plot_data(
        decision_entropy_data,
        "Decision Entropy",
        "Number of Samples",
        "Average Entropy of Decision Distribution",
        "plots/distinction/power_law_normal_decision_entropy.png",
    )?;

    Ok(())
}
