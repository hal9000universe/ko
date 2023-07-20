use crate::probability::utils::empirical_moment::empirical_standardized_moment;

pub fn empirical_covariance(samples: &Vec<(f64, f64)>) -> f64 {
    //! Calculates the empirical covariance of a set of samples.
    //!
    //! ## Arguments
    //! * `samples`: `&Vec<(f64, f64)`, The samples to calculate the empirical covariance of.
    //!
    //! ## Returns
    //! * `f64`: The empirical covariance of the samples.
    let x_mean: f64 = samples.iter().map(|(x, _)| x).sum::<f64>() / samples.len() as f64;
    let y_mean: f64 = samples.iter().map(|(_, y)| y).sum::<f64>() / samples.len() as f64;
    let covariance: f64 = samples
        .iter()
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>()
        / samples.len() as f64;
    covariance
}

pub fn calc_pearson_coefficient(samples: &Vec<(f64, f64)>) -> f64 {
    //! Calculates the Pearson correlation coefficient of a set of samples.
    //!
    //! ## Arguments
    //! * `samples`: `&Vec<(f64, f64)`, The samples to calculate the Pearson correlation coefficient of.
    //!
    //! ## Returns
    //! * `f64`: The Pearson correlation coefficient of the samples.
    let covariance: f64 = empirical_covariance(&*samples);
    let x_std: f64 = empirical_standardized_moment(2, &samples.iter().map(|(x, _)| *x).collect());
    let y_std: f64 = empirical_standardized_moment(2, &samples.iter().map(|(_, y)| *y).collect());
    covariance / (x_std * y_std)
}
