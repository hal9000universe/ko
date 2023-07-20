pub fn empirical_moment<T>(n: usize, samples: &Vec<T>) -> f64
where
    T: Into<f64> + Copy,
{
    //! Computes the nth moment of a `Vec<T>`
    //!
    //! ## Arguments:
    //! * `n`: `usize`, specifies which moment to compute
    //! * `samples`: `Vec<T>`
    //!
    //! ## Returns:
    //! * nth moment of `samples`: `f64`
    samples
        .iter()
        .fold(0., |sum, x| sum + (*x).into().powi(n as i32))
        / samples.len() as f64
}

pub fn empirical_central_moment<T>(n: usize, samples: &Vec<T>) -> f64
where
    T: Into<f64> + Copy,
{
    //! Computes the nth central moment of a `Vec<T>`
    //!
    //! ## Arguments:
    //! * `n`: `usize`, specifies which central moment to compute
    //! * `samples`: `Vec<T>`
    //!
    //! ## Returns:
    //! * nth central moment of `samples`: `f64`
    let mean: f64 = empirical_moment(1, &samples);
    samples
        .iter()
        .fold(0., |sum, x| sum + ((*x).into() - mean).powi(n as i32))
        / samples.len() as f64
}

pub fn empirical_standardized_moment<T>(n: usize, samples: &Vec<T>) -> f64
where
    T: Into<f64> + Copy,
{
    //! Computes the nth standardized moment of a `Vec<T>`
    //!
    //! ## Arguments:
    //! * `n`: `usize`, specifies which standard moment to compute
    //! * `samples`: `Vec<T>`
    //!
    //! ## Returns:
    //! * nth standardized moment of `samples`: `f64`
    let mean: f64 = empirical_moment(1, &samples);
    let std_dev: f64 = empirical_central_moment(2, &samples).sqrt();
    samples
        .iter()
        .fold(0., |sum, x| sum + ((*x).into() - mean).powi(n as i32))
        / (samples.len() as f64 * std_dev.powi(n as i32))
}
