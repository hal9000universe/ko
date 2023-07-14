use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;

pub fn discrete_moment<T>(n: i32, dist: &DiscreteProbabilityDistribution<T>) -> f64
where
    T: Copy + Into<f64>,
{
    //! Computes the nth moment of a `DiscreteProbabilityDistribution<T>`
    //!
    //! ## Arguments:
    //! * `n`: `i32`, specifies which moment to compute
    //! * `dist`: `&DiscreteProbabilityDistribution<T>`
    //!
    //! ## Returns:
    //! * nth moment of `dist`: `f64`
    dist.outcomes
        .iter()
        .zip(dist.probabilities.iter())
        .fold(0., |sum, (x, p)| sum + (*x).into().powi(n) * p)
}

pub fn discrete_central_moment<T>(n: i32, dist: &DiscreteProbabilityDistribution<T>) -> f64
where
    T: Copy + Into<f64>,
{
    //! Computes the nth central moment of a `DiscreteProbabilityDistribution<T>`
    //!
    //! ## Arguments:
    //! * `n`: `i32`, specifies which central moment to compute
    //! * `dist`: `&DiscreteProbabilityDistribution<T>`
    //!
    //! ## Returns:
    //! * nth central moment of `dist`: `f64`
    let mean: f64 = discrete_moment(1, &dist);
    dist.outcomes
        .iter()
        .zip(dist.probabilities.iter())
        .fold(0., |sum, (x, p)| sum + ((*x).into() - mean).powi(n) * p)
}
