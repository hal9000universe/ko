use crate::DiscreteProbabilityDistribution;

pub fn moment<T>(n: i32, dist: &DiscreteProbabilityDistribution<T>) -> f64
where
    T: Copy + Into<f64>,
{
    // returns the nth moment of a discrete probability distribution
    dist.outcomes
        .iter()
        .zip(dist.probabilities.iter())
        .fold(0., |sum, (x, p)| sum + (*x).into().powi(n) * p)
}

pub fn central_moment<T>(n: i32, dist: &DiscreteProbabilityDistribution<T>) -> f64
where
    T: Copy + Into<f64>,
{
    // returns the nth central moment of a discrete probability distribution
    let mean: f64 = moment(1, &dist);
    dist.outcomes
        .iter()
        .zip(dist.probabilities.iter())
        .fold(0., |sum, (x, p)| sum + ((*x).into() - mean).powi(n) * p)
}
