use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;
use crate::probability::information_unit::InformationUnit;

const EPSILON: f64 = 1e-40;

pub fn discrete_entropy<T>(dist: &DiscreteProbabilityDistribution<T>) -> InformationUnit {
    //! Computes the entropy of a `DiscreteProbabilityDistribution` in bits
    //!
    //! ## Arguments:
    //! * `dist`: `&DiscreteProbabilityDistribution<T>`
    //!
    //! ## Returns:
    //! * `InformationUnit` corresponding to the entropy of the given `DiscreteProbabilityDistribution`
    InformationUnit::Bit(
        -dist
            .probabilities()
            .into_iter()
            .fold(0., |sum, p| sum + p * (p + EPSILON).log2()),
    )
}

pub fn discrete_kullback_leibler_divergence(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> InformationUnit {
    //! Computes the Kullback-Leibler Divergence of two `DiscreteProbabilityDistribution`s in bits
    //!
    //! ## Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution<i32`
    //! * `dist_y`: `&DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Returns:
    //! * `InformationUnit` corresponding to the Kullback-Leibler Divergence of the given
    //! `DiscreteProbabilityDistribution`s
    let mut outcomes: Vec<i32> = dist_x.outcomes();
    outcomes.append(&mut dist_y.outcomes());
    InformationUnit::Bit(
        outcomes
            .iter()
            .map(|&x| x)
            .collect::<std::collections::HashSet<i32>>()
            .iter()
            .map(|x| {
                let p_x: f64 = dist_x.pmf(x);
                let p_y: f64 = dist_y.pmf(x);
                p_x * ((p_x + EPSILON) / p_y).log2()
            })
            .sum(),
    )
}

fn discrete_average_distributions(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> DiscreteProbabilityDistribution<i32> {
    //! Averages two `DiscreteProbabilityDistribution`s
    //!
    //! ## Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution<i32>`
    //! * `dist_y`: `&DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Returns:
    //! * `DiscreteProbabilityDistribution` corresponding to the average distribution of the given two
    //! `DiscreteProbabilityDistribution`s
    let mut outcomes: Vec<i32> = dist_x.outcomes();
    outcomes.append(&mut dist_y.outcomes());
    let outcomes: Vec<i32> = outcomes
        .iter()
        .map(|&x| x)
        .collect::<std::collections::HashSet<i32>>()
        .iter()
        .map(|&x| x)
        .collect();
    let probabilities: Vec<f64> = outcomes
        .iter()
        .map(|x| (dist_x.pmf(x) + dist_y.pmf(x)) / 2.)
        .collect();
    DiscreteProbabilityDistribution::new(outcomes, probabilities)
}

pub fn discrete_jensen_shannon_divergence(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> InformationUnit {
    //! Computes the Jensen-Shannon Divergence of two `DiscreteProbabilityDistribution`s in bits
    //!
    //! Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution`
    //! * `dist_y`: `&DiscreteProbabilityDistribution`
    //!
    //! Returns:
    //! * `InformationUnit` corresponding to the Jensen-Shannon Divergence of the given
    //! `DiscreteProbabilityDistribution`s
    let m: DiscreteProbabilityDistribution<i32> = discrete_average_distributions(dist_x, dist_y);
    (discrete_kullback_leibler_divergence(dist_x, &m)
        + discrete_kullback_leibler_divergence(dist_y, &m))
    .apply(|x| x / 2.)
}
