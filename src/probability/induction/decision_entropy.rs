use crate::probability::discrete_distribution::DiscreteProbabilityDistribution;
use crate::probability::information::discrete_entropy;
use crate::probability::information_unit::InformationUnit;
use crate::probability::utils::softmax::softmax;

const EPSILON: f64 = 1e-10;

pub fn compute_decision_entropy(estimation_errors: &Vec<f64>) -> InformationUnit {
    //! Computes the decision entropy given a set of estimation errors (e.g. Kolmogorov-Smirnov Distances).
    //!
    //! ## Arguments:
    //! * `estimation_errors`: `&Vec<f64>`, a vector of estimation errors for every possible probability distribution
    //!
    //! ## Returns:
    //! * `f64`, the decision entropy
    let decision_probabilities: Vec<f64> = softmax(
        &estimation_errors
            .iter()
            .map(|x| 1. / (x + EPSILON))
            .collect(),
    );
    let decision_distribution: DiscreteProbabilityDistribution<i32> =
        DiscreteProbabilityDistribution::multinomial(decision_probabilities);
    discrete_entropy(&decision_distribution)
}
