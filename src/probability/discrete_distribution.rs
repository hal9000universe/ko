use rand::{rngs::ThreadRng, Rng};
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct DiscreteProbabilityDistribution<T> {
    pub outcomes: Vec<T>,
    pub probabilities: Vec<f64>,
}

impl<T> DiscreteProbabilityDistribution<T> {
    pub fn new(outcomes: Vec<T>, probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! outcomes and a vector of probabilities.
        //!
        //! ## Arguments:
        //! * `outcomes`: `Vec<T>`, vector of outcomes
        //! * `probabilities`: `Vec<f64>`, vector of probabilities
        //!
        //! ## Returns:
        //! * `DiscreteProbabilityDistribution<T>`, the discrete probability
        //! distribution
        //!
        //! ## Panics:
        //!
        //! Panics if the lengths of `outcomes` and `prbobabilities` are not equal, if the
        //! probabilities are not non-negative, or if the probabilities do not
        //! sum to 1.
        assert_eq!(
            outcomes.len(),
            probabilities.len(),
            "outcomes and probabilities must have the same length"
        );
        assert!(
            probabilities.iter().all(|&p| p >= -1e-10),
            "probabilities must be non-negative"
        );
        assert!(
            (probabilities.iter().sum::<f64>() - 1.).abs() < 1e-10,
            "probabilities must sum to 1"
        );
        Self {
            outcomes,
            probabilities,
        }
    }

    pub fn probabilities(&self) -> Vec<f64> {
        //! ## Returns:
        //! * `Vec<f64>`, clone of the probabilities
        self.probabilities.clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Clone,
{
    pub fn outcomes(&self) -> Vec<T> {
        //! ## Returns:
        //! * `Vec<T>`, clone of the outcomes
        self.outcomes.clone()
    }

    pub fn sample(&self) -> T {
        //! ## Returns:
        //! * `T`, a random outcome
        let mut rng: ThreadRng = rand::thread_rng();
        let mut u: f64 = rng.gen::<f64>();
        let mut i: usize = 0;
        while u > 0. {
            u -= self.probabilities[i];
            i += 1;
        }
        self.outcomes[i - 1].clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Eq,
{
    pub fn pmf(&self, x: &T) -> f64 {
        //! Computes the probability mass function of the outcome `x`.
        //!
        //! ## Arguments:
        //! * `x`: `&T`, outcome
        //!
        //! ## Returns:
        //! * `f64`, the probability mass function of the outcome `x`
        match self.outcomes.iter().position(|y| y == x) {
            Some(i) => self.probabilities[i],
            None => 0.,
        }
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Eq + Hash,
{
    pub fn measure(&self, domain: &Vec<T>) -> f64 {
        //! Computes the measure of the distribution over the set `domain`.
        //!
        //! ## Arguments:
        //! * `domain`: `&Vec<T>`, vector of outcomes
        //!
        //! ## Returns:
        //! * `f64`, the measure of the distribution over the set `domain`
        assert!(
            domain.len()
                == domain
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
        );
        domain.iter().map(|x| self.pmf(x)).sum()
    }
}

impl DiscreteProbabilityDistribution<i32> {
    pub fn multinomial(probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! probabilities. The outcomes are the integers from 0 to
        //! `probabilities.len() - 1`.
        //!
        //! ## Arguments:
        //! * `probabilities`: `Vec<f64>`, vector of probabilities
        //!
        //! ## Returns:
        //! * `DiscreteProbabilityDistribution<i32>`, the discrete probability
        //! distribution
        let outcomes: Vec<i32> = (0..probabilities.len() as i32).collect();
        Self::new(outcomes, probabilities)
    }

    pub fn binomial(p: f64) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a probability
        //! of success `p`.
        //!
        //! ## Arguments:
        //! * `p`: `f64`, probability of success
        //!
        //! ## Returns:
        //! * `DiscreteProbabilityDistribution<i32>`, the discrete probability
        //! distribution
        Self::multinomial(vec![1. - p, p])
    }
}

pub fn discrete_distribution_metric<T>(
    dist_x: &DiscreteProbabilityDistribution<T>,
    dist_y: &DiscreteProbabilityDistribution<T>,
) -> f64
where
    T: Eq + Hash + Copy,
{
    //! Calculates the metric between two `DiscreteProbabilityDistribution`s.
    //!
    //! ## Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution<T>`, one discrete probability
    //! distribution
    //! * `dist_y`: `&DiscreteProbabilityDistribution<T>`, another discrete probability
    //! distribution

    // define domain
    let mut domain: Vec<T> = dist_x.outcomes();
    domain.append(&mut dist_y.outcomes());
    // remove duplicates
    let domain: Vec<T> = domain
        .iter()
        .collect::<std::collections::HashSet<&T>>()
        .iter()
        .map(|&&x| x)
        .collect::<Vec<T>>();
    // calculate metric
    let mut metric: f64 = 0.;
    for x in domain {
        metric += (dist_x.pmf(&x) - dist_y.pmf(&x)).powi(2);
    }
    metric.sqrt()
}
