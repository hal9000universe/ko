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

pub fn discrete_convolution(
    dist_x: &DiscreteProbabilityDistribution<i32>,
    dist_y: &DiscreteProbabilityDistribution<i32>,
) -> DiscreteProbabilityDistribution<i32> {
    //! Computes a discrete convolution between two discrete probability distributions, the random variables of which are independent and integer-valued.
    //! The convolution is computed by summing the probabilities of the cartesian product of the outcomes of the two distributions.
    //!
    //! ## Arguments:
    //! * `dist_x`: `&DiscreteProbabilityDistribution<i32>`
    //! * `dist_y`: `&DiscreteProbabilityDistribution<i32>`
    //!
    //! ## Returns:
    //! * `DiscreteProbabilityDistribution<i32>`, the discrete probability distribution
    let min: i32 =
        dist_x.outcomes().iter().min().unwrap() + dist_y.outcomes().iter().min().unwrap();
    let max: i32 =
        dist_x.outcomes().iter().max().unwrap() + dist_y.outcomes().iter().max().unwrap();
    let outcomes: Vec<i32> = (min..max + 1).collect();
    // compute probabilities
    let probabilities: Vec<f64> = outcomes
        .iter()
        .map(|&z| {
            dist_x
                .outcomes()
                .iter()
                .map(|&k| dist_x.pmf(&k) * dist_y.pmf(&(z - k)))
                .sum()
        })
        .collect();
    // filter out outcomes with zero probability
    let outcomes: Vec<i32> = outcomes
        .iter()
        .zip(probabilities.iter())
        .filter(|(_, &p)| p > 0.)
        .map(|(&z, _)| z)
        .collect();
    // filter out probabilities with zero probability
    let probabilities: Vec<f64> = probabilities
        .iter()
        .filter(|&p| p > &0.)
        .map(|&p| p)
        .collect();
    DiscreteProbabilityDistribution::new(outcomes, probabilities)
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

    pub fn convoluted_multinomial(n: usize, probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! probabilities by convoluting the distribution with itself `n` times.
        //!
        //! ## Arguments:
        //! * `n`: `usize`, number of convolutions
        //! * `probabilities`: `Vec<f64>`, vector of probabilities
        //!
        //! ## Returns:
        //! * `DiscreteProbabilityDistribution<i32>`
        let mut dist: DiscreteProbabilityDistribution<i32> =
            DiscreteProbabilityDistribution::multinomial(probabilities);
        for _ in 1..n {
            dist = discrete_convolution(&dist, &dist);
        }
        dist
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

    pub fn convoluted_binomial(n: usize, probabilities: Vec<f64>) -> Self {
        //! Creates a new `DiscreteProbabilityDistribution` from a vector of
        //! probabilities by convoluting the distribution with itself `n` times.
        //!
        //! ## Arguments:
        //! * `n`: `usize`, number of convolutions
        //! * `probabilities`: `Vec<f64>`, vector of probabilities
        //!
        //! ## Returns:
        //! * `DiscreteProbabilityDistribution<i32>`
        let mut dist: DiscreteProbabilityDistribution<i32> =
            DiscreteProbabilityDistribution::binomial(probabilities[1]);
        for _ in 1..n {
            dist = discrete_convolution(&dist, &dist);
        }
        dist
    }
}

pub fn discrete_average_distributions(
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
