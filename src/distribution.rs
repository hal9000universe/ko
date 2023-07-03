#[derive(Clone, Debug)]
pub struct DiscreteProbabilityDistribution<T> {
    pub outcomes: Vec<T>,
    pub probabilities: Vec<f64>,
}

impl<T> DiscreteProbabilityDistribution<T> {
    pub fn new(outcomes: Vec<T>, probabilities: Vec<f64>) -> Self {
        assert_eq!(
            outcomes.len(),
            probabilities.len(),
            "outcomes and probabilities must have the same length"
        );
        assert!(
            probabilities.iter().all(|&p| p >= 0.),
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
        self.probabilities.clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Clone,
{
    pub fn outcomes(&self) -> Vec<T> {
        self.outcomes.clone()
    }
}

impl<T> DiscreteProbabilityDistribution<T>
where
    T: Eq,
{
    pub fn pmf(&self, x: T) -> f64 {
        match self.outcomes.iter().position(|y| y == &x) {
            Some(i) => self.probabilities[i],
            None => 0.,
        }
    }
}

impl DiscreteProbabilityDistribution<i32> {
    pub fn multinomial(probabilities: Vec<f64>) -> Self {
        let outcomes: Vec<i32> = (0..probabilities.len() as i32).collect();
        Self::new(outcomes, probabilities)
    }
}
