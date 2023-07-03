#[macro_export]
macro_rules! joint_distribution {
    // joins n independent discrete probability distributions with integer outcomes
    ( $( $x:ident ),* ) => {
        {
            let mut distributions = Vec::new();
            // assemble distributions
            $(
                distributions.push($x.clone());
            )*
            // assemble outcomes and probabilities
            let mut outcomes = Vec::new();
            let mut probabilities = Vec::new();
            for dist in distributions.iter() {
                outcomes.push(dist.outcomes.clone());
                probabilities.push(dist.probabilities.clone());
            }
            let joint_outcomes = cartesian_product!(outcomes);
            let joint_probabilities = cartesian_product!(probabilities).into_iter().map(|x| { x.into_iter().fold(1., |prod, y| prod * y) }).collect();
            // return joint distribution
            DiscreteProbabilityDistribution::new(joint_outcomes, joint_probabilities)
        }
    };
}
