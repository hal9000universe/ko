const EPSILON: f64 = 1e-10;

pub fn softmax(x: &Vec<f64>) -> Vec<f64> {
    //! Softmax function
    //!
    //! ## Arguments:
    //! * `x`: `&Vec<f64>` - Input vector
    //!
    //! ## Returns:
    //! * `Vec<f64>` - Softmax of input vector
    let exp_x: Vec<f64> = x.iter().map(|x_i| x_i.exp() + EPSILON).collect();
    let sum_exp_x: f64 = exp_x.iter().sum();
    exp_x.iter().map(|exp_x_i| exp_x_i / sum_exp_x).collect()
}
