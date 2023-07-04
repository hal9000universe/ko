//! Cartesian product of vectors.
//!
//! # Example 1
//! ```
//! use ko::cartesian_product;
//!
//! // vectors to be multiplied
//! let a = vec![1, 2];
//! let b = vec![4, 5];
//!
//! // calculate cartesian product
//! let product = cartesian_product!(a, b);
//!
//! // check result
//! assert_eq!(product, vec![
//!   vec![1, 4], vec![2, 4],
//!  vec![1, 5], vec![2, 5],
//! ]);
//! ```
//!
//! # Example 2
//! ```
//! use ko::cartesian_product;
//!
//! // vectors to be multiplied
//! let a = vec![1, 2];
//! let b = vec![4, 5];
//!
//! // concatenate vectors
//! let vectors = vec![a, b];
//!
//! // calculate cartesian product
//! let product = cartesian_product!(vectors);
//!
//! // check result
//! assert_eq!(product, vec![
//!   vec![1, 4], vec![2, 4],
//!  vec![1, 5], vec![2, 5],
//! ]);
//! ```
#[macro_export]
macro_rules! cartesian_product {
    ( $vectors:ident ) => {
        {
            // number of elements in final product
            let num_elements = $vectors.iter().fold(1, |acc, x| acc * x.len());
            // number of sets to be multiplied
            let num_sets = $vectors.len();
            // floors for indexing
            let mut floors: Vec<usize> = vec![1];
            for i in 0..num_sets - 1 {
                floors.push(floors[floors.len() - 1] * $vectors[i].len());
            }
            // set to be returned
            let mut product = Vec::new();
            for i in 0..num_elements {
                // add elements to product
                let mut element = Vec::new();
                for j in 0..num_sets {
                    // calculate index of sub-element via bijection
                    let index: usize = (i / floors[j]) % $vectors[j].len();
                    // access sub-element
                    let sub_element = $vectors[j][index];
                    // push sub-element to element
                    element.push(sub_element);
                }
                // push element to product
                product.push(element);
            }
        // return cartesian product
        product
        }
    };

    ( $( $vectors:ident ),* ) => {
        {
            let vectors = vec![$($vectors),*];
            // number of elements in final product
            let num_elements = vectors.iter().fold(1, |acc, x| acc * x.len());
            // number of sets to be multiplied
            let num_sets = vectors.len();
            // floors for indexing
            let mut floors: Vec<usize> = vec![1];
            for i in 0..num_sets - 1 {
                floors.push(floors[floors.len() - 1] * vectors[i].len());
            }
            // set to be returned
            let mut product = Vec::new();
            for i in 0..num_elements {
                // add elements to product
                let mut element = Vec::new();
                for j in 0..num_sets {
                    // calculate index of sub-element via bijection
                    let index: usize = (i / floors[j]) % vectors[j].len();
                    // access sub-element
                    let sub_element = vectors[j][index];
                    // push sub-element to element
                    element.push(sub_element);
                }
                // push element to product
                product.push(element);
            }
        // return cartesian product
        product
        }
    };
}
