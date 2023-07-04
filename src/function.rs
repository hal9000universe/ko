//! This module contains the `Function` trait and its implementations.
//! The `Function` trait is used to represent a function from one set to another.
//!
//! # Example Continuous Function
//! ```
//! use ko::function::{ContinuousFunction, Function};
//!
//! let domain = vec![0.0, 1.0];
//! let range = vec![0.0, 1.0];
//! let func = |x: &f64| *x;
//! let f = ContinuousFunction::new(domain, range, func);
//! ```
//!
//! # Example Discrete Function
//! ```
//! use ko::function::{DiscreteFunction, Function};
//!
//! let domain = vec![0, 1];
//! let func = |x: &i32| *x;
//! let f = DiscreteFunction::new(domain, func);
//! ```

use std::hash::Hash;
use std::ops::Add;

pub trait Function<T, S> {
    fn domain(&self) -> Vec<T>;
    fn range(&self) -> Vec<S>;
    fn eval(&self, x: &T) -> S;
    fn measure(&self, x: &Vec<T>) -> S;
}

#[derive(Clone)]
pub struct ContinuousFunction {
    domain: Vec<f64>,
    range: Vec<f64>,
    func: fn(&f64) -> f64,
    step_size: f64,
}

impl ContinuousFunction {
    pub fn new(domain: Vec<f64>, range: Vec<f64>, func: fn(&f64) -> f64) -> Self {
        //! Constructs a new `ContinuousFunction` from a domain, range, and function.
        //! The domain and range are represented by intervals.
        //!
        //! # Arguments
        //! * `domain` - The domain of the function.
        //! * `range` - The range of the function.
        //! * `func` - The function.
        //!
        //! # Returns
        //! A new `ContinuousFunction`.
        //!
        //! # Examples
        //!
        //! ```
        //! use ko::function::ContinuousFunction;
        //!
        //! let domain = vec![0.0, 1.0];
        //! let range = vec![0.0, 1.0];
        //! let func = |x: &f64| *x;
        //! let f = ContinuousFunction::new(domain, range, func);
        //! ```
        //!
        //! # Panics
        //!
        //! Panics if the domain and range are not intervals.
        assert_eq!(domain.len(), 2);
        assert_eq!(range.len(), 2);
        let step_size = 1e-6;
        Self {
            domain,
            range,
            func,
            step_size,
        }
    }

    pub fn with_set_step_size(&mut self, step_size: f64) -> Self {
        //! Sets the step size of the function and returns a clone of the function.
        //!
        //! # Arguments
        //! * `step_size` - The step size.
        //!
        //! # Returns
        //! A clone of the function with the step size set.
        //!
        //! # Examples
        //!
        //! ```
        //! use ko::function::ContinuousFunction;
        //!
        //! let domain = vec![0.0, 1.0];
        //! let range = vec![0.0, 1.0];
        //! let func = |x: &f64| *x;
        //! let mut f = ContinuousFunction::new(domain, range, func);
        //! f = f.with_set_step_size(1e-1);
        //! ```
        self.step_size = step_size;
        self.clone()
    }
}

impl Function<f64, f64> for ContinuousFunction {
    fn domain(&self) -> Vec<f64> {
        //! Returns a clone of the domain of the function.
        self.domain.clone()
    }

    fn range(&self) -> Vec<f64> {
        //! Returns a clone of the range of the function.
        self.range.clone()
    }

    fn eval(&self, x: &f64) -> f64 {
        //! Evaluates the function at a given point.
        //!
        //! # Arguments
        //! * `x` - The point.
        //!
        //! # Returns
        //! The value of the function at the point.
        //!
        //! # Panics
        //!
        //! Panics if the point is not in the domain.
        assert!(x >= &self.domain[0] && x <= &self.domain[1]);
        (self.func)(&x)
    }

    fn measure(&self, domain: &Vec<f64>) -> f64 {
        //! Measures the function over a given interval.
        //!
        //! # Arguments
        //! * `x` - The interval.
        //!
        //! # Returns
        //! The measure of the function over the interval.
        //!
        //! # Examples
        //!
        //! ```
        //! use ko::function::{Function, ContinuousFunction};
        //!
        //! let domain = vec![0.0, 1.0];
        //! let range = vec![0.0, 1.0];
        //! let func = |x: &f64| *x;
        //! let mut identity_function = ContinuousFunction::new(domain, range, func);
        //! let x = vec![0.0, 1.0];
        //! let measure = identity_function.with_set_step_size(1e-6).measure(&x);
        //! assert!((measure - 0.5).abs() < 1e-4);
        //! ```
        //!
        //! # Panics
        //!
        //! Panics if the interval is not an interval.

        // assert that domain is an interval
        assert_eq!(domain.len(), 2);
        assert!(domain[0] < domain[1]);

        // measure function over interval
        let mut measure = 0.0;
        let mut x = domain[0];
        while x < domain[1] {
            if x + self.step_size < domain[1] {
                measure += self.step_size * self.eval(&x);
                x += self.step_size;
            } else {
                measure += (domain[1] - x) * self.eval(&domain[1]);
                break;
            }
        }
        measure
    }
}

#[derive(Clone)]
pub struct DiscreteFunction<T, S>
where
    T: Copy,
    S: Copy,
{
    domain: Vec<T>,
    range: Vec<S>,
    func: fn(&T) -> S,
}

impl<T, S> DiscreteFunction<T, S>
where
    T: Copy + Eq + Hash,
    S: Copy + Eq + Hash,
{
    pub fn new(domain: Vec<T>, func: fn(&T) -> S) -> Self {
        //! Constructs a new `DiscreteFunction` from a domain, range, and function.
        //! The domain and range are represented by sets.
        //!
        //! # Arguments
        //! * `domain` - The domain of the function.
        //! * `func` - The function.
        //!
        //! # Returns
        //! A new `DiscreteFunction`.
        //!
        //! # Examples
        //!
        //! ```
        //! use ko::function::DiscreteFunction;
        //!
        //! let domain = vec![0, 1];
        //! let func = |x: &i32| *x;
        //! let f = DiscreteFunction::new(domain, func);
        //! ```
        //!
        //! # Panics
        //!
        //! Panics if the domain is not a set.

        // assert that domain is a set of unique elements
        assert!(
            domain.len()
                == domain
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
        );

        // map domain to range and reduce to unique elements hash set
        let range: Vec<S> = domain
            .iter()
            .map(func)
            .collect::<std::collections::HashSet<S>>()
            .into_iter()
            .collect::<Vec<S>>();
        Self {
            domain,
            range,
            func,
        }
    }
}

impl<T, S> Function<T, S> for DiscreteFunction<T, S>
where
    T: Copy + Eq + Hash,
    S: Copy + Add<Output = S>,
    i32: Into<S>,
{
    fn domain(&self) -> Vec<T> {
        //! Returns a clone of the domain of the function.
        self.domain.clone()
    }

    fn range(&self) -> Vec<S> {
        //! Returns a clone of the range of the function.
        self.range.clone()
    }

    fn eval(&self, x: &T) -> S {
        //! Evaluates the function at a given point.
        //!
        //! # Arguments
        //! * `x` - The point.
        //!
        //! # Returns
        //! The value of the function at the point.
        //!
        //! # Panics
        //!
        //! Panics if the point is not in the domain.
        //!
        //! # Examples
        //!
        //! ```
        //! use ko::function::{Function, DiscreteFunction};
        //!
        //! let domain = vec![0, 1, 2];
        //! let func = |x: &i32| *x;
        //! let mut identity_function = DiscreteFunction::new(domain, func);
        //! let x = 1;
        //! let y = identity_function.eval(&x);
        //! assert_eq!(y, 1);
        //! ```
        assert!(self.domain.contains(x));
        (self.func)(&x)
    }

    fn measure(&self, domain: &Vec<T>) -> S {
        //! Measures the function over a given set.
        //!
        //! # Arguments
        //! * `x` - The set.
        //!
        //! # Returns
        //! The measure of the function over the set.
        //!
        //! # Examples
        //!
        //! ```
        //! use ko::function::{Function, DiscreteFunction};
        //!
        //! let domain = vec![0, 1, 2];
        //! let func = |x: &i32| *x;
        //! let mut identity_function = DiscreteFunction::new(domain, func);
        //! let x = vec![0, 1];
        //! let measure = identity_function.measure(&x);
        //! assert_eq!(measure, 1);
        //! ```
        //!
        //! # Panics
        //!
        //! Panics if the set is not a set.
        assert!(
            domain.len()
                == domain
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
        );
        domain
            .iter()
            .fold((0).into(), |acc: S, x: &T| acc + self.eval(&x))
    }
}
