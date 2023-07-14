use std::{
    f64::consts::E,
    ops::{Add, Sub},
};

#[derive(Debug, Copy, Clone)]
pub enum InformationUnit {
    Bit(f64),
    Nat(f64),
}

impl InformationUnit {
    pub fn to_bits(&self) -> InformationUnit {
        //! Converts `self` to bits
        match self {
            InformationUnit::Bit(x) => InformationUnit::Bit(*x),
            InformationUnit::Nat(x) => InformationUnit::Bit(x * E.log2()),
        }
    }

    pub fn to_nats(&self) -> InformationUnit {
        //! Converts `self` to nats
        match self {
            InformationUnit::Bit(x) => InformationUnit::Nat(x * 2f64.ln()),
            InformationUnit::Nat(x) => InformationUnit::Nat(*x),
        }
    }

    pub fn to_float(&self) -> f64 {
        //! Returns the value assigned to `self` as f64
        match self {
            InformationUnit::Bit(x) => *x,
            InformationUnit::Nat(x) => *x,
        }
    }

    pub fn apply(&self, func: impl Fn(f64) -> f64) -> InformationUnit {
        //! Applies a transformation to the value assigned to `self`
        //!
        //! ## Arguments:
        //! * `func`: `impl Fn(f64) -> f64`
        //!
        //! ## Returns:
        //! * an `InformationUnit` the value of which is the transformed value of `self`
        match self {
            InformationUnit::Bit(x) => InformationUnit::Bit(func(*x)),
            InformationUnit::Nat(x) => InformationUnit::Nat(func(*x)),
        }
    }
}

impl Add for InformationUnit {
    type Output = InformationUnit;

    fn add(self, other: InformationUnit) -> InformationUnit {
        match (self, other) {
            (InformationUnit::Bit(x), InformationUnit::Bit(y)) => InformationUnit::Bit(x + y),
            (InformationUnit::Nat(x), InformationUnit::Nat(y)) => InformationUnit::Nat(x + y),
            (InformationUnit::Bit(x), InformationUnit::Nat(y)) => {
                InformationUnit::Bit(x + y * E.log2())
            }
            (InformationUnit::Nat(x), InformationUnit::Bit(y)) => {
                InformationUnit::Bit(x * E.log2() + y)
            }
        }
    }
}

impl Sub for InformationUnit {
    type Output = InformationUnit;

    fn sub(self, other: InformationUnit) -> InformationUnit {
        match (self, other) {
            (InformationUnit::Bit(x), InformationUnit::Bit(y)) => InformationUnit::Bit(x - y),
            (InformationUnit::Nat(x), InformationUnit::Nat(y)) => InformationUnit::Nat(x - y),
            (InformationUnit::Bit(x), InformationUnit::Nat(y)) => {
                InformationUnit::Bit(x - y / 2f64.ln())
            }
            (InformationUnit::Nat(x), InformationUnit::Bit(y)) => {
                InformationUnit::Bit(x / 2f64.ln() - y)
            }
        }
    }
}
