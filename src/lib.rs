#[allow(unused_imports)]
use im::{hashmap, hashset, ordset, vector, HashMap, HashSet, OrdSet, Vector};
// use itertools::FoldWhile::{Continue, Done};
// use itertools::Itertools;
// use libm;
// use log::Level;
// use rpds::{list, List};
// extern crate petgraph;
#[allow(unused_imports)]
use tap::prelude::*;

pub mod common;
pub mod plot;

pub use common::*;

pub mod util;
pub use util::*;


// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution1() {
    }
}
