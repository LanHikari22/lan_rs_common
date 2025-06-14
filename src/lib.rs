#![allow(unused_imports)]
#![allow(dead_code)]

use im::{hashmap, hashset, ordset, vector, HashMap, HashSet, OrdSet, Vector};
use tap::prelude::*;

pub mod common;
pub mod plot;
pub mod graph_search;

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
