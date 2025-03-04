#[allow(unused_imports)]
use im::{hashmap, hashset, ordset, vector, HashMap, HashSet, OrdSet, Vector};
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
// use libm;
// use log::Level;
// use rpds::{list, List};
// extern crate petgraph;
use tap::prelude::*;
pub mod common;
pub use common::*;

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution1() {
        // assert_eq!(
        //     //_
        //     "\n\
        //     89010123\n\
        //     78121874\n\
        //     87430965\n\
        //     96549874\n\
        //     45678903\n\
        //     32019012\n\
        //     01329801\n\
        //     10456732\n\
        //     "
        //     .pipe(|s| solve_part1(s)),
        //     36,
        // );

        // panic!("display");
    }
}
