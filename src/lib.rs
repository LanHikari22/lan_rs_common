#[cfg(feature = "incl_plot")]
pub mod plot;

#[cfg(feature = "incl_graph_search")]
pub mod graph_search;

#[cfg(feature = "incl_common")]
pub mod common;

#[cfg(feature = "incl_common")]
pub use common::*;


pub mod common_selective;

pub mod util;

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_solution1() {
    }
}
