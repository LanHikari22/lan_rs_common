// fn main() {
// }

use itertools::Itertools;
// include!("experiments/graph_recursion_pipeline1.rs");
pub use lan_rs_common::*;
pub use tap::prelude::*;
use std::{collections::HashSet, hash::{DefaultHasher, Hash, Hasher}};

// Seems Hash derive is deep by default? so it's not safe, especially for cyclic graphs
#[derive(Debug, PartialEq, Eq, Clone)]
struct GraphNode<T> {
    data: T,
    targets: Vec<GraphNode<T>>
}

trait TrShallowHash {
    fn shallow_hash(&self) -> u64;
}

impl<T: Hash> TrShallowHash for GraphNode<T> {
    fn shallow_hash(&self) -> u64 {
        let mut mut_hasher = DefaultHasher::new();
        self.data.hash(&mut mut_hasher);
        mut_hasher.finish()
    }
}

#[derive(Clone)]
struct UniqueIter<I> {
    iter: I,
    visited: HashSet<u64>,
}

impl<I, T> Iterator for UniqueIter<I>
where 
    I: Iterator<Item = T>,
    T: TrShallowHash,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.iter.next() {
            let hash = item.shallow_hash();
            if self.visited.insert(hash) {
                return Some(item)
            }
        }

        None
    }
}


trait TrShallowUnique {
    type Item: Clone;
    fn shallow_unique(&self) -> Self::Item;
}

impl<I> TrShallowUnique for I
where 
    I: Iterator + Clone,
    I::Item: TrShallowHash,
{
    type Item = UniqueIter<I>;
    
    fn shallow_unique(&self) -> Self::Item {
        UniqueIter { iter: self.clone(), visited: HashSet::new() }
    }
}


#[derive(Clone)]
struct GraphBfsIter<T> {
    stack: Vec<GraphNode<T>>,
}

impl<T: Clone> Iterator for GraphBfsIter<T> {
    type Item = GraphNode<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || {
            if self.stack.is_empty() {
                None
            } else {
                Some(self.stack.remove(0))
            }
        };

        while let Some(node) = strat_based_pop_fn() {
            node.targets
                .iter()
                .for_each(|target| self.stack.push(target.clone()));

            return Some(node)
        }

        None
    }
}

trait TrBfsIter {
    type Item: Clone;

    fn bfs_iter(&self) -> Self::Item;
}

impl<T: Clone> TrBfsIter for GraphNode<T> {
    type Item = GraphBfsIter<T>;

    fn bfs_iter(&self) -> Self::Item {
        GraphBfsIter { stack: vec![self.clone()], }
    }
}



#[derive(Clone)]
struct GraphDfsIter<T> {
    stack: Vec<GraphNode<T>>,
}

impl<T: Clone> Iterator for GraphDfsIter<T> {
    type Item = GraphNode<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || {
            self.stack.pop()
        };

        while let Some(node) = strat_based_pop_fn() {
            // For next, work at the level of the node's targets
            node.targets
                .iter()
                .rev()
                .for_each(|target| self.stack.push(target.clone()));

            // Yield the last node
            return Some(node)
        }

        None
    }
}

trait TrDfsIter {
    type Item: Clone;

    fn dfs_iter(&self) -> Self::Item;
}

impl<T: Clone> TrDfsIter for GraphNode<T> {
    type Item = GraphDfsIter<T>;

    fn dfs_iter(&self) -> Self::Item {
        GraphDfsIter { stack: vec![self.clone()], }
    }
}

#[derive(Clone)]
struct GraphBfsPathIter<T> {
    stack: Vec<(Vec<usize>, GraphNode<T>)>,
    branch: Vec<(Vec<usize>, GraphNode<T>)>,
}

impl<T: Clone> Iterator for GraphBfsPathIter<T> {
    type Item = Vec<GraphNode<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || {
            if self.stack.is_empty() {
                None
            } else {
                Some(self.stack.remove(0))
            }
        };

        while let Some((id_branch, node)) = strat_based_pop_fn() {
            node.targets
                .iter()
                .enumerate()
                .for_each(|(i, target)| {
                    self.stack.push(
                        (
                            vec![id_branch.clone(), vec![i]].concat(), 
                            target.clone()
                        )
                    )
                });

            let mut mut_new_branch = {
                self.branch
                    .clone()
                    .into_iter()
                    .filter(|(id_br2, _)| {
                        id_br2.len() < id_branch.len() &&
                        id_br2[..] == id_branch[..id_br2.len()]
                    })
                    .collect::<Vec<_>>()
            };

            mut_new_branch.push((id_branch.clone(), node.clone()));
            self.branch.push((id_branch, node));

            return Some(
                mut_new_branch
                    .clone()
                    .into_iter()
                    .map(|(_, node)| node)
                    .collect::<Vec<_>>()
            )
        }

        None
    }
}

trait TrBfsPathIter {
    type Item: Clone;

    fn bfs_path_iter(&self) -> Self::Item;
}

impl<T: Clone> TrBfsPathIter for GraphNode<T> {
    type Item = GraphBfsPathIter<T>;

    fn bfs_path_iter(&self) -> Self::Item {
        GraphBfsPathIter { stack: vec![(vec![0], self.clone())], branch: vec![], }
    }
}

#[derive(Clone)]
struct GraphDfsPathIter<T> {
    stack: Vec<(Vec<usize>, GraphNode<T>)>,
    branch: Vec<(Vec<usize>, GraphNode<T>)>,
}

impl<T: Clone> Iterator for GraphDfsPathIter<T> {
    type Item = Vec<GraphNode<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || {
            self.stack.pop()
        };

        while let Some((id_branch, node)) = strat_based_pop_fn() {
            node.targets
                .iter()
                .rev()
                .enumerate()
                .for_each(|(i, target)| {
                    self.stack.push(
                        (
                            vec![id_branch.clone(), vec![i]].concat(), 
                            target.clone()
                        )
                    )
                });

            let mut mut_new_branch = {
                self.branch
                    .clone()
                    .into_iter()
                    .filter(|(id_br2, _)| {
                        id_br2.len() < id_branch.len() &&
                        id_br2[..] == id_branch[..id_br2.len()]
                    })
                    .collect::<Vec<_>>()
            };

            mut_new_branch.push((id_branch.clone(), node.clone()));
            self.branch.push((id_branch, node));

            return Some(
                mut_new_branch
                    .clone()
                    .into_iter()
                    .map(|(_, node)| node)
                    .collect::<Vec<_>>()
            )
        }

        None
    }
}

trait TrDfsPathIter {
    type Item: Clone;

    fn dfs_path_iter(&self) -> Self::Item;
}

impl<T: Clone> TrDfsPathIter for GraphNode<T> {
    type Item = GraphDfsPathIter<T>;

    fn dfs_path_iter(&self) -> Self::Item {
        GraphDfsPathIter { stack: vec![(vec![0], self.clone())], branch: vec![], }
    }
}

#[derive(Clone)]
struct GraphFrontierPathIter<T> {
    stack: Vec<Vec<GraphNode<T>>>,
}

impl<T: Clone> Iterator for GraphFrontierPathIter<T> {
    type Item = Vec<Vec<GraphNode<T>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || {
            if self.stack.is_empty() {
                None
            } else {
                Some(self.stack.remove(0))
            }
        };

        let next_layer_branches = {
            self.stack
                .iter()
                .flat_map(|node_per_branch| {
                    let last_node = node_per_branch.last().unwrap();

                    let node_per_branch_per_target = {
                        last_node
                            .targets
                            .iter()
                            .map(|node| {
                                vec![node_per_branch.clone(), vec![node.clone()]].concat()
                            })
                            .collect::<Vec<_>>()
                    };

                    node_per_branch_per_target
                })
                .collect::<Vec<_>>()
        };

        let curr_layer = self.stack.clone();

        self.stack = next_layer_branches;

        if curr_layer.len() == 0 {
            None
        } else {
            Some(curr_layer)
        }
    }
}

trait TrFrontierPathIter {
    type Item: Clone;

    fn frontier_path_iter(&self) -> Self::Item;
}

impl<T: Clone> TrFrontierPathIter for GraphNode<T> {
    type Item = GraphFrontierPathIter<T>;

    fn frontier_path_iter(&self) -> Self::Item {
        GraphFrontierPathIter { stack: vec![vec![self.clone()]], }
    }
}



enum GraphWalkStrat {
    DFS,
    BFS,
    Frontier,
    DfsPath,
    BfsPath,
}

trait TrGraphWalkStrat {
    fn graph_walk_strat(&self) -> GraphWalkStrat;
}

impl<T: Clone> TrGraphWalkStrat for GraphBfsIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat { GraphWalkStrat::BFS }
}
impl<T: Clone> TrGraphWalkStrat for GraphDfsIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat { GraphWalkStrat::DFS }
}

impl<T: Clone> TrGraphWalkStrat for GraphBfsPathIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat { GraphWalkStrat::BfsPath }
}



fn main() {
    let d = GraphNode::<String> { data: "D".to_string(), targets: vec![]};

    let graph= {
        GraphNode::<String> { data: "A".to_string(), targets: vec![
            GraphNode::<String> { data: "B".to_string(), targets: vec![d.clone()]},
            GraphNode::<String> { data: "C".to_string(), targets: vec![d.clone()]},
        ]}
    };

    println!("{:?}", 
        graph
            .frontier_path_iter()
            .map(|ns_per_branch| {
                ns_per_branch
                    .into_iter()
                    .map(|ns| {
                        ns
                            .into_iter()
                            .map(|n| n.data.clone())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            // .map(|ns| ns.iter().map(|n| n.data.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
}