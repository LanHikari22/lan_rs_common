use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
};
pub use tap::prelude::*;

// Seems Hash derive is deep by default? so it's not safe, especially for cyclic graphs
#[derive(Debug, PartialEq, Eq, Clone)]
struct GraphNode<T> {
    data: T,
    targets: Vec<GraphNode<T>>,
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
                return Some(item);
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
        UniqueIter {
            iter: self.clone(),
            visited: HashSet::new(),
        }
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

            return Some(node);
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
        GraphBfsIter {
            stack: vec![self.clone()],
        }
    }
}

#[derive(Clone)]
struct GraphDfsIter<T> {
    stack: Vec<GraphNode<T>>,
}

impl<T: Clone> Iterator for GraphDfsIter<T> {
    type Item = GraphNode<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || self.stack.pop();

        while let Some(node) = strat_based_pop_fn() {
            // For next, work at the level of the node's targets
            node.targets
                .iter()
                .rev()
                .for_each(|target| self.stack.push(target.clone()));

            // Yield the last node
            return Some(node);
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
        GraphDfsIter {
            stack: vec![self.clone()],
        }
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
            node.targets.iter().enumerate().for_each(|(i, target)| {
                self.stack
                    .push((vec![id_branch.clone(), vec![i]].concat(), target.clone()))
            });

            let mut mut_new_branch = {
                self.branch
                    .clone()
                    .into_iter()
                    .filter(|(id_br2, _)| {
                        id_br2.len() < id_branch.len() && id_br2[..] == id_branch[..id_br2.len()]
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
                    .collect::<Vec<_>>(),
            );
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
        GraphBfsPathIter {
            stack: vec![(vec![0], self.clone())],
            branch: vec![],
        }
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
        let mut strat_based_pop_fn = || self.stack.pop();

        while let Some((id_branch, node)) = strat_based_pop_fn() {
            node.targets
                .iter()
                .rev()
                .enumerate()
                .for_each(|(i, target)| {
                    self.stack
                        .push((vec![id_branch.clone(), vec![i]].concat(), target.clone()))
                });

            let mut mut_new_branch = {
                self.branch
                    .clone()
                    .into_iter()
                    .filter(|(id_br2, _)| {
                        id_br2.len() < id_branch.len() && id_br2[..] == id_branch[..id_br2.len()]
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
                    .collect::<Vec<_>>(),
            );
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
        GraphDfsPathIter {
            stack: vec![(vec![0], self.clone())],
            branch: vec![],
        }
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
                            .map(|node| vec![node_per_branch.clone(), vec![node.clone()]].concat())
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
        GraphFrontierPathIter {
            stack: vec![vec![self.clone()]],
        }
    }
}

/// Checks if the final joint build of pairs of node hashes produces a singular connected whole or disjoint islands
fn check_node_hash_pair_per_n_joins_is_fragmented(
    node_hash_pair_per_n_joins: &Vec<Vec<(u64, u64)>>,
) -> bool {
    // we need to get all the available nodes
    let nodes = {
        node_hash_pair_per_n_joins
            .iter()
            .flat_map(|node_hash_pair_per_n_join| {
                node_hash_pair_per_n_join
                    .iter()
                    .flat_map(|(from, to)| vec![*from, *to])
            })
            .unique()
            .collect::<Vec<_>>()
    };

    if nodes.len() == 0 {
        return true;
    }

    // build an adjacency matrix using the joins
    let adj_mat = {
        let mut mut_adj_mat: HashMap<u64, HashMap<u64, bool>> = HashMap::new();

        nodes.iter().for_each(|node| {
            mut_adj_mat.insert(*node, HashMap::new());
        });

        nodes
            .iter()
            .cartesian_product(nodes.iter())
            .for_each(|(from, to)| {
                mut_adj_mat.get_mut(from).unwrap().insert(*to, false);
            });

        node_hash_pair_per_n_joins
            .iter()
            .for_each(|node_hash_pair_per_n_join| {
                node_hash_pair_per_n_join.iter().for_each(|(from, to)| {
                    mut_adj_mat.get_mut(from).unwrap().insert(*to, true);
                })
            });

        mut_adj_mat
    };

    // for each node, traverse the adjacency matrix using a stack-based depth-first search and note all visited nodes
    // if for any node the amount is less than # nodes, it is fragmented.
    let fragmented = {
        let mut mut_stack: Vec<u64> = vec![nodes[0]];
        let mut mut_nodes_visited: HashSet<u64> = HashSet::new();
        mut_nodes_visited.insert(nodes[0]);

        while mut_stack.len() != 0 {
            mut_stack.clone().iter().for_each(|from| {
                // find all tos for this from that have not been visited
                adj_mat[from].iter().for_each(|(to, conn)| {
                    if !mut_nodes_visited.contains(to) && *conn {
                        mut_nodes_visited.insert(*to);
                    }
                })
            })
        }

        mut_nodes_visited.len() != nodes.len()
    };

    fragmented
}

#[derive(Clone)]
struct ConnectedSubgraphsIter<T> {
    last_node_per_part_per_n_arrows: Vec<Vec<GraphNode<T>>>,
    node_per_part_per_1_arrow: Vec<Vec<GraphNode<T>>>,
    cur_num_arrows: usize,
    root: GraphNode<T>,
}

impl<T: Clone + Hash + TrShallowHash> Iterator for ConnectedSubgraphsIter<T> {
    type Item = Vec<Vec<GraphNode<T>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut strat_based_pop_fn = || {
            if self.last_node_per_part_per_n_arrows.is_empty() {
                None
            } else {
                Some(self.last_node_per_part_per_n_arrows.remove(0))
            }
        };

        if self.node_per_part_per_1_arrow.len() == 0 {
            // initialize all permutations
            self.node_per_part_per_1_arrow = {
                self.root
                    .bfs_iter()
                    .shallow_unique()
                    .filter(|node| node.targets.len() != 0)
                    .flat_map(|from_node| {
                        from_node
                            .targets
                            .iter()
                            .map(|to_node| vec![from_node.clone(), to_node.clone()])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
        };

        let next_layer_branches = {
            self.last_node_per_part_per_n_arrows
                .iter()
                .flat_map(|node_per_branch| {
                    let last_node = node_per_branch.last().unwrap();

                    let node_per_branch_per_target = {
                        last_node
                            .targets
                            .iter()
                            .map(|node| vec![node_per_branch.clone(), vec![node.clone()]].concat())
                            .collect::<Vec<_>>()
                    };

                    node_per_branch_per_target
                })
                .collect::<Vec<_>>()
        };

        let curr_layer = self.last_node_per_part_per_n_arrows.clone();

        self.last_node_per_part_per_n_arrows = next_layer_branches;

        if curr_layer.len() == 0 {
            None
        } else {
            Some(curr_layer)
        }
    }
}

trait TrConnectedSubgraphsIter {
    type Item: Clone;

    fn connected_subgraphs_iter(&self) -> Self::Item;
}

impl<T: Clone> TrConnectedSubgraphsIter for GraphNode<T> {
    type Item = ConnectedSubgraphsIter<T>;

    fn connected_subgraphs_iter(&self) -> Self::Item {
        ConnectedSubgraphsIter {
            last_node_per_part_per_n_arrows: vec![],
            node_per_part_per_1_arrow: vec![],
            cur_num_arrows: 1,
            root: self.clone(),
        }
    }
}

enum GraphWalkStrat {
    DFS,
    BFS,
    DfsPath,
    BfsPath,
    FrontierPath,
    ConnectedSubgraphs,
}

trait TrGraphWalkStrat {
    fn graph_walk_strat(&self) -> GraphWalkStrat;
}

impl<T: Clone> TrGraphWalkStrat for GraphBfsIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat {
        GraphWalkStrat::BFS
    }
}
impl<T: Clone> TrGraphWalkStrat for GraphDfsIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat {
        GraphWalkStrat::DFS
    }
}
impl<T: Clone> TrGraphWalkStrat for GraphBfsPathIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat {
        GraphWalkStrat::BfsPath
    }
}
impl<T: Clone> TrGraphWalkStrat for GraphFrontierPathIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat {
        GraphWalkStrat::FrontierPath
    }
}
impl<T: Clone> TrGraphWalkStrat for ConnectedSubgraphsIter<T> {
    fn graph_walk_strat(&self) -> GraphWalkStrat {
        GraphWalkStrat::ConnectedSubgraphs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_misc() {
    }

    #[test]
    fn test_frontier_path_iter_quick() {
        let d = GraphNode::<String> {
            data: "D".to_string(),
            targets: vec![],
        };

        let graph = {
            GraphNode::<String> {
                data: "A".to_string(),
                targets: vec![
                    GraphNode::<String> {
                        data: "B".to_string(),
                        targets: vec![d.clone()],
                    },
                    GraphNode::<String> {
                        data: "C".to_string(),
                        targets: vec![d.clone()],
                    },
                ],
            }
        };

        let s = format!(
            "{:?}",
            graph
                .frontier_path_iter()
                .map(|ns_per_branch| {
                    ns_per_branch
                        .into_iter()
                        .map(|ns| ns.into_iter().map(|n| n.data.clone()).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                // .map(|ns| ns.iter().map(|n| n.data.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );

        assert_eq!(s, r#"[[["A"]], [["A", "B"], ["A", "C"]], [["A", "B", "D"], ["A", "C", "D"]]]"#);
    }
}