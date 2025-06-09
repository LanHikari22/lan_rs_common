extern crate petgraph;
use im::{HashMap, Vector};
use petgraph::graph::Graph;
use petgraph_evcxr::draw_graph;
use tap::prelude::*;

/* \begin{graph} */

pub fn create_u32_directed_graph<I: Iterator<Item = (u32, u32)>>(edges: I) -> Graph<u32, String> {
    let mut g: Graph<u32, String> = Graph::new();
    let mut nodes_map: HashMap<u32, petgraph::graph::NodeIndex> = HashMap::new();

    for (head, tail) in edges {
        if !nodes_map.contains_key(&head) {
            let node_idx = g.add_node(head);
            nodes_map.insert(head, node_idx);
        }
        if !nodes_map.contains_key(&tail) {
            let node_idx = g.add_node(tail);
            nodes_map.insert(tail, node_idx);
        }

        g.add_edge(nodes_map[&tail], nodes_map[&head], "".to_string());
    }

    g
}

pub fn visualize_u32_directed_graph<I: Iterator<Item = (u32, u32)>>(edges: I) {
    create_u32_directed_graph(edges)
        .pipe(|g| draw_graph(&g))
}


#[cfg(feature = "use_ndarray")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DirectedGraph {
    /// From passed in edge values to internal node numbers
    pub lookup: HashMap<u32, u32>,
    /// From internal node numbers to numbers in passed in edge values
    pub rev_lookup: HashMap<u32, u32>,
    pub length: usize,
    pub adj_matrix: Array2<u32>,
}

#[cfg(feature = "use_ndarray")]
impl DirectedGraph {
    pub fn from_edges(edges: &Vector<(u32, u32)>) -> Self {
        let (lookup, rev_lookup, num_nodes) = //_
            Self::translate_u32_edge_values(edges);
        let adj_matrix = //_
            Self::create_adjacency_matrix(edges, &lookup, num_nodes as usize);

        DirectedGraph {
            lookup,
            rev_lookup,
            length: num_nodes as usize,
            adj_matrix,
        }
    }

    /// Translates edge values which can be arbitrary to ids for use of the adjacency matrix.
    /// Returns the lookup, reverse lookup, and the number of nodes
    pub fn translate_u32_edge_values(
        edges: &Vector<(u32, u32)>,
    ) -> (HashMap<u32, u32>, HashMap<u32, u32>, u32) {
        edges //_
            .into_iter()
            .fold(
                (hashmap! {}, hashmap! {}, 0),
                |(lookup, rev_lookup, last_unused_id), (tail, head)| {
                    vec![tail, head] //_
                        .into_iter()
                        .fold(
                            (lookup, rev_lookup, last_unused_id),
                            |(lookup, rev_lookup, last_unused_id), n| {
                                //_
                                match lookup.contains_key(n) {
                                    true => (lookup, rev_lookup, last_unused_id),
                                    false => (
                                        lookup.update(*n, last_unused_id),
                                        rev_lookup.update(last_unused_id, *n),
                                        last_unused_id + 1,
                                    ),
                                }
                            },
                        )
                },
            )
    }

    pub fn get_nodes(edges: &Vector<(u32, u32)>) -> Vector<u32> {
        edges
            .into_iter()
            .fold(vector![], |nodes: Vector<u32>, (head, tail)| {
                match nodes.contains(head) {
                    true => nodes,
                    false => &nodes + &vector![*head],
                }
                .pipe(|nodes| match nodes.contains(tail) {
                    true => nodes,
                    false => &nodes + &vector![*tail],
                })
            })
    }

    #[cfg(feature = "use_ndarray")]
    pub fn create_adjacency_matrix(
        edges: &Vector<(u32, u32)>,
        lookup: &HashMap<u32, u32>,
        length: usize,
    ) -> Array2<u32> {
        let mut result: Array2<u32> = Array2::<u32>::zeros((length, length).f());

        for (tail, head) in edges.iter() {
            result[(lookup[tail] as usize, lookup[head] as usize)] = 1;
            // for undirected:
            // result[(*head as usize, *tail as usize)] = 1;
        }

        result
    }

    #[cfg(feature = "use_ndarray")]
    pub fn adj_matrix_of_degree(&self, n: u32) -> Array2<u32> {
        (0..n - 1)
            .into_iter()
            .fold(self.adj_matrix.clone(), |acc, _i| //_
                acc.dot(&self.adj_matrix))
    }

    #[cfg(feature = "use_ndarray")]
    pub fn adj_matrix_of_degree_bin(&self, n: u32) -> Array2<u32> {
        (0..n - 1)
            .fold(self.adj_matrix.clone(), |acc, _i| {
                Self::to_bin_matrix(&acc.dot(&self.adj_matrix), self.length)
            })
    }

    #[cfg(feature = "use_ndarray")]
    pub fn adj_matrices_to_degree(&self, n: u32) -> Vector<Array2<u32>> {
        (0..n - 1)
            .into_iter()
            .fold_while(vector![self.adj_matrix.clone()], |acc, _i| {
                let is_zero = {
                    acc[acc.len()-1]
                        .iter()
                        .all(|x| *x == 0)
                };

                if is_zero {
                    Done(acc)
                } else {
                    let new_mat = {
                        if is_zero {
                            acc[acc.len()-1].clone()
                        } else {
                            acc[acc.len()-1].dot(&self.adj_matrix)
                        }
                    };

                    Continue(&acc + &vector![new_mat])
                }
            }).into_inner()
    }

    #[cfg(feature = "use_ndarray")]
    pub fn adj_matrices_to_degree_bin(&self, n: u32) -> Vector<Array2<u32>> {
        (0..n - 1)
            .into_iter()
            .fold(vector![self.adj_matrix.clone()], |acc, _i| {
                let new_mat = {
                    let is_zero = {
                        acc[acc.len()-1]
                            .iter()
                            .all(|x| *x == 0)
                    };

                    if is_zero {
                        acc[acc.len()-1].clone()
                    } else {
                        Self::to_bin_matrix(&acc[acc.len()-1].dot(&self.adj_matrix), self.length)
                    }
                };

                &acc + &vector![new_mat]
            })
    }

    #[cfg(feature = "use_ndarray")]
    /// Computes whether any 1-path...n-path exists between any two nodes
    pub fn adj_matrix_of_degree_up_to(&self, n: u32) -> Array2<u32> {
        (0..n - 1)
            .into_iter()
            .fold(self.adj_matrix.clone(), |acc, i| {
                acc + self.adj_matrix_of_degree_bin(i + 1)
                // .pipe(|acc| Self::to_bin_matrix(&acc, self.length))
            })
    }

    pub fn nodes_adjacent(&self, node: u32) -> Option<Arc<[u32]>> {
        if node >= self.length as u32 {
            None
        } else {
            self.adj_matrix
                .slice(s![node as usize, ..])
                .iter()
                .enumerate()
                .filter(|(_head, conn)| **conn == 1)
                .map(|(head, _conn)| head as u32)
                .collect::<Arc<[u32]>>()
                .pipe(|res| Some(res))

            // self.edges.iter()
            //     .filter(|(tail, _head)| *tail == node)
            //     .map(|(_tail, head)| *head)
            //     .collect::<Arc<[u32]>>()
            //     .pipe(|vec| Some(vec))
        }
    }

    #[cfg(feature = "use_ndarray")]
    pub fn to_bin_matrix(mat: &Array2<u32>, length: usize) -> Array2<u32> {
        let mut result = mat.clone();

        for i in 0..length {
            for j in 0..length {
                if mat[(i, j)] != 0 {
                    result[(i, j)] = 1;
                }
            }
        }

        result
    }

    pub fn get_node_degrees(&self, path_deg: u32, outgoing: bool) -> Vector<u32> {
        (0..self.length)
            .map(|node| {
                if outgoing {
                    self.adj_matrix_of_degree(path_deg)
                        .slice(s![node as usize, ..])
                        .sum() as u32
                } else {
                    self.adj_matrix_of_degree(path_deg)
                        .slice(s![.., node as usize])
                        .sum() as u32
                }
            })
            .collect::<Vector<u32>>()
    }

    pub fn get_translated_node_degrees(&self, path_deg: u32, outgoing: bool) -> HashMap<u32, u32> {
        self.get_node_degrees(path_deg, outgoing)
            .into_iter()
            .enumerate()
            .fold(hashmap! {}, |map, (node, deg)| {
                map.update(self.rev_lookup[&(node as u32)], deg)
            })
    }

    pub fn nodes_of_degree(&self, degree: u32) -> Arc<[u32]> {
        (0..self.length)
            .map(|node| (node, self.adj_matrix.slice(s![node as usize, ..]).sum()))
            .map(|(node, deg)| (node, deg as u32))
            .filter(|&(_node, deg)| deg == degree)
            .map(|(node, _deg)| node as u32)
            .collect::<Arc<[u32]>>()
    }

    pub fn find_all_node_islands(&self) -> Vec<OrdSet<u32>> {
        self.nodes_of_degree(1)
            .iter()
            .map(|&leaf| self.find_all_connected_nodes_to(ordset![], leaf))
            .sorted()
            .unique()
            .collect::<Vec<OrdSet<u32>>>()
    }

    pub fn find_all_connected_nodes_to(&self, prev: OrdSet<u32>, node: u32) -> OrdSet<u32> {
        prev.update(node).pipe(|prev| {
            self.nodes_adjacent(node)
                .unwrap()
                .iter()
                .fold(prev, |acc, &neighbor| {
                    if acc.contains(&neighbor) {
                        acc
                    } else {
                        self.find_all_connected_nodes_to(acc, neighbor)
                    }
                })
        })
    }

    pub fn visualize(&self) {
        let mut g: Graph<String, &str> = Graph::new();
        let mut nodes_map: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

        for tail in 0..self.length {
            for head in 0..self.length {
                if tail >= head {
                    continue;
                }

                if self.adj_matrix[(tail as usize, head as usize)] == 1 {
                    if !nodes_map.contains_key(&tail.to_string()) {
                        let node_idx = g.add_node(tail.to_string());
                        nodes_map.insert(tail.to_string(), node_idx);
                    }

                    if !nodes_map.contains_key(&head.to_string()) {
                        let node_idx = g.add_node(head.to_string());
                        nodes_map.insert(head.to_string(), node_idx);
                    }

                    g.add_edge(
                        nodes_map[&tail.to_string()],
                        nodes_map[&head.to_string()],
                        "",
                    );
                }
            }
        }

        draw_graph(&g);
    }

    #[cfg(feature = "use_ndarray")]
    pub fn init_path_reconstruction_context(&self) -> Vector<Array2<u32>> {
        let adj_matrices_per_deg = {
            self.adj_matrices_to_degree_bin(self.length as u32)
        };

        adj_matrices_per_deg
    }

    #[cfg(feature = "use_ndarray")]
    pub fn reconstruct_paths_between_nodes(
        &self,
        context: &Vector<Array2<u32>>,
        tail_inner: u32,
        head_inner: u32,
    ) -> Vector<Vector<u32>> {
        let adj_matrices_per_deg = //_
            context;

        let outgoing_connections_per_deg = {
            adj_matrices_per_deg
                .iter()
                .map(|adj_mat| {
                    adj_mat
                        .column(tail_inner as usize)
                        .into_iter()
                        .enumerate()
                        .filter(|(_head_inner, conn)| **conn > 0)
                        .map(|(head_inner, _conn)| head_inner)
                        .collect::<Vector<_>>()
                })
                .collect::<Vector<_>>()
        };

        let incoming_connections_per_deg = {
            adj_matrices_per_deg
                .iter()
                .map(|adj_mat| {
                    adj_mat
                        .row(head_inner as usize)
                        .into_iter()
                        .enumerate()
                        .filter(|(_tail_inner, conn)| **conn > 0)
                        .map(|(tail_inner, _conn)| tail_inner)
                        .collect::<Vector<_>>()
                })
                .collect::<Vector<_>>()
        };

        let path_degrees_to_head = {
            outgoing_connections_per_deg
                .iter()
                .enumerate()
                .filter(|(_deg, connections)| connections.contains(&(head_inner as usize)))
                .map(|(deg, _connections)| deg)
                .collect::<Vector<_>>()
        };

        let reconstructed_paths = {
            path_degrees_to_head //_
                .into_iter()
                .map(|k| {
                    // there exists at least one `k` long path from tail to head. We need to reconstruct
                    // the paths by making checks at each level (1,k-1), (2,k-2), ... (k-1,1) and seeing matches
                    // - note that 0 is to be interpreted as degree 1. For the outgoing then, k would be the tail
                    // which is known as a match. We coming i (0 being deg1 and on) with (k-1-i), being just after
                    // the guaranteed match with tail.
                    let matching_connections_per_deg = {
                        (0..k)
                            .map(|i| {
                                outgoing_connections_per_deg[i]
                                    .iter()
                                    .filter(|outgoing| {
                                        incoming_connections_per_deg[(k - 1) - i]
                                            .contains(*outgoing)
                                    })
                                    .collect::<Vector<_>>()
                            })
                            .collect::<Vector<_>>()
                    };

                    matching_connections_per_deg
                })
                .map(|matching_conections_per_deg| {
                    // go layer by layer, constructing a path
                    matching_conections_per_deg.into_iter().fold(
                        vector![vector![tail_inner]],
                        |paths, layer| {
                            // go through each path, and find if it has a connection to each node in `vec`, if so
                            // then duplicate this path with that connection
                            paths
                                .into_iter()
                                .map(|path| {
                                    let head = //_
                                        path[path.len()-1];

                                    let head_col = //_
                                        adj_matrices_per_deg[0]
                                            .column(head as usize);

                                    let head_connects_to_node_fn = |node| {
                                        head_col
                                            .into_iter()
                                            .enumerate()
                                            .any(|(to_node, conn)| to_node == node && *conn > 0)
                                    };

                                    layer
                                        .iter()
                                        .filter(|node| head_connects_to_node_fn(***node))
                                        .map(|node| &path + &vector![**node as u32])
                                        .collect::<Vector<_>>()
                                })
                                .fold(vector![], |acc, paths| &acc + &paths)
                        },
                    )
                })
                // collect together all paths from each deg
                .fold(vector![], |acc, paths| &acc + &paths)
                // Finally just make all paths connect to the head
                .into_iter()
                .map(|path| &path + &vector![head_inner])
                .collect::<Vector<_>>()
        };

        reconstructed_paths
    }
}

/* \end{graph} */