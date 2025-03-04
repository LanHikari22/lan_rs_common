#[allow(unused_imports)]
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
// use petgraph::Directed;
// use rand::seq::SliceRandom;
#[allow(unused_imports)]
use im::{hashmap, hashset, ordset, vector, HashMap, HashSet, OrdSet, Vector};
#[allow(unused_imports)]
use rpds::{list, List};
extern crate petgraph;
use petgraph::graph::Graph;
// use petgraph::dot::Dot;
// use petgraph_evcxr::{draw_dot};
use ndarray::{s, Array2, ShapeBuilder};
use petgraph_evcxr::draw_graph;
use reqwest::blocking::Client;
// use std::collections::HashMap;
use bitflags::bitflags;
use libm;
use polars::prelude::*;
use regex::Regex;
use std::fs::File;
use std::io::prelude::*; // needed for traits like the Read trait
use std::io::{self, Error};
use std::path::Path;
use std::sync::Arc;
use tap::prelude::*;

/*** IO  ***/

pub fn read_user_input(prompt: &str) -> String {
    let mut input = String::new();

    if prompt != "" {
        // eprint because it's not line-buffered like print, prints to stderr
        eprint!("{0}: ", prompt);
    }

    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input line");

    input.trim().to_string()
}

pub fn read_user_input_from_file(input_path: &str) -> String {
    let path = Path::new(input_path);
    let mut file = File::open(&path).unwrap();
    let mut out = String::new();

    file.read_to_string(&mut out).unwrap();

    out
}

pub fn write_to_file(s: &str, output_path: &str) -> Result<usize, Error> {
    let path = Path::new(output_path);
    let mut file = File::create(&path).unwrap();

    file.write(s.as_bytes())
}

pub fn parse_fixed_i32_tuple_lines(
    s: &str,
    tuple_size: usize,
) -> Result<Vector<Vector<i32>>, String> {
    s.trim()
        .lines()
        .map(|line| {
            line.split_ascii_whitespace()
                .collect::<Vector<_>>()
                .pipe(|tokens| match tokens.len() == tuple_size {
                    true => Ok(tokens),
                    false => Err(format!("Invalid length {tuple_size} for line {line}")),
                })
                .and_then(|tokens| {
                    tokens
                        .iter()
                        .map(|&num_s| {
                            num_s
                                .parse::<i32>()
                                .or(Err(format!("{num_s} is not an i32")))
                        })
                        .collect::<Result<Vector<_>, String>>()
                })
        })
        .collect::<Result<Vector<_>, String>>()
}

pub fn parse_i32_tuple_lines(s: &str) -> Result<Vector<Vector<i32>>, String> {
    s.trim()
        .lines()
        .map(|line| {
            line.split_ascii_whitespace()
                .collect::<Vector<_>>()
                .pipe(|tokens| {
                    tokens
                        .iter()
                        .map(|&num_s| {
                            num_s
                                .parse::<i32>()
                                .or(Err(format!("{num_s} is not an i32")))
                        })
                        .collect::<Result<Vector<_>, String>>()
                })
        })
        .collect::<Result<Vector<_>, String>>()
}

pub fn get_square_buf_length(s: &str) -> Option<usize> {
    libm::sqrtf(s.len() as f32)
        // .tap(|res| println!("sqrtf of len is: {res} and s.len() is {}", s.len()))
        .pipe(|f| {
            if has_decimals(f) {
                None
            } else {
                Some(f as usize)
            }
        })
}

pub fn read_square_char_input(s: &str) -> (usize, String) {
    s.trim()
        .replace("\n", "")
        .pipe(|s| (get_square_buf_length(&s).expect("Must have square buf"), s))
}

/*** /IO  ***/

/*** regex  ***/

pub fn regex_capture_once(s: &str, re: &Regex) -> Result<Vector<String>, String> {
    re.pipe(|re| {
        re.captures(&s)
            .pipe(|x| x)
            .and_then(|captures| captures.iter().collect::<Option<Vector<_>>>())
    })
    .pipe(|res| match res {
        Some(res) => Ok(res),
        None => Err(format!("Failed to capture")),
    })?
    .into_iter()
    .map(|m| m.as_str().to_string())
    .collect::<Vector<_>>()
    .pipe(|res| Ok(res))
}

pub fn regex_captures(s: &str, regex_s: &str) -> Result<Vector<Vector<String>>, String> {
    Regex::new(regex_s)
        .or(Err(format!("Failed to compile regex")))?
        .pipe(|re| {
            re.captures_iter(&s)
                .map(|captures| captures.iter().collect::<Option<Vector<_>>>())
                .collect::<Option<Vector<Vector<_>>>>()
        })
        .pipe(|res| match res {
            Some(res) => Ok(res),
            None => Err(format!("Failed to capture")),
        })?
        .into_iter()
        .map(|ms| {
            ms.into_iter()
                .map(|m| m.as_str().to_string())
                .collect::<Vector<_>>()
        })
        .collect::<Vector<Vector<_>>>()
        .pipe(|res| Ok(res))
}

/// This does not assume exhaustive scanning, so if the handlers fail it advances by 1
pub fn process_using_scanners<T: Clone>(
    s: &str,
    try_scan_fn: impl Fn(&str) -> Option<(T, usize)>,
) -> Vector<T> {
    let mut result = vector![];
    let mut buf = &s[..];

    while buf.len() != 0 {
        if let Some((item, advance)) = try_scan_fn(buf) {
            result.push_back(item);
            // println!("YES {buf}");
            buf = &buf[advance..];
        } else {
            // the scanners failed to yield a result, so we advance once by default as this is not
            // exhaustive (meaning we might find valid items at different offsets but not all)
            // println!("NAH {buf}");
            buf = &buf[1..];
        }
    }

    result
}

/// impl Fn and move |s| for returning closures. See https://stackoverflow.com/a/38947708
/// builder_fn should just create a new value. Since it is inside the move Box<dyn Fn> it is moved by
/// value and required guarantee that it doesn't have references that may not outlive it
pub fn init_regex_scanner<T: Clone>(
    regex: &str,
    builder_fn: impl Fn(&Vector<String>) -> T + 'static,
) -> Box<dyn Fn(&str) -> Option<(T, usize)>> {
    Regex::new(regex)
        .unwrap() // This is to avoid recompiling the regex which can be expensive
        .pipe(|re| {
            Box::new(move |s: &str| -> Option<(T, usize)> {
                regex_capture_once(s, &re)
                    .pipe(|res| match res {
                        Ok(res) => Some(res),
                        Err(_) => None,
                    })?
                    .pipe(|capture| (builder_fn(&capture), capture[0].len()))
                    .pipe(|res| Some(res))
            })
        })
}

pub fn chain_scanners<T: Clone>(
    s: &str,
    scanners: &Vec<impl Fn(&str) -> Option<(T, usize)>>,
) -> Option<(T, usize)> {
    scanners
        .iter()
        .fold(None, |res: Option<(T, usize)>, scanner| match res {
            Some(res) => Some(res),
            None => scanner(s),
        })
}

/*** /regex  ***/

/*** graph  ***/

pub fn create_u32_directed_graph(edges: &Vector<(u32, u32)>) -> Graph<u32, &str> {
    let mut g: Graph<u32, &str> = Graph::new();
    let mut nodes_map: HashMap<u32, petgraph::graph::NodeIndex> = HashMap::new();

    for (head, tail) in edges {
        if !nodes_map.contains_key(&head) {
            let node_idx = g.add_node(*head);
            nodes_map.insert(*head, node_idx);
        }
        if !nodes_map.contains_key(&tail) {
            let node_idx = g.add_node(*tail);
            nodes_map.insert(*tail, node_idx);
        }

        g.add_edge(nodes_map[&tail], nodes_map[&head], "");
    }

    g
}

pub fn visualize_u32_directed_graph(edges: Vector<(u32, u32)>) {
    create_u32_directed_graph(&edges).pipe(|g| draw_graph(&g))
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DirectedGraph {
    /// From passed in edge values to internal node numbers
    pub lookup: HashMap<u32, u32>,
    /// From internal node numbers to numbers in passed in edge values
    pub rev_lookup: HashMap<u32, u32>,
    pub length: usize,
    pub adj_matrix: Array2<u32>,
}

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

    pub fn adj_matrix_of_degree(&self, n: u32) -> Array2<u32> {
        (0..n - 1)
            .into_iter()
            .fold(self.adj_matrix.clone(), |acc, _i| //_
                acc.dot(&self.adj_matrix))
    }

    pub fn adj_matrix_of_degree_bin(&self, n: u32) -> Array2<u32> {
        (0..n - 1)
            .fold(self.adj_matrix.clone(), |acc, _i| {
                Self::to_bin_matrix(&acc.dot(&self.adj_matrix), self.length)
            })
    }

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

    pub fn init_path_reconstruction_context(&self) -> Vector<Array2<u32>> {
        let adj_matrices_per_deg = {
            self.adj_matrices_to_degree_bin(self.length as u32)
        };

        adj_matrices_per_deg
    }

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

/*** /graph  ***/

/*** ansi style  ***/

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct AnsiStyles: u32 {
        const Bold      = 0x00000001;
        const Italics   = 0x00000002;
        const Underline = 0x00000004;
        const Crossline = 0x00000008;
        const Black     = 0x00000010;
        const Red       = 0x00000020;
        const Green     = 0x00000040;
        const Yellow    = 0x00000080;
        const Blue      = 0x00000100;
        const Cyan      = 0x00000200;
        const Purple    = 0x00000400;
        const White     = 0x00000800;
        const BgBlack   = 0x00001000;
        const BgRed     = 0x00002000;
        const BgGreen   = 0x00004000;
        const BgYellow  = 0x00008000;
        const BgBlue    = 0x00010000;
        const BgCyan    = 0x00020000;
        const BgPurple  = 0x00040000;
        const BgWhite   = 0x00080000;
    }
}

pub const ANSI_ESCAPE_RESET: &str = "\x1b[0m";
pub const ANSI_ESCAPE_BOLD: &str = "\x1b[1m";
pub const ANSI_ESCAPE_ITALICS: &str = "\x1b[3m";
pub const ANSI_ESCAPE_UNDERLINE: &str = "\x1b[4m";
pub const ANSI_ESCAPE_CROSSLINE: &str = "\x1b[9m";
pub const ANSI_ESCAPE_COLOR_BLACK: &str = "\x1b[30m";
pub const ANSI_ESCAPE_COLOR_RED: &str = "\x1b[31m";
pub const ANSI_ESCAPE_COLOR_GREEN: &str = "\x1b[32m";
pub const ANSI_ESCAPE_COLOR_YELLOW: &str = "\x1b[33m";
pub const ANSI_ESCAPE_COLOR_BLUE: &str = "\x1b[34m";
pub const ANSI_ESCAPE_COLOR_PURPLE: &str = "\x1b[35m";
pub const ANSI_ESCAPE_COLOR_CYAN: &str = "\x1b[36m";
pub const ANSI_ESCAPE_COLOR_WHITE: &str = "\x1b[37m";
pub const ANSI_ESCAPE_COLOR_BG_BLACK: &str = "\x1b[40m";
pub const ANSI_ESCAPE_COLOR_BG_RED: &str = "\x1b[41m";
pub const ANSI_ESCAPE_COLOR_BG_GREEN: &str = "\x1b[42m";
pub const ANSI_ESCAPE_COLOR_BG_YELLOW: &str = "\x1b[43m";
pub const ANSI_ESCAPE_COLOR_BG_BLUE: &str = "\x1b[44m";
pub const ANSI_ESCAPE_COLOR_BG_PURPLE: &str = "\x1b[45m";
pub const ANSI_ESCAPE_COLOR_BG_CYAN: &str = "\x1b[46m";
pub const ANSI_ESCAPE_COLOR_BG_WHITE: &str = "\x1b[47m";

// concatenates per flag
pub fn ansi_style_flags_to_str(styles: AnsiStyles) -> String {
    styles.iter().fold(String::new(), |acc, flag| {
        match flag {
            AnsiStyles::Bold => ANSI_ESCAPE_BOLD,
            AnsiStyles::Italics => ANSI_ESCAPE_ITALICS,
            AnsiStyles::Underline => ANSI_ESCAPE_UNDERLINE,
            AnsiStyles::Crossline => ANSI_ESCAPE_CROSSLINE,
            AnsiStyles::Black => ANSI_ESCAPE_COLOR_BLACK,
            AnsiStyles::Red => ANSI_ESCAPE_COLOR_RED,
            AnsiStyles::Green => ANSI_ESCAPE_COLOR_GREEN,
            AnsiStyles::Yellow => ANSI_ESCAPE_COLOR_YELLOW,
            AnsiStyles::Blue => ANSI_ESCAPE_COLOR_BLUE,
            AnsiStyles::Cyan => ANSI_ESCAPE_COLOR_CYAN,
            AnsiStyles::Purple => ANSI_ESCAPE_COLOR_PURPLE,
            AnsiStyles::White => ANSI_ESCAPE_COLOR_WHITE,
            AnsiStyles::BgBlack => ANSI_ESCAPE_COLOR_BG_BLACK,
            AnsiStyles::BgRed => ANSI_ESCAPE_COLOR_RED,
            AnsiStyles::BgGreen => ANSI_ESCAPE_COLOR_BG_GREEN,
            AnsiStyles::BgYellow => ANSI_ESCAPE_COLOR_BG_YELLOW,
            AnsiStyles::BgBlue => ANSI_ESCAPE_COLOR_BG_BLUE,
            AnsiStyles::BgCyan => ANSI_ESCAPE_COLOR_BG_CYAN,
            AnsiStyles::BgPurple => ANSI_ESCAPE_COLOR_BG_PURPLE,
            AnsiStyles::BgWhite => ANSI_ESCAPE_COLOR_BG_WHITE,
            _ => panic!("Unknown style flag"),
        }
        .pipe(|s| acc + s)
    })
}

pub fn str_style(s: &str, styles: AnsiStyles) -> String {
    ansi_style_flags_to_str(styles).pipe(|prefix| format!("{prefix}{s}{ANSI_ESCAPE_RESET}"))

    // for i in 0..300 {
    //     println!("{i} \x1b[1;38;5;{i}mHello World{ANSI_ESCAPE_RESET}");
    // }
}

pub fn str_unstyle(s: &str) -> String {
    vector![
        ANSI_ESCAPE_RESET,
        ANSI_ESCAPE_BOLD,
        ANSI_ESCAPE_ITALICS,
        ANSI_ESCAPE_UNDERLINE,
        ANSI_ESCAPE_CROSSLINE,
        ANSI_ESCAPE_COLOR_BLACK,
        ANSI_ESCAPE_COLOR_RED,
        ANSI_ESCAPE_COLOR_GREEN,
        ANSI_ESCAPE_COLOR_YELLOW,
        ANSI_ESCAPE_COLOR_BLUE,
        ANSI_ESCAPE_COLOR_PURPLE,
        ANSI_ESCAPE_COLOR_CYAN,
        ANSI_ESCAPE_COLOR_WHITE,
        ANSI_ESCAPE_COLOR_BG_BLACK,
        ANSI_ESCAPE_COLOR_BG_RED,
        ANSI_ESCAPE_COLOR_BG_GREEN,
        ANSI_ESCAPE_COLOR_BG_YELLOW,
        ANSI_ESCAPE_COLOR_BG_BLUE,
        ANSI_ESCAPE_COLOR_BG_PURPLE,
        ANSI_ESCAPE_COLOR_BG_CYAN,
        ANSI_ESCAPE_COLOR_BG_WHITE,
    ]
    .pipe(|ansi_escape_codes| {
        ansi_escape_codes
            .into_iter()
            .fold(s.to_string(), |acc, code| acc.replace(code, ""))
    })
}

pub fn highlight_vect_u32(v: &Vector<u32>, style_lookup: &HashMap<u32, AnsiStyles>) -> String {
    format!(
        "[{}]",
        v.into_iter().fold(String::new(), |acc, n| {
            match style_lookup.get(n) {
                Some(styles) => acc + &str_style(&format!("{n}, "), *styles),
                None => acc + &format!("{n}, "),
            }
        })
    )
}

pub fn display_matrix(
    mat: &Array2<u32>,
    length: usize,
    style_lookup: &HashMap<u32, AnsiStyles>,
) -> String {
    let mut result: String = "".to_string();

    result += "[";

    for i in 0..length {
        if i == 0 {
            result += "[";
        } else {
            result += " [";
        }

        for j in 0..length {
            let n = mat[(i, j)];
            match style_lookup.get(&n) {
                Some(styles) => result += &str_style(&format!("{n}, "), *styles),
                None => result += &format!("{n}, "),
            }
        }

        if i == length - 1 {
            result += "]";
        } else {
            result += "],\n";
        }
    }

    result += "]";

    result
}

/*** /ansi style ***/

/*** pipe ***/

#[macro_export]
macro_rules! pass_ref {
    ($func:expr) => {
        |a| $func(&a)
    };
}

#[macro_export]
macro_rules! pass_ref_tup2_0 {
    ($func:tt) => {
        |(a, b)| $func(&a, b)
    };
}

#[macro_export]
macro_rules! pass_ref_tup2_1 {
    ($func:tt) => {
        |(a, b)| $func(a, &b)
    };
}

#[macro_export]
macro_rules! unpack_tup2 {
    ($func:expr) => {
        |(a, b)| $func(a, b)
    };
}

#[macro_export]
macro_rules! unpack_tup3 {
    ($func:expr) => {
        |(a, b, c)| $func(a, b, c)
    };
}

#[macro_export]
macro_rules! prepend_tup {
    ($func:expr) => {
        |a| ($func(&a), a)
    };
}

#[macro_export]
macro_rules! append_tup {
    ($func:expr) => {
        |a| rev_tup2(($func(&a), a))
    };
}

#[macro_export]
macro_rules! prepend_tup2 {
    ($func:expr) => {
        |(a, b)| ($func(&a, &b), a, b)
    };
}

#[macro_export]
macro_rules! replace_tup2_0 {
    ($func:expr) => {
        |(a, b)| ($func(a, b), b)
    };
}

#[macro_export]
macro_rules! replace_tup2_1 {
    ($func:expr) => {
        |(a, b)| ($func(a, b), b)
    };
}

#[macro_export]
macro_rules! replace_tup3_0 {
    ($func:expr) => {
        |(a, b, c)| ($func(a, b, c), b, c)
    };
}

#[macro_export]
macro_rules! replace_tup3_1 {
    ($func:expr) => {
        |(a, b, c)| (a, $func(a, b, c), b)
    };
}

#[macro_export]
macro_rules! replace_tup3_2 {
    ($func:expr) => {
        |(a, b, c)| (a, b, $func(a, b, c))
    };
}

#[macro_export]
macro_rules! update_tup2_0 {
    ($func:expr) => {
        |(a, b)| ($func(a), b)
    };
}

#[macro_export]
macro_rules! update_tup2_1 {
    ($func:expr) => {
        |(a, b)| (a, $func(b))
    };
}
#[macro_export]
macro_rules! update_tup3_0 {
    ($func:expr) => {
        |(a, b, c)| ($func(a), b, c)
    };
}

#[macro_export]
macro_rules! update_tup3_1 {
    ($func:expr) => {
        |(a, b, c)| (a, $func(b), c)
    };
}

#[macro_export]
macro_rules! update_tup3_2 {
    ($func:expr) => {
        |(a, b, c)| (a, b, $func(c))
    };
}

#[macro_export]
macro_rules! then_apply {
    ($func1:expr, $func2:expr) => {
        |a| ($func2($func1(a)))
    };
}

pub fn rev_tup2<T, U>(tup: (T, U)) -> (U, T) {
    (tup.1, tup.0)
}

pub fn rev_tup3<T, U, V>(tup: (T, U, V)) -> (V, U, T) {
    (tup.2, tup.1, tup.0)
}

pub fn tup3_left_rot<T, U, V>(tup: (T, U, V)) -> (U, V, T) {
    (tup.1, tup.2, tup.0)
}

/// if true, runs f to further process the initial, otherwise returns the initial
pub fn then_if<T>(cond: bool, f: impl Fn(T) -> T) -> impl Fn(T) -> T {
    move |initial| if cond { f(initial) } else { initial }
}

/*** /pipe  ***/

/*** other  ***/

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub fn do_add(left: u32, right: u32) -> u32 {
    left + right
}

pub fn create_n_vectors<T: Copy>(n: usize) -> Vector<Vector<T>> {
    (0..n).fold(vector![], |acc, _i| &acc + &vector![vector![]])
}

pub fn i32_rows_to_columns(tuples: Vector<Vector<i32>>, num_columns: usize) -> Vector<Vector<i32>> {
    tuples
        .iter()
        // turn the vector of rows into a vector of columns
        .fold(create_n_vectors::<i32>(num_columns), |cols, tup| {
            tup.iter().enumerate().fold(cols, |new_cols, (i, &n)| {
                new_cols.update(i, &new_cols[i] + &vector![n])
            })
        })
}

pub fn i32_columns_to_df(
    columns: Vector<Vector<i32>>,
    names: Vec<&str>,
) -> Result<DataFrame, PolarsError> {
    columns
        .into_iter()
        .enumerate()
        .map(|(i, col)| {
            col.into_iter()
                .collect::<Vec<i32>>() // turned into Vec for processing as a Column
                .pipe(|col| Column::new(names[i].into(), col))
        })
        .collect::<Vec<Column>>()
        .pipe(|columns| DataFrame::new(columns))
}

pub fn i32_column_to_freq_map(column: &Vector<i32>) -> HashMap<i32, i32> {
    column
        .into_iter()
        .fold(hashmap! {}, |freq, n| match freq.contains_key(n) {
            true => freq.update(*n, freq[n] + 1),
            false => freq.update(*n, 1),
        })
}

pub fn has_decimals(x: f32) -> bool {
    x - x.round() != 0.
}

pub fn is_palindrom(s: &str) -> bool {
    s.chars()
        .zip(s.chars().rev())
        .all(|(left, right)| left == right)
}

pub fn fetch(url: &str) -> Option<String> {
    // from https://phrohdoh.com/blog/sync-http-rust/
    Client::new()
        .get(url)
        .pipe(|request| request.send())
        .pipe(|result| match result {
            Ok(response) => Some(response.text().unwrap_or("".to_string())),
            Err(_err) => None,
        })
}

pub fn powerset_n<T: Clone>(items: &Vector<T>, n: u32) -> Vector<Vector<(u32, T)>> {
    if items.len() == 0 {
        vector![]
    } else {
        (0..n).into_iter().fold(vector![], |acc, i| {
            powerset_n(&items.clone().slice(1..), n).pipe(|pset| {
                vector![(i, items[0].clone())].pipe(|new_vect| {
                    if pset.len() == 0 {
                        &acc + &vector![new_vect]
                    } else {
                        &acc + &pset
                            .into_iter()
                            .map(|subset| &new_vect + &subset)
                            .collect::<Vector<_>>()
                            .pipe(|x| x)
                    }
                })
            })
        })
    }
}

pub fn vec_to_tup2<T: Clone>(mut vec: Vector<T>) -> (T, T) {
    (vec.pop_back().unwrap(), vec.pop_back().unwrap())
}

/*** /other ***/
