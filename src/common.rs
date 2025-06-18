#[allow(unused_imports)]
use im::{hashmap, hashset, ordset, vector, HashMap, HashSet, OrdSet, Vector};
#[allow(unused_imports)]
use itertools::FoldWhile::{Continue, Done};
#[allow(unused_imports)]
use itertools::Itertools;
#[cfg(feature = "use_network")]
use reqwest::blocking::Client;
#[allow(unused_imports)]
// use std::collections::HashMap;
use bitflags::bitflags;
use libm;
use regex::Regex;
use std::fs::File;
use std::io::prelude::*; // needed for traits like the Read trait
use std::io::{self, Error};
use std::path::Path;
use tap::prelude::*;

#[cfg(feature = "df")]
use polars::prelude::*;

#[cfg(feature = "use_ndarray")]
use ndarray::{s, Array2, ShapeBuilder};

/* \begin{other} */

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

#[cfg(feature = "df")]
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

#[cfg(feature = "use_network")]
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

pub fn vec_push<T>(mut vec: Vec<T>, v: T) -> Vec<T> {
    vec.push(v);
    vec
}

pub fn get_lineno_and_col_at_index(
    lines: std::str::Lines<'_>,
    index: usize,
) -> Option<(usize, usize)> {
    let (lineno_opt, cur_index) = {
        lines
            .enumerate()
            .fold_while((None, 0), |(_res_lineno, cur_index), (lineno, line)| {
                // println!("index: {index}, cur_index: {cur_index}, lineno: {lineno}");
                if cur_index <= index {
                    if index <= cur_index + line.len() {
                        // Within range!
                        Done((Some(lineno), cur_index + line.len()))
                    } else {
                        Continue((None, cur_index + line.len() + "\n".len()))
                    }
                } else {
                    // We passed where the index should be...
                    Done((None, cur_index))
                }
            })
            .into_inner()
    };

    match lineno_opt {
        Some(lineno) => Some((lineno, cur_index)),
        None => None,
    }
    .and_then(|(linenum, cur_index)| Some((linenum, cur_index - index)))
}

pub fn cartesian_product_str_concat(left: &Vector<String>, right: &Vec<&str>) -> Vector<String> {
    left.iter()
        .cartesian_product(right.into_iter())
        .fold(vector![], |acc, tup| {
            &acc + &vector![tup.0.to_owned() + tup.1]
        })
}



/* \end{other} */
