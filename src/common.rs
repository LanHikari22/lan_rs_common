#[allow(unused_imports)]
// use std::collections::HashMap;
use bitflags::bitflags;
use core::num;
#[allow(unused_imports)]
use im::{hashmap, hashset, ordset, HashMap, HashSet, OrdSet};
#[allow(unused_imports)]
use itertools::FoldWhile::{Continue, Done};
#[allow(unused_imports)]
use itertools::Itertools;
use libm;
use regex::Regex;
#[cfg(feature = "use_network")]
use reqwest::blocking::Client;
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

pub fn create_n_vectors<T: Copy>(n: usize) -> Vec<Vec<T>> {
    let mut mut_result: Vec<Vec<T>> = vec![];

    for _ in 0..n {
        mut_result.push(vec![]);
    }

    mut_result
}

pub fn i32_rows_to_columns(rows: Vec<Vec<i32>>, num_columns: usize) -> Vec<Vec<i32>> {
    let mut mut_cols = create_n_vectors::<i32>(num_columns);

    rows //_
        .iter()
        .for_each(|row| {
            row //_
                .iter()
                .enumerate()
                .for_each(|(which_col, value)| {
                    mut_cols[which_col].push(*value);
                });
        });

    mut_cols
}

#[cfg(feature = "df")]
pub fn i32_columns_to_df(
    columns: Vec<Vec<i32>>,
    names: Vec<&str>,
) -> Result<DataFrame, PolarsError> {
    columns
        .into_iter()
        .enumerate()
        .map(|(i, col)| {
            col //_
                .into_iter()
                .collect::<Vec<i32>>() // turned into Vec for processing as a Column
                .pipe(|col| Column::new(names[i].into(), col))
        })
        .collect::<Vec<Column>>()
        .pipe(|columns| DataFrame::new(columns))
}

pub fn i32_column_to_freq_map(column: &Vec<i32>) -> HashMap<i32, i32> {
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
            Errn(_err) => None,
        })
}

pub fn powerset_n<T: Clone>(items: &Vec<T>, n: u32) -> Vec<Vec<(u32, T)>> {
    if items.len() == 0 {
        vec![]
    } else {
        let mut mut_result: Vec<Vec<(u32, T)>> = vec![];

        for i in 0..n {
            let pset = powerset_n(&items[1..].to_vec(), n);

            let new_vect = vec![(i, items[0].clone())];

            if pset.len() == 0 {
                mut_result.push(new_vect);
            } else {
                pset //_
                    .into_iter()
                    .map(|subset| vec![new_vect.clone(), subset].concat())
                    .for_each(|vect| mut_result.push(vect));
            }
        }

        mut_result
    }
}

pub fn vec_to_tup2<T: Clone>(vec: Vec<T>) -> (T, T) {
    (vec[0].clone(), vec[1].clone())
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

pub fn cartesian_product_str_concat(left: &Vec<String>, right: &Vec<&str>) -> Vec<String> {
    let mut mut_result: Vec<String> = vec![];

    left //_
        .iter()
        .cartesian_product(right.into_iter())
        .for_each(|tup| {
            mut_result.push(tup.0.to_owned() + tup.1)

        });

    mut_result
}

/* \end{other} */
