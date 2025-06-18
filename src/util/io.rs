
/* \begin{IO} */

use std::{fs::File, io::{self, Read, Write}, path::{Path, PathBuf}};
use tap::prelude::*;
use im::Vector;
use std::error::Error;
use std::io::prelude::*; // needed for traits like the Read trait
#[cfg(feature = "use_serde")]
use csv::{WriterBuilder, ReaderBuilder};
#[cfg(feature = "use_serde")]
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::has_decimals;

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

pub fn read_file_to_string(input_path: &str) -> String {
    let path = Path::new(input_path);
    let mut file = File::open(&path).unwrap();
    let mut out = String::new();

    file.read_to_string(&mut out).unwrap();

    out
}

pub fn write_string_to_file(s: &str, output_path: &str) -> Result<usize, std::io::Error> {
    let path = std::path::Path::new(output_path);
    let mut file = std::fs::File::create(&path).unwrap();

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

#[cfg(feature = "use_serde")]
/// Mark a type as "#[derive(Serialize, Deserialize)]" and it should be usable through this
pub fn write_vec_to_csv<T: Serialize>(path: &str, data: &[T]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::NonNumeric) // ensures strings are quoted when needed
        .from_writer(file);

    for item in data {
        writer.serialize(item)?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(feature = "use_serde")]
pub fn read_vec_from_csv<T: DeserializeOwned>(path: &str) -> Result<Vec<T>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .from_reader(file);

    let mut result = Vec::new();
    for record in reader.deserialize() {
        result.push(record?);
    }
    Ok(result)
}

pub fn path_push<P>(left: &PathBuf, right: P) -> PathBuf
where
    P: AsRef<Path>,
{
    let mut left = left.clone();
    left.push(right);
    left
}

#[cfg(feature = "use_glob")]
pub fn glob_multiple_file_formats_in_path(
    path: &PathBuf,
    file_formats: &Vec<&str>,
) -> Vec<PathBuf> {
    // println!("glob on {path:?}");

    use itertools::Itertools;
    file_formats
        .into_iter()
        .map(|format| path_push(path, format!("**/*.{format}")))
        // .map(|path| {
        //     println!("new_path: {path:?}");
        //     path
        // })
        .map(|path| glob::glob(path.to_str().expect("Path must exist")))
        .map(|res| res.expect("Glob pattern must be valid"))
        .map(|paths| {
            paths
                .map(|res| res.expect("Should get valid path"))
                // .map(|path| path.to_str().expect("Should be valid str").to_string())
                .collect::<Vec<_>>()
        })
        .concat()
}


/* \end{IO} */