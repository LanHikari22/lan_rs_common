
/* \begin{IO} */

use std::{fs::File, io::{self, Read, Write}, path::Path};
use tap::prelude::*;

use im::Vector;

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

pub fn read_user_input_from_file(input_path: &str) -> String {
    let path = Path::new(input_path);
    let mut file = File::open(&path).unwrap();
    let mut out = String::new();

    file.read_to_string(&mut out).unwrap();

    out
}

pub fn write_to_file(s: &str, output_path: &str) -> Result<usize, std::io::Error> {
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

/* \end{IO} */