/* \begin{IO} */

#[cfg(feature = "use_serde")]
use csv::{ReaderBuilder, WriterBuilder};

#[cfg(feature = "use_im")]
use im::Vector;

#[cfg(any(feature = "use_serde", feature = "use_ron"))]
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::File,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use tap::prelude::*;

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

pub fn parse_fixed_i32_tuple_lines(s: &str, tuple_size: usize) -> Result<Vec<Vec<i32>>, String> {
    s.trim()
        .lines()
        .map(|line| {
            line.split_ascii_whitespace()
                .collect::<Vec<_>>()
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
                        .collect::<Result<Vec<_>, String>>()
                })
        })
        .collect::<Result<Vec<_>, String>>()
}

pub fn parse_i32_tuple_lines(s: &str) -> Result<Vec<Vec<i32>>, String> {
    s.trim()
        .lines()
        .map(|line| {
            line.split_ascii_whitespace()
                .collect::<Vec<_>>()
                .pipe(|tokens| {
                    tokens
                        .iter()
                        .map(|&num_s| {
                            num_s
                                .parse::<i32>()
                                .or(Err(format!("{num_s} is not an i32")))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
        })
        .collect::<Result<Vec<_>, String>>()
}

#[cfg(feature = "use_libm")]
pub fn get_square_buf_length(s: &str) -> Option<usize> {
    libm::sqrtf(s.len() as f32).pipe(|f| {
        if has_decimals(f) {
            None
        } else {
            Some(f as usize)
        }
    })
}

#[cfg(feature = "use_libm")]
pub fn read_square_char_input(s: &str) -> (usize, String) {
    s.trim()
        .replace("\n", "")
        .pipe(|s| (get_square_buf_length(&s).expect("Must have square buf"), s))
}

#[cfg(feature = "use_serde")]
/// Mark a type as "#[derive(Serialize, Deserialize)]" and it should be usable through this
pub fn write_vec_to_csv<T: Serialize>(path: &str, data: &[T]) -> Result<(), Box<dyn std::error::Error>> {
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
pub fn read_vec_from_csv<T: DeserializeOwned>(path: &str) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().from_reader(file);

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
    use itertools::Itertools;

    file_formats
        .into_iter()
        .map(|format| path_push(path, format!("**/*.{format}")))
        .map(|path| glob::glob(path.to_str().expect("Path must exist")))
        .map(|res| res.expect("Glob pattern must be valid"))
        .map(|paths| {
            paths
                .map(|res| res.expect("Should get valid path"))
                .collect::<Vec<_>>()
        })
        .concat()
}

/* \begin{RON IO} */
/// This uses example [repro003](<https://github.com/LanHikari22/rs_repro/blob/main/src/repro_tracked/repro003_ron_read_write.rs>)

#[cfg(feature = "use_ron")]
use ron::{
    error::SpannedResult,
    ser::PrettyConfig,
    Error as RonError,
};

#[cfg(feature = "use_ron")]
pub fn write_ron_obj_to_str<T: Serialize>(obj: &T) -> Result<String, RonError> {
    ron::ser::to_string_pretty(obj, PrettyConfig::default())
}

#[cfg(feature = "use_ron")]
pub fn write_ron_obj_to_file<T: Serialize>(path: &PathBuf, obj: &T) -> Result<usize, RonError> {
    let mut file = File::create(path)?;

    file.write(write_ron_obj_to_str(obj)?.as_bytes())
        .map_err(|err| RonError::Io(err.to_string()))
}

#[cfg(feature = "use_ron")]
pub fn read_ron_obj_from_str<T: DeserializeOwned>(s: &str) -> SpannedResult<T> {
    ron::from_str::<T>(s)
}

#[cfg(feature = "use_ron")]
pub fn read_ron_obj_from_file<T: DeserializeOwned>(path: &PathBuf) -> Result<T, RonError> {
    let mut file = File::open(path)?;

    let mut content = String::new();

    file.read_to_string(&mut content)?;

    read_ron_obj_from_str(&content).map_err(|e| e.code)
}

/// Serializes a list of T into a string with one record per line
#[cfg(feature = "use_ron")]
pub fn write_ron_vec_to_str<T: Serialize>(records: &[T]) -> Result<String, RonError> {
    let mut mut_str = String::new();

    let as_strings = {
        records
            .into_iter()
            .map(|record| {
                ron::ser::to_string_pretty(
                    &record,
                    PrettyConfig::new()
                        .compact_arrays(true)
                        .compact_maps(true)
                        .compact_structs(true)
                        .escape_strings(true),
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    as_strings.into_iter().for_each(|s| {
        mut_str.push_str(&s);
        mut_str.push_str(if cfg!(not(target_os = "windows")) {
            "\n"
        } else {
            "\r\n"
        })
    });

    Ok(mut_str)
}

/// Serializes a list of T into a text file with one record per line
#[cfg(feature = "use_ron")]
pub fn write_ron_vec_to_file<T: Serialize>(
    path: &PathBuf,
    records: &[T],
) -> Result<usize, RonError> {
    let mut file = File::create(path)?;

    file.write(write_ron_vec_to_str(records)?.as_bytes())
        .map_err(|err| RonError::Io(err.to_string()))
}

/// This reader assumes that every row has one entry, so it would not work if they are split across lines.
#[cfg(feature = "use_ron")]
pub fn read_ron_vec_from_str<T: DeserializeOwned>(s: &str) -> SpannedResult<Vec<T>> {
    s //_
        .lines()
        .map(|s| ron::from_str::<T>(s))
        .collect::<Result<Vec<_>, _>>()
}

#[cfg(feature = "use_ron")]
pub fn read_ron_vec_from_file<T: DeserializeOwned>(path: &PathBuf) -> Result<Vec<T>, RonError> {
    let mut file = File::open(path)?;

    let mut content = String::new();

    file.read_to_string(&mut content)?;

    read_ron_vec_from_str(&content).map_err(|e| e.code)
}

/* \end{RON IO} */

/* \end{IO} */
