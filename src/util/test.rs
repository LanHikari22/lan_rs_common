use std::{fs::File, path::PathBuf};

use rand::{distributions::Alphanumeric, Rng};

#[cfg(feature = "use_arbitrary")]
use arbitrary::{Arbitrary, Unstructured};

#[cfg(feature = "use_arbitrary")]
pub fn fuzzed_instance<T: for<'a> Arbitrary<'a>>() -> Option<T> {
    // Generate random bytes
    let mut bytes = vec![0u8; 512];
    rand::thread_rng().fill(&mut bytes[..]);

    let mut u = Unstructured::new(&bytes);
    T::arbitrary(&mut u).ok()
}


pub fn random_string(len: usize) -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

pub fn create_random_tmp_file() -> std::io::Result<(PathBuf, File)> {
    let tmp_dir = std::env::temp_dir();
    let filename = format!("rnd.{}", random_string(10));
    let path = tmp_dir.join(filename);
    let file = File::create(&path)?;
    Ok((path, file))
}