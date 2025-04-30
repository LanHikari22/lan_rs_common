/* \begin{UBig} */

use std::sync::{Arc, Mutex};

use generator::{done, Generator, Gn};
use ibig::{ubig, UBig};
use rayon::prelude::*;

pub fn ubig_range_inclusive(m: UBig, n: UBig) -> impl Iterator<Item = UBig> {
    std::iter::successors(Some(m.clone()), move |i| {
        if i >= &n {
            None // Stop iteration end of range
        } else {
            Some(i + 1) // Increment `UBig` manually
        }
    })
}

/// r_{n+1} = \frac{1}{2} {\left(r_n + \frac{k}{r_n}\right)}
/// Or in easier to compute (no 1/ns) form
/// r_{n+1} = \frac{k + r_n^2}{2 r_n}
pub fn ubig_sqrt_using_newtons_method(k: &UBig, num_iters: usize) -> UBig {
    (0..num_iters).fold(k.clone(), |prev_estimate, _| {
        (k + &prev_estimate * &prev_estimate) / (2 * prev_estimate)
    })
}

pub fn enumerate_ubig<'a, T: Send, I: Iterator<Item = T> + Send + 'a>(
    input: I,
) -> Generator<'a, (), (UBig, T)> {
    Gn::new_scoped(|mut s| {
        let mut counter: UBig = ubig!(0);

        for elem in input {
            s.yield_with((counter.clone(), elem));

            counter = counter + 1;
        }
        done!();
    })
}

pub fn enumerate_ubig_par<I: ParallelIterator>(iter: I) -> impl ParallelIterator<Item = (UBig, I::Item)> {
    let counter = Arc::new(Mutex::new(UBig::from(0u8)));

    iter.map_with(counter, |counter, item| {
        let mut guard = counter.lock().unwrap();
        let idx = guard.clone();
        *guard += 1u8;

        (idx, item)
    })
}

/* \end{UBig} */