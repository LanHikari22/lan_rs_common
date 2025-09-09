use ibig::{ubig, UBig};

#[cfg(feature = "incl_ubig")]
use super::ubig::*;

use rayon::prelude::*;
use tap::prelude::*;

/* \begin{math} */

pub fn ubig_is_prime(n: &UBig) -> bool {
    let sqrt_n: UBig = { ubig_sqrt_using_newtons_method(&n, 20) };

    ubig_range_inclusive(ubig!(2), sqrt_n) //_
        .all(|m| 
            //_
            m == *n || n % m != ubig!(0)
        )
}

pub fn ubig_primes(len: UBig) -> Vec<UBig> {
    ubig_range_inclusive(ubig!(2), len.clone())
        .collect::<Vec<_>>()
        .into_par_iter()
        // .into_iter()
        .skip(2)
        .filter(|n| ubig_is_prime(n))
        .pipe(|x| x)
        .pipe(enumerate_ubig_par)
        // .pipe(enumerate_ubig)
        // .take_while(|(nth_prime, _nth)| *nth_prime < len)
        // .pipe(|iter| benchmark_debug_log_mean_par(iter, "primes", Some(10000)))
        // .pipe(|iter| benchmark_debug_log_mean(iter, "primes", Some(10000)))
        .map(|(_nth_prime, nth)| nth)
        .collect::<Vec<_>>()
}

pub fn ubig_prime_factors(n: UBig) -> Vec<UBig> {
    ubig_primes(ubig_sqrt_using_newtons_method(&n, 20))
        .into_par_iter()
        .filter(|p| n.clone() % p == ubig!(0))
        .map(|n| n.clone())
        .collect::<Vec<UBig>>()
}

/* \end{math} */