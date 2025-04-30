use generator::{done, Generator, Gn};
use std::{sync::{Arc, Mutex}, time::{Duration, Instant}};
use tap::prelude::*;

use rayon::prelude::*;

/* \begin{benchmark} */


pub fn benchmark_collect_vec<'a, T: Send, I: Iterator<Item = T> + Send + 'a>(
    input: I,
) -> Generator<'a, (), (Duration, Vec<T>)> {
    Gn::new_scoped(|mut s| {
        let start = Instant::now();
        let res = input.collect::<Vec<T>>();
        let duration = start.elapsed();

        s.yield_with((duration, res));

        done!();
    })
}

pub fn benchmark_debug_log_mean<'a, T: Send, I: Iterator<Item = T> + Send + 'a>(
    input: I,
    label: &'a str,
    opt_batch: Option<u32>,
) -> Generator<'a, (), T> {
    Gn::new_scoped(move |mut s| {
        let label = if label != "" {format!("{label}: ")} else {"".to_string()};

        println!("{label}Measuring Time...");

        let start = Instant::now();
        let mut counter = 0;
        let mut prev_duration = start.elapsed();

        for elem in input {
            // println!("counter {counter}");
            counter += 1;
            s.yield_with(elem);

            if let Some(batch) = opt_batch {
                if counter % batch == 0 {
                    let duration = start.elapsed();
                    println!("{label}Batch Duration Measured: Total ({counter}): Per {batch} ({:?}, Mean: {:?}), Cumulative ({duration:?}, Mean: {:?})", 
                        (duration - prev_duration),
                        (duration - prev_duration).div_f32(batch as f32),
                        duration.div_f32(counter as f32));
                    prev_duration = duration;
                }
            }
        }

        let duration = start.elapsed();

        println!(
            "{label}Duration Measured: Total ({counter}): {duration:?}, Mean: {:?}",
            duration.div_f32(counter as f32)
        );

        done!();
    })
}

pub fn benchmark_debug_log_mean_par<I: ParallelIterator>(iter: I, label: &str, opt_batch: Option<u32>,) -> impl ParallelIterator<Item = I::Item> {
    let counter = Arc::new(Mutex::new(0));
    let label = if label != "" {format!("{label}: ")} else {"".to_string()};

    println!("{label}Measuring Time...");

    let start = Instant::now();
    let prev_duration = Arc::new(Mutex::new(start.elapsed()));

    let iter = {
        iter.map_with((counter.clone(), prev_duration.clone()),  |(counter, prev_duration), item| {
            let mut counter_guard = counter.lock().unwrap();
            let mut prev_duration_guard = prev_duration.lock().unwrap();

            *counter_guard += 1;

            if let Some(batch) = opt_batch {
                if *counter_guard % batch == 0 {
                    let duration = start.elapsed();
                    println!("{label}Batch Duration Measured: Total ({}): Per {batch} ({:?}, Mean: {:?}), Cumulative ({duration:?}, Mean: {:?})", 
                        *counter_guard,
                        (duration - *prev_duration_guard),
                        (duration - *prev_duration_guard).div_f32(batch as f32),
                        duration.div_f32(*counter_guard as f32));
                    *prev_duration_guard = duration;
                }
            }

            item
        })
    };

    iter.collect::<Vec<I::Item>>()
        .tap(|_| {
            let duration = start.elapsed();
            let counter_guard = counter.lock().unwrap();

            println!(
                "{label}Duration Measured: Total ({}): {duration:?}, Mean: {:?}",
                *counter_guard,
                duration.div_f32(*counter_guard as f32)
            );
        })
        .into_par_iter()
}

/* \end{benchmark} */