
/* \begin{regex} */

use im::{vector, Vector};
use regex::Regex;
use tap::prelude::*;

pub fn regex_capture_once(s: &str, re: &Regex) -> Result<Vector<String>, String> {
    re
        .pipe(|re| {
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

/// This will exhaust a string stream using `try_scan_fn`. It allows the callback to determine the
/// amount of advance in the stream for the next call, or it fails otherwise.
pub fn exhaustively_process_using_scanners<T: Clone>(
    s: &str,
    try_scan_fn: impl Fn(&str) -> Option<(T, usize)>,
) -> Result<Vec<T>, (Vec<T>, usize)> {

    let mut mut_acc_items: Vec<T> = vec![];
    let mut mut_acc_advance = 0;

    loop {
        let scan_opt = try_scan_fn(&s[mut_acc_advance..]);

        match scan_opt {
            Some((item, advance)) => {
                mut_acc_items.push(item);
                let prev_acc_advance = mut_acc_advance;
                mut_acc_advance += advance;

                if prev_acc_advance >= s.len() {
                    // We have reached the end
                    break;
                }
            },

            None => {
                // We have failed to exhaaustively scan the buffer
                break;
            },
        }
    }

    if mut_acc_advance == s.len() {
        Ok(mut_acc_items)
    } else{
        Err((mut_acc_items, mut_acc_advance))
    }
}


/* \end{regex} */