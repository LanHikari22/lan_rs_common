/* \begin{pipe} */

#[macro_export]
macro_rules! pass_ref {
    ($func:expr) => {
        |a| $func(&a)
    };
}

#[macro_export]
macro_rules! pass_ref_tup2_0 {
    ($func:tt) => {
        |(a, b)| $func(&a, b)
    };
}

#[macro_export]
macro_rules! pass_ref_tup2_1 {
    ($func:tt) => {
        |(a, b)| $func(a, &b)
    };
}

#[macro_export]
macro_rules! unpack_tup2 {
    ($func:expr) => {
        |(a, b)| $func(a, b)
    };
}

#[macro_export]
macro_rules! unpack_tup3 {
    ($func:expr) => {
        |(a, b, c)| $func(a, b, c)
    };
}

#[macro_export]
macro_rules! prepend_tup {
    ($func:expr) => {
        |a| ($func(&a), a)
    };
}

#[macro_export]
macro_rules! append_tup {
    ($func:expr) => {
        |a| rev_tup2(($func(&a), a))
    };
}

#[macro_export]
macro_rules! prepend_tup2 {
    ($func:expr) => {
        |(a, b)| ($func(&a, &b), a, b)
    };
}

#[macro_export]
macro_rules! replace_tup2_0 {
    ($func:expr) => {
        |(a, b)| ($func(a, b), b)
    };
}

#[macro_export]
macro_rules! replace_tup2_1 {
    ($func:expr) => {
        |(a, b)| ($func(a, b), b)
    };
}

#[macro_export]
macro_rules! replace_tup3_0 {
    ($func:expr) => {
        |(a, b, c)| ($func(a, b, c), b, c)
    };
}

#[macro_export]
macro_rules! replace_tup3_1 {
    ($func:expr) => {
        |(a, b, c)| (a, $func(a, b, c), b)
    };
}

#[macro_export]
macro_rules! replace_tup3_2 {
    ($func:expr) => {
        |(a, b, c)| (a, b, $func(a, b, c))
    };
}

#[macro_export]
macro_rules! update_tup2_0 {
    ($func:expr) => {
        |(a, b)| ($func(a), b)
    };
}

#[macro_export]
macro_rules! update_tup2_1 {
    ($func:expr) => {
        |(a, b)| (a, $func(b))
    };
}
#[macro_export]
macro_rules! update_tup3_0 {
    ($func:expr) => {
        |(a, b, c)| ($func(a), b, c)
    };
}

#[macro_export]
macro_rules! update_tup3_1 {
    ($func:expr) => {
        |(a, b, c)| (a, $func(b), c)
    };
}

#[macro_export]
macro_rules! update_tup3_2 {
    ($func:expr) => {
        |(a, b, c)| (a, b, $func(c))
    };
}

#[macro_export]
macro_rules! then_apply {
    ($func1:expr, $func2:expr) => {
        |a| ($func2($func1(a)))
    };
}

pub fn rev_tup2<T, U>(tup: (T, U)) -> (U, T) {
    (tup.1, tup.0)
}

pub fn rev_tup3<T, U, V>(tup: (T, U, V)) -> (V, U, T) {
    (tup.2, tup.1, tup.0)
}

pub fn tup3_left_rot<T, U, V>(tup: (T, U, V)) -> (U, V, T) {
    (tup.1, tup.2, tup.0)
}

/// if true, runs f to further process the initial, otherwise returns the initial
pub fn then_if<T>(cond: bool, f: impl Fn(T) -> T) -> impl Fn(T) -> T {
    move |initial| if cond { f(initial) } else { initial }
}

/* \end{pipe} */