/* \begin{ansi style} */

use bitflags::bitflags;
use im::{HashMap, Vector};
use tap::prelude::*;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct AnsiStyles: u32 {
        const Bold      = 0x00000001;
        const Italics   = 0x00000002;
        const Underline = 0x00000004;
        const Crossline = 0x00000008;
        const Black     = 0x00000010;
        const Red       = 0x00000020;
        const Green     = 0x00000040;
        const Yellow    = 0x00000080;
        const Blue      = 0x00000100;
        const Cyan      = 0x00000200;
        const Purple    = 0x00000400;
        const White     = 0x00000800;
        const BgBlack   = 0x00001000;
        const BgRed     = 0x00002000;
        const BgGreen   = 0x00004000;
        const BgYellow  = 0x00008000;
        const BgBlue    = 0x00010000;
        const BgCyan    = 0x00020000;
        const BgPurple  = 0x00040000;
        const BgWhite   = 0x00080000;
    }
}

pub const ANSI_ESCAPE_RESET: &str = "\x1b[0m";
pub const ANSI_ESCAPE_BOLD: &str = "\x1b[1m";
pub const ANSI_ESCAPE_ITALICS: &str = "\x1b[3m";
pub const ANSI_ESCAPE_UNDERLINE: &str = "\x1b[4m";
pub const ANSI_ESCAPE_CROSSLINE: &str = "\x1b[9m";
pub const ANSI_ESCAPE_COLOR_BLACK: &str = "\x1b[30m";
pub const ANSI_ESCAPE_COLOR_RED: &str = "\x1b[31m";
pub const ANSI_ESCAPE_COLOR_GREEN: &str = "\x1b[32m";
pub const ANSI_ESCAPE_COLOR_YELLOW: &str = "\x1b[33m";
pub const ANSI_ESCAPE_COLOR_BLUE: &str = "\x1b[34m";
pub const ANSI_ESCAPE_COLOR_PURPLE: &str = "\x1b[35m";
pub const ANSI_ESCAPE_COLOR_CYAN: &str = "\x1b[36m";
pub const ANSI_ESCAPE_COLOR_WHITE: &str = "\x1b[37m";
pub const ANSI_ESCAPE_COLOR_BG_BLACK: &str = "\x1b[40m";
pub const ANSI_ESCAPE_COLOR_BG_RED: &str = "\x1b[41m";
pub const ANSI_ESCAPE_COLOR_BG_GREEN: &str = "\x1b[42m";
pub const ANSI_ESCAPE_COLOR_BG_YELLOW: &str = "\x1b[43m";
pub const ANSI_ESCAPE_COLOR_BG_BLUE: &str = "\x1b[44m";
pub const ANSI_ESCAPE_COLOR_BG_PURPLE: &str = "\x1b[45m";
pub const ANSI_ESCAPE_COLOR_BG_CYAN: &str = "\x1b[46m";
pub const ANSI_ESCAPE_COLOR_BG_WHITE: &str = "\x1b[47m";

// concatenates per flag
pub fn ansi_style_flags_to_str(styles: AnsiStyles) -> String {
    styles.iter().fold(String::new(), |acc, flag| {
        match flag {
            AnsiStyles::Bold => ANSI_ESCAPE_BOLD,
            AnsiStyles::Italics => ANSI_ESCAPE_ITALICS,
            AnsiStyles::Underline => ANSI_ESCAPE_UNDERLINE,
            AnsiStyles::Crossline => ANSI_ESCAPE_CROSSLINE,
            AnsiStyles::Black => ANSI_ESCAPE_COLOR_BLACK,
            AnsiStyles::Red => ANSI_ESCAPE_COLOR_RED,
            AnsiStyles::Green => ANSI_ESCAPE_COLOR_GREEN,
            AnsiStyles::Yellow => ANSI_ESCAPE_COLOR_YELLOW,
            AnsiStyles::Blue => ANSI_ESCAPE_COLOR_BLUE,
            AnsiStyles::Cyan => ANSI_ESCAPE_COLOR_CYAN,
            AnsiStyles::Purple => ANSI_ESCAPE_COLOR_PURPLE,
            AnsiStyles::White => ANSI_ESCAPE_COLOR_WHITE,
            AnsiStyles::BgBlack => ANSI_ESCAPE_COLOR_BG_BLACK,
            AnsiStyles::BgRed => ANSI_ESCAPE_COLOR_RED,
            AnsiStyles::BgGreen => ANSI_ESCAPE_COLOR_BG_GREEN,
            AnsiStyles::BgYellow => ANSI_ESCAPE_COLOR_BG_YELLOW,
            AnsiStyles::BgBlue => ANSI_ESCAPE_COLOR_BG_BLUE,
            AnsiStyles::BgCyan => ANSI_ESCAPE_COLOR_BG_CYAN,
            AnsiStyles::BgPurple => ANSI_ESCAPE_COLOR_BG_PURPLE,
            AnsiStyles::BgWhite => ANSI_ESCAPE_COLOR_BG_WHITE,
            _ => panic!("Unknown style flag"),
        }
        .pipe(|s| acc + s)
    })
}

pub fn str_style(s: &str, styles: AnsiStyles) -> String {
    ansi_style_flags_to_str(styles).pipe(|prefix| format!("{prefix}{s}{ANSI_ESCAPE_RESET}"))

    // for i in 0..300 {
    //     println!("{i} \x1b[1;38;5;{i}mHello World{ANSI_ESCAPE_RESET}");
    // }
}

pub fn str_unstyle(s: &str) -> String {
    vec![
        ANSI_ESCAPE_RESET,
        ANSI_ESCAPE_BOLD,
        ANSI_ESCAPE_ITALICS,
        ANSI_ESCAPE_UNDERLINE,
        ANSI_ESCAPE_CROSSLINE,
        ANSI_ESCAPE_COLOR_BLACK,
        ANSI_ESCAPE_COLOR_RED,
        ANSI_ESCAPE_COLOR_GREEN,
        ANSI_ESCAPE_COLOR_YELLOW,
        ANSI_ESCAPE_COLOR_BLUE,
        ANSI_ESCAPE_COLOR_PURPLE,
        ANSI_ESCAPE_COLOR_CYAN,
        ANSI_ESCAPE_COLOR_WHITE,
        ANSI_ESCAPE_COLOR_BG_BLACK,
        ANSI_ESCAPE_COLOR_BG_RED,
        ANSI_ESCAPE_COLOR_BG_GREEN,
        ANSI_ESCAPE_COLOR_BG_YELLOW,
        ANSI_ESCAPE_COLOR_BG_BLUE,
        ANSI_ESCAPE_COLOR_BG_PURPLE,
        ANSI_ESCAPE_COLOR_BG_CYAN,
        ANSI_ESCAPE_COLOR_BG_WHITE,
    ]
    .pipe(|ansi_escape_codes| {
        ansi_escape_codes
            .into_iter()
            .fold(s.to_string(), |acc, code| acc.replace(code, ""))
    })
}

pub fn highlight_vect_u32(v: &Vector<u32>, style_lookup: &HashMap<u32, AnsiStyles>) -> String {
    format!(
        "[{}]",
        v.into_iter().fold(String::new(), |acc, n| {
            match style_lookup.get(n) {
                Some(styles) => acc + &str_style(&format!("{n}, "), *styles),
                None => acc + &format!("{n}, "),
            }
        })
    )
}

#[cfg(feature = "use_ndarray")]
pub fn display_matrix(
    mat: &Array2<u32>,
    length: usize,
    style_lookup: &HashMap<u32, AnsiStyles>,
) -> String {
    let mut result: String = "".to_string();

    result += "[";

    for i in 0..length {
        if i == 0 {
            result += "[";
        } else {
            result += " [";
        }

        for j in 0..length {
            let n = mat[(i, j)];
            match style_lookup.get(&n) {
                Some(styles) => result += &str_style(&format!("{n}, "), *styles),
                None => result += &format!("{n}, "),
            }
        }

        if i == length - 1 {
            result += "]";
        } else {
            result += "],\n";
        }
    }

    result += "]";

    result
}

/* \end{ansi style} */