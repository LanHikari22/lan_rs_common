#[cfg(feature = "incl_io")]
pub fn has_decimals(x: f32) -> bool {
    x - x.round() != 0.
}