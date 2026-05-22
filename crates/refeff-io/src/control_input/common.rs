use std::fmt::Write as _;

use crate::Result;

/// FEFF10 Bohr radius in Angstrom, from `COMMON/m_constants.f90`.
pub const FEFF_BOHR_ANGSTROM: f64 = 0.529_177_249;
pub(super) const DENSITY_FILENAME_WIDTH: usize = 30;

pub(super) fn control_bool(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

pub(super) fn write_f13_5_line(out: &mut String, values: [f64; 3]) -> Result<()> {
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        values[0], values[1], values[2]
    )?;
    Ok(())
}

pub(super) fn fixed_left(value: &str, width: usize) -> String {
    let mut out: String = value.chars().take(width).collect();
    while out.len() < width {
        out.push(' ');
    }
    out
}

pub(super) fn fortran_fixed_string(value: &str, width: usize) -> String {
    let mut end = 0;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > width {
            break;
        }
        end = next;
    }
    value[..end].to_string()
}
