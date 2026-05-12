use std::fmt::Write as _;

/// FEFF frequently writes fixed-width integer lists using Fortran formats such
/// as `20(1x,i4)`. This helper produces the repeated `1x,iN` form.
pub fn repeated_ints(values: impl IntoIterator<Item = i64>, width: usize) -> String {
    let mut out = String::new();
    for value in values {
        let _ = write!(out, " {value:>width$}");
    }
    out
}

/// Format a float using a Fortran-like scientific notation.
///
/// Rust prints exponents as `e+00`; Fortran output in FEFF is generally
/// accepted with either `e` or `E`, so this keeps lowercase `e` while preserving
/// explicit sign and exponent width.
pub fn exp(value: f64, width: usize, precision: usize) -> String {
    format!("{value:>width$.precision$e}")
}

/// Format a float like a Fortran `Ew.d` field with a two-digit exponent.
#[must_use]
pub fn fortran_exp(value: f64, width: usize, precision: usize) -> String {
    let raw = format!("{value:.precision$E}");
    let Some((mantissa, exponent)) = raw.split_once('E') else {
        return format!("{raw:>width$}");
    };
    let (sign, digits) = match exponent.as_bytes().first() {
        Some(b'-') => ('-', &exponent[1..]),
        Some(b'+') => ('+', &exponent[1..]),
        _ => ('+', exponent),
    };
    let field = format!("{mantissa}E{sign}{digits:0>2}");
    format!("{field:>width$}")
}

pub fn repeated_exp(
    values: impl IntoIterator<Item = f64>,
    width: usize,
    precision: usize,
) -> String {
    let mut out = String::new();
    for value in values {
        out.push(' ');
        out.push_str(&exp(value, width, precision));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_repeated_ints_like_fortran_list() {
        assert_eq!(repeated_ints([1, -2, 345], 4), "    1   -2  345");
    }

    #[test]
    fn formats_fixed_width_exponents() {
        assert_eq!(exp(12.5, 14, 7), "   1.2500000e1");
    }

    #[test]
    fn formats_fortran_style_e_fields() {
        assert_eq!(fortran_exp(1.5073e-4, 12, 4), "  1.5073E-04");
        assert_eq!(fortran_exp(-0.7625, 12, 4), " -7.6250E-01");
        assert_eq!(fortran_exp(0.0, 12, 4), "  0.0000E+00");
        assert_eq!(fortran_exp(12_345.0, 12, 4), "  1.2345E+04");
    }
}
