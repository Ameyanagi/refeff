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
    let mut out = String::new();
    let _ = write_fortran_exp(&mut out, value, width, precision);
    out
}

/// Append a float like a Fortran `Ew.d` field with a two-digit exponent.
pub fn write_fortran_exp(
    out: &mut impl std::fmt::Write,
    value: f64,
    width: usize,
    precision: usize,
) -> std::fmt::Result {
    let raw = format!("{value:.precision$E}");
    let Some((mantissa, exponent)) = raw.split_once('E') else {
        return write!(out, "{raw:>width$}");
    };
    let (sign, digits) = match exponent.as_bytes().first() {
        Some(b'-') => ('-', &exponent[1..]),
        Some(b'+') => ('+', &exponent[1..]),
        _ => ('+', exponent),
    };
    let exponent_width = digits.len().max(2);
    let field_width = mantissa.len() + 2 + exponent_width;
    for _ in 0..width.saturating_sub(field_width) {
        out.write_char(' ')?;
    }
    out.write_str(mantissa)?;
    out.write_char('E')?;
    out.write_char(sign)?;
    for _ in 0..exponent_width.saturating_sub(digits.len()) {
        out.write_char('0')?;
    }
    out.write_str(digits)
}

/// Format a float like a canonical Fortran `Ew.d` field.
///
/// This form keeps the mantissa in `[0.1, 1.0)` for non-zero values, matching
/// Fortran output such as `0.3809984030E-01`.
#[must_use]
pub fn fortran_zero_scaled_exp(value: f64, width: usize, precision: usize) -> String {
    let mut out = String::new();
    let _ = write_fortran_zero_scaled_exp(&mut out, value, width, precision);
    out
}

/// Append a float like a canonical Fortran `Ew.d` field.
pub fn write_fortran_zero_scaled_exp(
    out: &mut impl std::fmt::Write,
    value: f64,
    width: usize,
    precision: usize,
) -> std::fmt::Result {
    let exponent = if value == 0.0 {
        0
    } else {
        value.abs().log10().floor() as i32 + 1
    };
    let mantissa = if value == 0.0 {
        0.0
    } else {
        value / 10.0_f64.powi(exponent)
    };
    let mantissa = format!("{mantissa:.precision$}");
    let sign = if exponent < 0 { '-' } else { '+' };
    let exponent_digits = exponent.abs().to_string();
    let exponent_width = exponent_digits.len().max(2);
    let field_width = mantissa.len() + 2 + exponent_width;
    for _ in 0..width.saturating_sub(field_width) {
        out.write_char(' ')?;
    }
    out.write_str(&mantissa)?;
    out.write_char('E')?;
    out.write_char(sign)?;
    for _ in 0..exponent_width.saturating_sub(exponent_digits.len()) {
        out.write_char('0')?;
    }
    out.write_str(&exponent_digits)
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

/// Format one `f64` as gfortran writes a double precision value in a
/// list-directed `WRITE(unit,*)` record.
///
/// FEFF uses this implicit format for a few scalar handoff files. gfortran
/// emits each double in a 26-character field, using fixed notation for
/// magnitudes in `[0.1, 1e17)` and scientific notation otherwise.
#[must_use]
pub fn fortran_list_directed_f64(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude != 0.0 && !(0.1..1.0e17).contains(&magnitude) {
        let scientific = fortran_list_directed_exponent(value);
        format!("{scientific:>26}")
    } else {
        let decimals = list_directed_decimal_places(magnitude);
        let mut fixed = format!("{value:.decimals$}");
        if decimals == 0 {
            fixed.push('.');
        }
        format!("{fixed:>21}     ")
    }
}

fn list_directed_decimal_places(magnitude: f64) -> usize {
    if magnitude == 0.0 {
        16
    } else if magnitude < 1.0 {
        17
    } else {
        let digits_before_decimal = magnitude.log10().floor() as usize + 1;
        17_usize.saturating_sub(digits_before_decimal)
    }
}

fn fortran_list_directed_exponent(value: f64) -> String {
    let raw = format!("{value:.16E}");
    let Some((mantissa, exponent)) = raw.split_once('E') else {
        return raw;
    };
    let (sign, digits) = match exponent.as_bytes().first() {
        Some(b'-') => ('-', &exponent[1..]),
        Some(b'+') => ('+', &exponent[1..]),
        _ => ('+', exponent),
    };
    match digits.parse::<u32>() {
        Ok(exponent) => format!("{mantissa}E{sign}{exponent:03}"),
        Err(_) => raw,
    }
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
        let mut out = String::from("prefix");
        assert!(write_fortran_exp(&mut out, -18.69, 11, 4).is_ok());
        assert_eq!(out, "prefix-1.8690E+01");
    }

    #[test]
    fn formats_zero_scaled_fortran_e_fields() {
        assert_eq!(
            fortran_zero_scaled_exp(0.038_099_840_3, 20, 10),
            "    0.3809984030E-01"
        );
        assert_eq!(fortran_zero_scaled_exp(-4.3629, 13, 5), " -0.43629E+01");
        assert_eq!(fortran_zero_scaled_exp(0.0, 13, 5), "  0.00000E+00");
    }

    #[test]
    fn formats_list_directed_double_fields_like_gfortran() {
        assert_eq!(fortran_list_directed_f64(0.0), "   0.0000000000000000     ");
        assert_eq!(
            fortran_list_directed_f64(330.319_156_029_843_7),
            "   330.31915602984373     "
        );
        assert_eq!(
            fortran_list_directed_f64(6.354_647_093_099_486e-2),
            "   6.3546470930994858E-002"
        );
        assert_eq!(
            fortran_list_directed_f64(-7.729_278_779_143_69),
            "  -7.7292787791436899     "
        );
        assert_eq!(
            fortran_list_directed_f64(1.0e20),
            "   1.0000000000000000E+020"
        );
        assert_eq!(
            fortran_list_directed_f64(1.0e16),
            "   10000000000000000.     "
        );
        assert_eq!(fortran_list_directed_f64(0.1), "  0.10000000000000001     ");
    }
}
