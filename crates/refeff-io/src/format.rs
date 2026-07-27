use std::fmt::Write as _;

/// A single fixed-width Fortran column format, as used by FEFF's column
/// layouts (e.g. `E13.6`, `F11.4`, `G15.6`).
///
/// Column layouts are the thing golden byte-for-byte compatibility depends
/// on; naming them with [`FortranField`] turns invisible magic numbers such
/// as `write_fortran_exp(out, *value, 13, 6)` into a documented constant such
/// as `const CHI_ROW_VALUE: FortranField = FortranField::E { width: 13,
/// precision: 6 };`.
///
/// Each variant mirrors one of this module's free formatting functions:
/// [`FortranField::E`] is [`write_fortran_exp`], [`FortranField::F`] is a
/// plain Rust fixed-point field (matching Fortran's `Fw.d`),
/// [`FortranField::G`] is [`write_fortran_g`], and
/// [`FortranField::ZeroScaledE`] is
/// [`write_fortran_zero_scaled_exp_with_exponent_width`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FortranField {
    /// Fortran `Ew.d` scientific field with a two-digit exponent, e.g.
    /// `E13.6`. See [`write_fortran_exp`].
    E {
        /// Total field width in characters.
        width: usize,
        /// Digits after the decimal point.
        precision: usize,
    },
    /// Fortran `Fw.d` fixed-point field, e.g. `F11.4`.
    F {
        /// Total field width in characters.
        width: usize,
        /// Digits after the decimal point.
        precision: usize,
    },
    /// Fortran `Gw.d` field that switches between fixed and scientific
    /// notation, e.g. `G15.6`. See [`write_fortran_g`].
    G {
        /// Total field width in characters.
        width: usize,
        /// Digits after the decimal point.
        precision: usize,
    },
    /// Canonical zero-scaled Fortran `Ew.dEe` field with a configurable
    /// exponent width. See
    /// [`write_fortran_zero_scaled_exp_with_exponent_width`].
    ZeroScaledE {
        /// Total field width in characters.
        width: usize,
        /// Digits after the decimal point.
        precision: usize,
        /// Minimum digits in the exponent.
        exp_width: usize,
    },
}

impl FortranField {
    /// Append `value` to `out` using this field's Fortran format.
    pub fn write(self, out: &mut impl std::fmt::Write, value: f64) -> std::fmt::Result {
        match self {
            FortranField::E { width, precision } => write_fortran_exp(out, value, width, precision),
            FortranField::F { width, precision } => write!(out, "{value:width$.precision$}"),
            FortranField::G { width, precision } => write_fortran_g(out, value, width, precision),
            FortranField::ZeroScaledE {
                width,
                precision,
                exp_width,
            } => write_fortran_zero_scaled_exp_with_exponent_width(
                out, value, width, precision, exp_width,
            ),
        }
    }
}

/// Write one row of Fortran fields and their values in order, joining
/// consecutive fields with the literal `separator` (FEFF's column gap, such
/// as a single space for a `1x` descriptor or an empty string for
/// back-to-back fields). No separator is written before the first field.
pub fn write_fortran_row(
    out: &mut impl std::fmt::Write,
    separator: &str,
    fields: impl IntoIterator<Item = (FortranField, f64)>,
) -> std::fmt::Result {
    let mut first = true;
    for (field, value) in fields {
        if first {
            first = false;
        } else {
            out.write_str(separator)?;
        }
        field.write(out, value)?;
    }
    Ok(())
}

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
    write_fortran_zero_scaled_exp_with_exponent_width(out, value, width, precision, 2)
}

/// Append a canonical Fortran `Ew.dEe` field with a requested exponent width.
pub fn write_fortran_zero_scaled_exp_with_exponent_width(
    out: &mut impl std::fmt::Write,
    value: f64,
    width: usize,
    precision: usize,
    min_exponent_width: usize,
) -> std::fmt::Result {
    let exponent = if value == 0.0 {
        0
    } else {
        value.abs().log10().floor() as i32 + 1
    };
    // Preserve IEEE-754 signed zero. FEFF/gfortran emits `-0.000...` for a
    // negative zero, and dropping that sign breaks byte-exact handoff
    // roundtrips even though the numeric values compare equal.
    let mantissa = if value == 0.0 {
        value
    } else {
        value / 10.0_f64.powi(exponent)
    };
    let mantissa = format!("{mantissa:.precision$}");
    let sign = if exponent < 0 { '-' } else { '+' };
    let exponent_digits = exponent.abs().to_string();
    let exponent_width = exponent_digits.len().max(min_exponent_width);
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

/// Format a float like a Fortran `Gw.d` field.
///
/// This covers the FEFF-compatible fixed/scientific switch used by output such
/// as `G14.6`: fixed notation for magnitudes in `[0.1, 10^d)` and canonical
/// zero-scaled scientific notation otherwise.
#[must_use]
pub fn fortran_g(value: f64, width: usize, precision: usize) -> String {
    let mut out = String::new();
    let _ = write_fortran_g(&mut out, value, width, precision);
    out
}

/// Append a float like a Fortran `Gw.d` field.
pub fn write_fortran_g(
    out: &mut impl std::fmt::Write,
    value: f64,
    width: usize,
    precision: usize,
) -> std::fmt::Result {
    let magnitude = value.abs();
    let fixed_upper = 10.0_f64.powi(precision as i32);
    if value == 0.0 || ((0.1..fixed_upper).contains(&magnitude)) {
        let digits_before_decimal = if magnitude >= 1.0 {
            magnitude.log10().floor() as usize + 1
        } else {
            0
        };
        let decimals = if value == 0.0 {
            precision.saturating_sub(1)
        } else if magnitude < 1.0 {
            precision
        } else {
            precision.saturating_sub(digits_before_decimal)
        };
        let mut fixed = format!("{value:.decimals$}");
        if decimals == 0 && !fixed.contains('.') {
            fixed.push('.');
        }
        let fixed_width = width.saturating_sub(4);
        for _ in 0..fixed_width.saturating_sub(fixed.len()) {
            out.write_char(' ')?;
        }
        out.write_str(&fixed)?;
        for _ in 0..width.saturating_sub(fixed_width) {
            out.write_char(' ')?;
        }
        Ok(())
    } else {
        write_fortran_zero_scaled_exp(out, value, width, precision)
    }
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

/// Format one `f64` as the 24-column list-directed record used by older FEFF
/// modules such as COMPTON and CRPA.
///
/// Those routines write double precision values with 15 significant digits.
/// Fixed-form values occupy the first 19 columns and leave five trailing
/// blanks; scientific values occupy the full 24-column field with a
/// three-digit exponent.
#[must_use]
pub fn fortran_list_directed_g15_f64(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude != 0.0 && (0.1..1.0e15).contains(&magnitude) {
        let decimals = list_directed_g15_decimal_places(magnitude);
        let mut fixed = format!("{value:.decimals$}");
        if decimals == 0 && !fixed.contains('.') {
            fixed.push('.');
        }
        format!("{fixed:>19}     ")
    } else {
        let scientific = fortran_list_directed_g15_exponent(value);
        format!("{scientific:>24}")
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

fn list_directed_g15_decimal_places(magnitude: f64) -> usize {
    if magnitude < 1.0 {
        15
    } else {
        let digits_before_decimal = magnitude.log10().floor() as usize + 1;
        15_usize.saturating_sub(digits_before_decimal)
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

fn fortran_list_directed_g15_exponent(value: f64) -> String {
    let raw = format!("{value:.15E}");
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
    fn fortran_field_write_matches_free_functions() {
        let mut e_field = String::new();
        assert!(
            FortranField::E {
                width: 13,
                precision: 6
            }
            .write(&mut e_field, -18.69)
            .is_ok()
        );
        assert_eq!(e_field, fortran_exp(-18.69, 13, 6));

        let mut f_field = String::new();
        assert!(
            FortranField::F {
                width: 11,
                precision: 4
            }
            .write(&mut f_field, 0.0)
            .is_ok()
        );
        assert_eq!(f_field, format!("{:11.4}", 0.0_f64));

        let mut g_field = String::new();
        assert!(
            FortranField::G {
                width: 14,
                precision: 6
            }
            .write(&mut g_field, 8979.41)
            .is_ok()
        );
        assert_eq!(g_field, fortran_g(8979.41, 14, 6));

        let mut zero_scaled_field = String::new();
        assert!(
            FortranField::ZeroScaledE {
                width: 20,
                precision: 10,
                exp_width: 3
            }
            .write(&mut zero_scaled_field, -0.293_644_216_9)
            .is_ok()
        );
        let mut expected = String::new();
        assert!(
            write_fortran_zero_scaled_exp_with_exponent_width(
                &mut expected,
                -0.293_644_216_9,
                20,
                10,
                3,
            )
            .is_ok()
        );
        assert_eq!(zero_scaled_field, expected);
    }

    #[test]
    fn write_fortran_row_joins_fields_with_separator() {
        let mut out = String::new();
        let field = FortranField::E {
            width: 13,
            precision: 6,
        };
        assert!(
            write_fortran_row(&mut out, " ", [(field, 1.5), (field, -2.5), (field, 0.0)]).is_ok()
        );
        assert_eq!(
            out,
            format!(
                "{} {} {}",
                fortran_exp(1.5, 13, 6),
                fortran_exp(-2.5, 13, 6),
                fortran_exp(0.0, 13, 6)
            )
        );
    }

    #[test]
    fn write_fortran_row_supports_empty_separator() {
        let mut out = String::new();
        assert!(
            write_fortran_row(
                &mut out,
                "",
                [
                    (
                        FortranField::F {
                            width: 12,
                            precision: 3
                        },
                        11076.317
                    ),
                    (
                        FortranField::F {
                            width: 11,
                            precision: 3
                        },
                        -40.0
                    ),
                ]
            )
            .is_ok()
        );
        assert_eq!(out, format!("{:12.3}{:11.3}", 11076.317_f64, -40.0_f64));
    }

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
        assert_eq!(fortran_zero_scaled_exp(-0.0, 13, 5), " -0.00000E+00");
        let mut explicit = String::new();
        assert!(
            write_fortran_zero_scaled_exp_with_exponent_width(
                &mut explicit,
                -0.293_644_216_9,
                20,
                10,
                3,
            )
            .is_ok()
        );
        assert_eq!(explicit, "  -0.2936442169E+000");
    }

    #[test]
    fn formats_fortran_g_fields() {
        assert_eq!(fortran_g(8979.41, 14, 6), "   8979.41    ");
        assert_eq!(fortran_g(273.822, 14, 6), "   273.822    ");
        assert_eq!(fortran_g(276.260, 14, 6), "   276.260    ");
        assert_eq!(fortran_g(100.0, 14, 6), "   100.000    ");
        assert_eq!(fortran_g(0.0, 14, 6), "   0.00000    ");
        assert_eq!(fortran_g(0.1, 14, 6), "  0.100000    ");
        assert_eq!(fortran_g(999_999.0, 14, 6), "   999999.    ");
        assert_eq!(fortran_g(1.0e6, 14, 6), "  0.100000E+07");
        assert_eq!(fortran_g(0.308_730e-7, 14, 6), "  0.308730E-07");
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

    #[test]
    fn formats_legacy_g15_list_directed_double_fields_like_feff() {
        assert_eq!(
            fortran_list_directed_g15_f64(0.0),
            "  0.000000000000000E+000"
        );
        assert_eq!(
            fortran_list_directed_g15_f64(5.005_004_815_757_275e-3),
            "  5.005004815757275E-003"
        );
        assert_eq!(
            fortran_list_directed_g15_f64(2.744_767_348_503_43),
            "   2.74476734850343     "
        );
        assert_eq!(
            fortran_list_directed_g15_f64(-0.167_258_739_984_332),
            " -0.167258739984332     "
        );
        assert_eq!(
            fortran_list_directed_g15_f64(1.0),
            "   1.00000000000000     "
        );
    }
}
