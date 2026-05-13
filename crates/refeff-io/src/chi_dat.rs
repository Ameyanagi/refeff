//! FEFF `chi.dat` EXAFS spectrum text codec.
//!
//! FEFF writes the final EXAFS `chi.dat` table with four numeric columns:
//! photoelectron wave number `k`, EXAFS `chi`, complex-path magnitude, and
//! unwrapped phase. Diagnostic runs can append real and imaginary `ckp`
//! columns, while per-path `chipNNNN.dat` files append `phase - 2kr`.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const CHI_DAT_STANDARD_ROW_WIDTH: usize = 4;
const CHI_DAT_PATH_ROW_WIDTH: usize = 5;
const CHI_DAT_CKP_ROW_WIDTH: usize = 6;
const CHI_DAT_ALLOWED_ROW_WIDTHS: &str = "4, 5, or 6";

/// Parsed FEFF `chi.dat` or `chipNNNN.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiDatData {
    /// Header and comment lines before and around the numeric spectrum table.
    pub header_lines: Vec<String>,
    /// Photoelectron wave number in inverse Angstrom.
    pub wave_number: Array1<f64>,
    /// EXAFS fine structure value.
    pub chi: Array1<f64>,
    /// Magnitude of the complex accumulated EXAFS contribution.
    pub magnitude: Array1<f64>,
    /// Unwrapped complex phase in radians.
    pub phase: Array1<f64>,
    /// Optional per-path `phase - 2kr` column from `chipNNNN.dat`.
    pub phase_minus_2kr: Option<Array1<f64>>,
    /// Optional real part of diagnostic complex `ckp`.
    pub ckp_real: Option<Array1<f64>>,
    /// Optional imaginary part of diagnostic complex `ckp`.
    pub ckp_imag: Option<Array1<f64>>,
}

impl ChiDatData {
    /// Number of spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.wave_number.len()
    }

    /// Whether this table has the per-path `phase - 2kr` column.
    #[must_use]
    pub fn has_path_phase(&self) -> bool {
        self.phase_minus_2kr.is_some()
    }

    /// Whether this table has diagnostic real/imaginary `ckp` columns.
    #[must_use]
    pub fn has_complex_wave_number(&self) -> bool {
        self.ckp_real.is_some() && self.ckp_imag.is_some()
    }
}

/// Render FEFF-compatible `chi.dat` or `chipNNNN.dat` text.
pub fn chi_dat_string(data: &ChiDatData) -> Result<String> {
    validate_chi_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    match (&data.phase_minus_2kr, &data.ckp_real, &data.ckp_imag) {
        (None, None, None) => {
            for (((k, chi), magnitude), phase) in data
                .wave_number
                .iter()
                .zip(data.chi.iter())
                .zip(data.magnitude.iter())
                .zip(data.phase.iter())
            {
                write_chi_row(&mut out, *k, [*chi, *magnitude, *phase])?;
            }
        }
        (Some(phase_minus_2kr), None, None) => {
            for ((((k, chi), magnitude), phase), path_phase) in data
                .wave_number
                .iter()
                .zip(data.chi.iter())
                .zip(data.magnitude.iter())
                .zip(data.phase.iter())
                .zip(phase_minus_2kr.iter())
            {
                write_chi_row(&mut out, *k, [*chi, *magnitude, *phase, *path_phase])?;
            }
        }
        (None, Some(ckp_real), Some(ckp_imag)) => {
            for (((((k, chi), magnitude), phase), ckp_real), ckp_imag) in data
                .wave_number
                .iter()
                .zip(data.chi.iter())
                .zip(data.magnitude.iter())
                .zip(data.phase.iter())
                .zip(ckp_real.iter())
                .zip(ckp_imag.iter())
            {
                write_chi_row(
                    &mut out,
                    *k,
                    [*chi, *magnitude, *phase, *ckp_real, *ckp_imag],
                )?;
            }
        }
        _ => {
            return Err(invalid_chi_dat(
                "optional columns",
                "unsupported column combination",
            ));
        }
    }

    Ok(out)
}

fn write_chi_row<const N: usize>(
    out: &mut String,
    wave_number: f64,
    fields: [f64; N],
) -> Result<()> {
    write!(out, "{wave_number:11.4}   ")?;
    if let Some((first, rest)) = fields.split_first() {
        write_fortran_exp(out, *first, 13, 6)?;
        for value in rest {
            out.push(' ');
            write_fortran_exp(out, *value, 13, 6)?;
        }
    }
    out.push('\n');
    Ok(())
}

/// Parse FEFF `chi.dat` or `chipNNNN.dat` text.
pub fn parse_chi_dat(text: &str) -> Result<ChiDatData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut wave_number = Vec::new();
    let mut chi = Vec::new();
    let mut magnitude = Vec::new();
    let mut phase = Vec::new();
    let mut phase_minus_2kr = Vec::new();
    let mut ckp_real = Vec::new();
    let mut ckp_imag = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(
                width,
                CHI_DAT_STANDARD_ROW_WIDTH | CHI_DAT_PATH_ROW_WIDTH | CHI_DAT_CKP_ROW_WIDTH
            ) {
                return Err(IoError::ChiDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: CHI_DAT_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::ChiDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: row_width_label(expected),
                    });
                }
            } else {
                row_width = Some(width);
            }

            wave_number.push(parse_f64(line_number, "wave number", tokens[0])?);
            chi.push(parse_f64(line_number, "chi", tokens[1])?);
            magnitude.push(parse_f64(line_number, "magnitude", tokens[2])?);
            phase.push(parse_f64(line_number, "phase", tokens[3])?);
            if width == CHI_DAT_PATH_ROW_WIDTH {
                phase_minus_2kr.push(parse_f64(line_number, "phase minus 2kr", tokens[4])?);
            }
            if width == CHI_DAT_CKP_ROW_WIDTH {
                ckp_real.push(parse_f64(line_number, "ckp real", tokens[4])?);
                ckp_imag.push(parse_f64(line_number, "ckp imaginary", tokens[5])?);
            }
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let data = ChiDatData {
        header_lines,
        wave_number: Array1::from_vec(wave_number),
        chi: Array1::from_vec(chi),
        magnitude: Array1::from_vec(magnitude),
        phase: Array1::from_vec(phase),
        phase_minus_2kr: (row_width == Some(CHI_DAT_PATH_ROW_WIDTH))
            .then(|| Array1::from_vec(phase_minus_2kr)),
        ckp_real: (row_width == Some(CHI_DAT_CKP_ROW_WIDTH)).then(|| Array1::from_vec(ckp_real)),
        ckp_imag: (row_width == Some(CHI_DAT_CKP_ROW_WIDTH)).then(|| Array1::from_vec(ckp_imag)),
    };
    validate_chi_dat(&data)?;
    Ok(data)
}

/// Write FEFF `chi.dat` or `chipNNNN.dat` text to a file.
pub fn write_chi_dat(path: impl AsRef<Path>, data: &ChiDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, chi_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `chi.dat` or `chipNNNN.dat` text from a file.
pub fn read_chi_dat(path: impl AsRef<Path>) -> Result<ChiDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_chi_dat(&text)
}

fn validate_chi_dat(data: &ChiDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_chi_dat(
            "rows",
            "at least one spectrum row is required",
        ));
    }
    validate_len("chi", data.chi.len(), point_count)?;
    validate_len("magnitude", data.magnitude.len(), point_count)?;
    validate_len("phase", data.phase.len(), point_count)?;

    match (&data.phase_minus_2kr, &data.ckp_real, &data.ckp_imag) {
        (None, None, None) => {}
        (Some(phase_minus_2kr), None, None) => {
            validate_len("phase_minus_2kr", phase_minus_2kr.len(), point_count)?;
        }
        (None, Some(ckp_real), Some(ckp_imag)) => {
            validate_len("ckp_real", ckp_real.len(), point_count)?;
            validate_len("ckp_imag", ckp_imag.len(), point_count)?;
        }
        _ => {
            return Err(invalid_chi_dat(
                "optional columns",
                "use either phase_minus_2kr, ckp_real with ckp_imag, or no optional columns",
            ));
        }
    }

    for (row, (((k, chi), magnitude), phase)) in data
        .wave_number
        .iter()
        .zip(data.chi.iter())
        .zip(data.magnitude.iter())
        .zip(data.phase.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite_row("wave number", *k, row)?;
        validate_finite_row("chi", *chi, row)?;
        validate_finite_row("magnitude", *magnitude, row)?;
        validate_finite_row("phase", *phase, row)?;
    }
    if let Some(phase_minus_2kr) = &data.phase_minus_2kr {
        for (row, value) in phase_minus_2kr.iter().enumerate() {
            validate_finite_row("phase minus 2kr", *value, row + 1)?;
        }
    }
    if let Some(ckp_real) = &data.ckp_real {
        for (row, value) in ckp_real.iter().enumerate() {
            validate_finite_row("ckp real", *value, row + 1)?;
        }
    }
    if let Some(ckp_imag) = &data.ckp_imag {
        for (row, value) in ckp_imag.iter().enumerate() {
            validate_finite_row("ckp imaginary", *value, row + 1)?;
        }
    }

    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::ChiDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::ChiDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidChiDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_chi_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidChiDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

fn row_width_label(width: usize) -> &'static str {
    match width {
        CHI_DAT_STANDARD_ROW_WIDTH => "4",
        CHI_DAT_PATH_ROW_WIDTH => "5",
        CHI_DAT_CKP_ROW_WIDTH => "6",
        _ => CHI_DAT_ALLOWED_ROW_WIDTHS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feff_chi_reference_shape() -> Result<()> {
        let data = parse_chi_dat(CHI_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert!(!data.has_path_phase());
        assert!(!data.has_complex_wave_number());
        assert_eq!(data.wave_number[0], 0.0);
        assert_eq!(data.chi[1], -1.194138e-1);
        assert_eq!(data.magnitude[2], 2.750836e-1);
        assert_eq!(data.phase[0], -2.698164);
        Ok(())
    }

    #[test]
    fn parses_per_path_phase_column() -> Result<()> {
        let data = parse_chi_dat(CHIP_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(data.has_path_phase());
        assert_eq!(
            data.phase_minus_2kr
                .as_ref()
                .ok_or_else(|| invalid_chi_dat("phase_minus_2kr", "missing optional column"))?[1],
            2.5
        );
        Ok(())
    }

    #[test]
    fn parses_diagnostic_ckp_columns() -> Result<()> {
        let data = parse_chi_dat(CHI_CKP_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(data.has_complex_wave_number());
        assert_eq!(
            data.ckp_real
                .as_ref()
                .ok_or_else(|| invalid_chi_dat("ckp_real", "missing optional column"))?[0],
            1.25
        );
        assert_eq!(
            data.ckp_imag
                .as_ref()
                .ok_or_else(|| invalid_chi_dat("ckp_imag", "missing optional column"))?[1],
            -0.0625
        );
        Ok(())
    }

    #[test]
    fn roundtrips_chi_text() -> Result<()> {
        let data = parse_chi_dat(CHI_DAT)?;
        let rendered = chi_dat_string(&data)?;
        assert_eq!(rendered, CHI_DAT);
        assert_eq!(parse_chi_dat(&rendered)?, data);

        let chip = parse_chi_dat(CHIP_DAT)?;
        assert_eq!(chi_dat_string(&chip)?, CHIP_DAT);
        let ckp = parse_chi_dat(CHI_CKP_DAT)?;
        assert_eq!(chi_dat_string(&ckp)?, CHI_CKP_DAT);
        Ok(())
    }

    #[test]
    fn rejects_bad_chi_inputs() {
        assert!(parse_chi_dat("# no data\n").is_err());
        assert!(parse_chi_dat("1 2 3\n").is_err());
        assert!(parse_chi_dat("1 2 3 4 5 6 7\n").is_err());
        assert!(parse_chi_dat("1 2 3 NaN\n").is_err());
        assert!(parse_chi_dat("1 2 3 4\n2 3 4 5 6\n").is_err());
    }

    const CHI_DAT: &str = r#"# # Cu                                                           FEFF 10.0
#     0/   0 paths used
#  -----------------------------------------------------------------------
#       k          chi          mag           phase @#
     0.0000   -1.159383E-01  2.702278E-01 -2.698164E+00
     0.0500   -1.194138E-01  2.726708E-01 -2.688285E+00
     0.1000   -1.229126E-01  2.750836E-01 -2.678386E+00
"#;

    const CHIP_DAT: &str = r#"# path contribution
 -----------------------------------------------------------------------
       k         chi           mag          phase        phase-2kr  @#
     0.0000    1.000000E-01  2.000000E-01  1.000000E+00  1.500000E+00
     0.0500    1.250000E-01  2.250000E-01  2.000000E+00  2.500000E+00
"#;

    const CHI_CKP_DAT: &str = r#"# diagnostic ckp
#       k          chi          mag           phase @#
     0.0000    1.000000E-01  2.000000E-01  1.000000E+00  1.250000E+00 -1.250000E-01
     0.0500    1.250000E-01  2.250000E-01  2.000000E+00  1.500000E+00 -6.250000E-02
"#;
}
