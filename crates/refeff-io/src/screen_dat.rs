//! FEFF screened-core-hole radial diagnostic table support.
//!
//! The SCREEN module writes `wscrn.dat` with the radial grid, screened
//! potential, and core-hole potential. XSPH can then write `vtot.dat` when it
//! folds the screened core-hole potential into the central-atom total
//! potential. Both files are simple FEFF three-column radial tables.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const WSCRN_DAT_PATH: &str = "wscrn.dat";
const VTOT_DAT_PATH: &str = "vtot.dat";
const WSCRN_DEFAULT_HEADER: &str = "# r       w_scrn(r)      v_ch(r)";

/// Parsed contents of FEFF `wscrn.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct WscrnDatData {
    /// Header/comment lines before the numeric radial table.
    pub header_lines: Vec<String>,
    /// Radial grid in bohr.
    pub radius_bohr: Array1<f64>,
    /// Screened potential `w_scrn(r)` in atomic units.
    pub screened_potential: Array1<f64>,
    /// Core-hole potential `v_ch(r)` in atomic units.
    pub core_hole_potential: Array1<f64>,
}

impl WscrnDatData {
    /// Number of radial grid rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.radius_bohr.len()
    }
}

/// Parsed contents of FEFF `vtot.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct VtotDatData {
    /// Header/comment lines before the numeric radial table.
    pub header_lines: Vec<String>,
    /// Radial grid in bohr.
    pub radius_bohr: Array1<f64>,
    /// Original total potential before the screened core-hole update.
    pub total_potential: Array1<f64>,
    /// Screened core-hole potential read from `wscrn.dat`.
    pub screened_core_hole_potential: Array1<f64>,
}

impl VtotDatData {
    /// Number of radial grid rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.radius_bohr.len()
    }
}

/// Parse FEFF `wscrn.dat` text.
pub fn parse_wscrn_dat(text: &str) -> Result<WscrnDatData> {
    let table = parse_three_column_table(text, WSCRN_DAT_PATH)?;
    let data = WscrnDatData {
        header_lines: table.header_lines,
        radius_bohr: Array1::from_vec(table.first),
        screened_potential: Array1::from_vec(table.second),
        core_hole_potential: Array1::from_vec(table.third),
    };
    validate_wscrn_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `vtot.dat` text.
pub fn parse_vtot_dat(text: &str) -> Result<VtotDatData> {
    let table = parse_three_column_table(text, VTOT_DAT_PATH)?;
    let data = VtotDatData {
        header_lines: table.header_lines,
        radius_bohr: Array1::from_vec(table.first),
        total_potential: Array1::from_vec(table.second),
        screened_core_hole_potential: Array1::from_vec(table.third),
    };
    validate_vtot_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `wscrn.dat` text.
pub fn wscrn_dat_string(data: &WscrnDatData) -> Result<String> {
    validate_wscrn_dat(data)?;
    if data.header_lines.is_empty() {
        three_column_string(
            &[WSCRN_DEFAULT_HEADER],
            &data.radius_bohr,
            &data.screened_potential,
            &data.core_hole_potential,
        )
    } else {
        three_column_string(
            &data.header_lines,
            &data.radius_bohr,
            &data.screened_potential,
            &data.core_hole_potential,
        )
    }
}

/// Render FEFF-compatible `vtot.dat` text.
pub fn vtot_dat_string(data: &VtotDatData) -> Result<String> {
    validate_vtot_dat(data)?;
    three_column_string(
        &data.header_lines,
        &data.radius_bohr,
        &data.total_potential,
        &data.screened_core_hole_potential,
    )
}

/// Read FEFF `wscrn.dat` text from a file.
pub fn read_wscrn_dat(path: impl AsRef<Path>) -> Result<WscrnDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_wscrn_dat(&text)
}

/// Read FEFF `vtot.dat` text from a file.
pub fn read_vtot_dat(path: impl AsRef<Path>) -> Result<VtotDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_vtot_dat(&text)
}

/// Write FEFF `wscrn.dat` text to a file.
pub fn write_wscrn_dat(path: impl AsRef<Path>, data: &WscrnDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, wscrn_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `vtot.dat` text to a file.
pub fn write_vtot_dat(path: impl AsRef<Path>, data: &VtotDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, vtot_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

#[derive(Debug)]
struct ThreeColumnTable {
    header_lines: Vec<String>,
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
}

fn parse_three_column_table(text: &str, path: &'static str) -> Result<ThreeColumnTable> {
    let mut header_lines = Vec::new();
    let mut first = Vec::new();
    let mut second = Vec::new();
    let mut third = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            header_lines.push(raw.trim_end().to_owned());
            continue;
        }

        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 3 {
            return parse_error(
                path,
                line_number,
                format!("radial table row has {} token(s), expected 3", tokens.len()),
            );
        }
        first.push(parse_f64(path, line_number, "first", tokens[0])?);
        second.push(parse_f64(path, line_number, "second", tokens[1])?);
        third.push(parse_f64(path, line_number, "third", tokens[2])?);
    }

    if first.is_empty() {
        return parse_error(path, 0, "at least one radial table row is required");
    }
    Ok(ThreeColumnTable {
        header_lines,
        first,
        second,
        third,
    })
}

fn three_column_string(
    header_lines: &[impl AsRef<str>],
    first: &Array1<f64>,
    second: &Array1<f64>,
    third: &Array1<f64>,
) -> Result<String> {
    let mut out = String::new();
    for line in header_lines {
        writeln!(out, "{}", line.as_ref())?;
    }
    for ((first, second), third) in first.iter().zip(second.iter()).zip(third.iter()) {
        write_fortran_zero_scaled_exp(&mut out, *first, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, *second, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, *third, 20, 10)?;
        out.push('\n');
    }
    Ok(out)
}

fn validate_wscrn_dat(data: &WscrnDatData) -> Result<()> {
    validate_three_columns(
        WSCRN_DAT_PATH,
        "wscrn",
        &data.radius_bohr,
        &data.screened_potential,
        &data.core_hole_potential,
    )
}

fn validate_vtot_dat(data: &VtotDatData) -> Result<()> {
    validate_three_columns(
        VTOT_DAT_PATH,
        "vtot",
        &data.radius_bohr,
        &data.total_potential,
        &data.screened_core_hole_potential,
    )
}

fn validate_three_columns(
    path: &'static str,
    table: &'static str,
    first: &Array1<f64>,
    second: &Array1<f64>,
    third: &Array1<f64>,
) -> Result<()> {
    if first.is_empty() {
        return parse_error(
            path,
            0,
            format!("{table} table must contain at least one row"),
        );
    }
    if second.len() != first.len() {
        return parse_error(
            path,
            0,
            format!(
                "second column length {} does not match radius length {}",
                second.len(),
                first.len()
            ),
        );
    }
    if third.len() != first.len() {
        return parse_error(
            path,
            0,
            format!(
                "third column length {} does not match radius length {}",
                third.len(),
                first.len()
            ),
        );
    }

    for (row, ((first, second), third)) in first
        .iter()
        .zip(second.iter())
        .zip(third.iter())
        .enumerate()
    {
        let line = row + 1;
        validate_finite(path, line, "first", *first)?;
        validate_finite(path, line, "second", *second)?;
        validate_finite(path, line, "third", *third)?;
    }
    Ok(())
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value = token.replace(['D', 'd'], "E").parse::<f64>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })?;
    validate_finite(path, line, field, value)?;
    Ok(value)
}

fn validate_finite(path: &'static str, line: usize, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(path, line, format!("{field} must be finite"))
    }
}

fn parse_error<T>(path: &'static str, line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(path, line, message))
}

fn parse_error_value(path: &'static str, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn parses_wscrn_dat() -> Result<()> {
        let parsed = parse_wscrn_dat(WSCRN_DAT)?;
        assert_eq!(
            parsed.header_lines,
            vec![" # r       w_scrn(r)      v_ch(r)"]
        );
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.radius_bohr[0], 0.150_733_046_3E-03);
        assert_eq!(parsed.screened_potential[1], 0.267_288_167_8E+02);
        assert_eq!(parsed.core_hole_potential[2], 0.291_616_320_4E+02);

        let rendered = wscrn_dat_string(&parsed)?;
        assert_eq!(rendered, WSCRN_DAT);
        assert_eq!(parse_wscrn_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_vtot_dat() -> Result<()> {
        let parsed = parse_vtot_dat(VTOT_DAT)?;
        assert!(parsed.header_lines.is_empty());
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.radius_bohr[0], 0.150_733_046_3E-03);
        assert_eq!(parsed.total_potential[1], -0.182_900_133_6E+06);
        assert_eq!(parsed.screened_core_hole_potential[2], 0.267_288_030_6E+02);

        let rendered = vtot_dat_string(&parsed)?;
        assert_eq!(rendered, VTOT_DAT);
        assert_eq!(parse_vtot_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_wscrn_dat("# h\n1.0D+00 2.0D+00 3.0D+00\n")?;
        assert_eq!(parsed.radius_bohr[0], 1.0);
        assert_eq!(parsed.screened_potential[0], 2.0);
        assert_eq!(parsed.core_hole_potential[0], 3.0);
        Ok(())
    }

    #[test]
    fn rejects_bad_screen_tables() {
        assert!(parse_wscrn_dat("").is_err());
        assert!(parse_wscrn_dat("# only a header\n").is_err());
        assert!(parse_wscrn_dat("1 2\n").is_err());
        assert!(parse_wscrn_dat("1 2 3 4\n").is_err());
        assert!(parse_wscrn_dat("1 NaN 3\n").is_err());
        assert!(
            wscrn_dat_string(&WscrnDatData {
                header_lines: Vec::new(),
                radius_bohr: array![1.0, 2.0],
                screened_potential: array![3.0],
                core_hole_potential: array![4.0, 5.0],
            })
            .is_err()
        );
    }

    const WSCRN_DAT: &str = r#" # r       w_scrn(r)      v_ch(r)
    0.1507330463E-03    0.2672882346E+02    0.2916165244E+02
    0.1584612949E-03    0.2672881678E+02    0.2916164576E+02
    0.1665857792E-03    0.2672880306E+02    0.2916163204E+02
"#;

    const VTOT_DAT: &str = r#"    0.1507330463E-03   -0.1922832821E+06    0.2672882346E+02
    0.1584612949E-03   -0.1829001336E+06    0.2672881678E+02
    0.1665857792E-03   -0.1739746063E+06    0.2672880306E+02
"#;
}
