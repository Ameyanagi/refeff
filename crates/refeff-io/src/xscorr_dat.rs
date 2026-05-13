//! FEFF XSCORR intermediate table readers.
//!
//! The FF2X `xscorr` convolution writes diagnostic/intermediate tables during
//! spectrum post-processing: `prexmu.dat`, `residue.dat`, `contour.dat`,
//! `curve.dat`, and `raw.dat`. They are plain formatted numeric files, but the
//! Rust port keeps them typed so reference runs can be checked without
//! ad-hoc parsing.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;

use crate::error::{IoError, Result};

/// Three-column FEFF XSCORR table: real energy and one complex value.
#[derive(Debug, Clone, PartialEq)]
pub struct XscorrComplexTable {
    /// Real energy grid in Hartree.
    pub energy_hartree: Array1<f64>,
    /// Complex table value for each energy point.
    pub values: Array1<Complex64>,
}

/// Four-column FEFF `curve.dat` contour table.
#[derive(Debug, Clone, PartialEq)]
pub struct XscorrCurveDatData {
    /// Complex contour energy points.
    pub energy: Array1<Complex64>,
    /// Complex function values on the contour.
    pub values: Array1<Complex64>,
}

/// Parsed contents of FEFF XSCORR `raw.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct XscorrRawDatData {
    /// First scalar header value, written by FEFF as `Temperature (Hatree)`.
    pub temperature_hartree: f64,
    /// Electronic temperature in eV.
    pub electronic_temperature_ev: f64,
    /// Loss broadening in eV.
    pub loss_ev: f64,
    /// Fermi energy in eV.
    pub fermi_energy_ev: f64,
    /// Number of pole terms included in the convolution.
    pub pole_count: usize,
    /// Real energy grid in Hartree.
    pub omega_hartree: Array1<f64>,
    /// Complex cumulative convolution value.
    pub cchi: Array1<Complex64>,
    /// FEFF `1-Fermi` / step-factor column.
    pub one_minus_fermi: Array1<f64>,
    /// Complex reference `xmu0` values.
    pub xmu0: Array1<Complex64>,
}

impl XscorrComplexTable {
    /// Number of table rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.energy_hartree.len()
    }
}

impl XscorrCurveDatData {
    /// Number of contour rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.energy.len()
    }
}

impl XscorrRawDatData {
    /// Number of raw-convolution rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.omega_hartree.len()
    }
}

/// Parse FEFF `prexmu.dat` text.
pub fn parse_prexmu_dat(text: &str) -> Result<XscorrComplexTable> {
    parse_complex_table("prexmu.dat", text)
}

/// Render FEFF-compatible `prexmu.dat` text.
pub fn prexmu_dat_string(data: &XscorrComplexTable) -> Result<String> {
    complex_table_string("prexmu.dat", data)
}

/// Read FEFF `prexmu.dat` text from a file.
pub fn read_prexmu_dat(path: impl AsRef<Path>) -> Result<XscorrComplexTable> {
    read_complex_table(path, parse_prexmu_dat)
}

/// Write FEFF `prexmu.dat` text to a file.
pub fn write_prexmu_dat(path: impl AsRef<Path>, data: &XscorrComplexTable) -> Result<()> {
    write_text(path, prexmu_dat_string(data)?)
}

/// Parse FEFF `residue.dat` text.
pub fn parse_residue_dat(text: &str) -> Result<XscorrComplexTable> {
    parse_complex_table("residue.dat", text)
}

/// Render FEFF-compatible `residue.dat` text.
pub fn residue_dat_string(data: &XscorrComplexTable) -> Result<String> {
    complex_table_string("residue.dat", data)
}

/// Read FEFF `residue.dat` text from a file.
pub fn read_residue_dat(path: impl AsRef<Path>) -> Result<XscorrComplexTable> {
    read_complex_table(path, parse_residue_dat)
}

/// Write FEFF `residue.dat` text to a file.
pub fn write_residue_dat(path: impl AsRef<Path>, data: &XscorrComplexTable) -> Result<()> {
    write_text(path, residue_dat_string(data)?)
}

/// Parse FEFF `contour.dat` text.
pub fn parse_contour_dat(text: &str) -> Result<XscorrComplexTable> {
    parse_complex_table("contour.dat", text)
}

/// Render FEFF-compatible `contour.dat` text.
pub fn contour_dat_string(data: &XscorrComplexTable) -> Result<String> {
    complex_table_string("contour.dat", data)
}

/// Read FEFF `contour.dat` text from a file.
pub fn read_contour_dat(path: impl AsRef<Path>) -> Result<XscorrComplexTable> {
    read_complex_table(path, parse_contour_dat)
}

/// Write FEFF `contour.dat` text to a file.
pub fn write_contour_dat(path: impl AsRef<Path>, data: &XscorrComplexTable) -> Result<()> {
    write_text(path, contour_dat_string(data)?)
}

/// Parse FEFF `curve.dat` text.
pub fn parse_curve_dat(text: &str) -> Result<XscorrCurveDatData> {
    let path = "curve.dat";
    let mut energy = Vec::new();
    let mut values = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let row = parse_numeric_row(path, line_number, line, 4)?;
        energy.push(Complex64::new(row[0], row[1]));
        values.push(Complex64::new(row[2], row[3]));
    }
    let data = XscorrCurveDatData {
        energy: Array1::from_vec(energy),
        values: Array1::from_vec(values),
    };
    validate_curve_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `curve.dat` text.
pub fn curve_dat_string(data: &XscorrCurveDatData) -> Result<String> {
    validate_curve_dat(data)?;
    let mut out = String::new();
    for (energy, value) in data.energy.iter().zip(data.values.iter()) {
        writeln!(
            out,
            "{:20.10E} {:20.10E} {:20.10E} {:20.10E}",
            energy.re, energy.im, value.re, value.im
        )?;
    }
    Ok(out)
}

/// Read FEFF `curve.dat` text from a file.
pub fn read_curve_dat(path: impl AsRef<Path>) -> Result<XscorrCurveDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_curve_dat(&text)
}

/// Write FEFF `curve.dat` text to a file.
pub fn write_curve_dat(path: impl AsRef<Path>, data: &XscorrCurveDatData) -> Result<()> {
    write_text(path, curve_dat_string(data)?)
}

/// Parse FEFF XSCORR `raw.dat` text.
pub fn parse_xscorr_raw_dat(text: &str) -> Result<XscorrRawDatData> {
    let path = "raw.dat";
    let mut temperature_hartree = None;
    let mut electronic_temperature_ev = None;
    let mut loss_ev = None;
    let mut fermi_energy_ev = None;
    let mut pole_count = None;
    let mut omega_hartree = Vec::new();
    let mut cchi = Vec::new();
    let mut one_minus_fermi = Vec::new();
    let mut xmu0 = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        if is_numeric_line(line) {
            let row = parse_numeric_row(path, line_number, line, 6)?;
            omega_hartree.push(row[0]);
            cchi.push(Complex64::new(row[1], row[2]));
            one_minus_fermi.push(row[3]);
            xmu0.push(Complex64::new(row[4], row[5]));
            continue;
        }
        parse_raw_header_line(
            line_number,
            line,
            &mut temperature_hartree,
            &mut electronic_temperature_ev,
            &mut loss_ev,
            &mut fermi_energy_ev,
            &mut pole_count,
        )?;
    }

    let data = XscorrRawDatData {
        temperature_hartree: temperature_hartree
            .ok_or_else(|| parse_error_value(path, 0, "missing Temperature header"))?,
        electronic_temperature_ev: electronic_temperature_ev
            .ok_or_else(|| parse_error_value(path, 0, "missing Electronic Temperature header"))?,
        loss_ev: loss_ev.ok_or_else(|| parse_error_value(path, 0, "missing xloss header"))?,
        fermi_energy_ev: fermi_energy_ev
            .ok_or_else(|| parse_error_value(path, 0, "missing efermi header"))?,
        pole_count: pole_count
            .ok_or_else(|| parse_error_value(path, 0, "missing Number of poles header"))?,
        omega_hartree: Array1::from_vec(omega_hartree),
        cchi: Array1::from_vec(cchi),
        one_minus_fermi: Array1::from_vec(one_minus_fermi),
        xmu0: Array1::from_vec(xmu0),
    };
    validate_raw_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible XSCORR `raw.dat` text.
pub fn xscorr_raw_dat_string(data: &XscorrRawDatData) -> Result<String> {
    validate_raw_dat(data)?;
    let mut out = String::new();
    writeln!(out, " Temperature (Hatree) = {}", data.temperature_hartree)?;
    writeln!(
        out,
        " Electronic Temperature (eV) = {}",
        data.electronic_temperature_ev
    )?;
    writeln!(out, " xloss = {:24.16}       eV", data.loss_ev)?;
    writeln!(out, " efermi = {:24.16}       eV", data.fermi_energy_ev)?;
    writeln!(out, " Number of poles = {}", data.pole_count)?;
    writeln!(
        out,
        " Omega(Hart)    Re CCHI     Im CCHI   1-Fermi   Re xmu0    Im xmu0"
    )?;
    for (((omega, cchi), one_minus_fermi), xmu0) in data
        .omega_hartree
        .iter()
        .zip(data.cchi.iter())
        .zip(data.one_minus_fermi.iter())
        .zip(data.xmu0.iter())
    {
        writeln!(
            out,
            "{omega:20.10E} {:20.10E} {:20.10E} {:20.10E} {:20.10E} {:20.10E}",
            cchi.re, cchi.im, one_minus_fermi, xmu0.re, xmu0.im
        )?;
    }
    Ok(out)
}

/// Read FEFF XSCORR `raw.dat` text from a file.
pub fn read_xscorr_raw_dat(path: impl AsRef<Path>) -> Result<XscorrRawDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xscorr_raw_dat(&text)
}

/// Write FEFF XSCORR `raw.dat` text to a file.
pub fn write_xscorr_raw_dat(path: impl AsRef<Path>, data: &XscorrRawDatData) -> Result<()> {
    write_text(path, xscorr_raw_dat_string(data)?)
}

fn parse_complex_table(path: &'static str, text: &str) -> Result<XscorrComplexTable> {
    let mut energy_hartree = Vec::new();
    let mut values = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let row = parse_numeric_row(path, line_number, line, 3)?;
        energy_hartree.push(row[0]);
        values.push(Complex64::new(row[1], row[2]));
    }
    let data = XscorrComplexTable {
        energy_hartree: Array1::from_vec(energy_hartree),
        values: Array1::from_vec(values),
    };
    validate_complex_table(path, &data)?;
    Ok(data)
}

fn complex_table_string(path: &'static str, data: &XscorrComplexTable) -> Result<String> {
    validate_complex_table(path, data)?;
    let mut out = String::new();
    for (energy, value) in data.energy_hartree.iter().zip(data.values.iter()) {
        writeln!(
            out,
            "{energy:20.10E} {:20.10E} {:20.10E}",
            value.re, value.im
        )?;
    }
    Ok(out)
}

fn read_complex_table(
    path: impl AsRef<Path>,
    parse: impl FnOnce(&str) -> Result<XscorrComplexTable>,
) -> Result<XscorrComplexTable> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse(&text)
}

fn write_text(path: impl AsRef<Path>, text: String) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, text).map_err(|source| IoError::io(path, source))
}

fn parse_raw_header_line(
    line_number: usize,
    line: &str,
    temperature_hartree: &mut Option<f64>,
    electronic_temperature_ev: &mut Option<f64>,
    loss_ev: &mut Option<f64>,
    fermi_energy_ev: &mut Option<f64>,
    pole_count: &mut Option<usize>,
) -> Result<()> {
    let path = "raw.dat";
    if let Some(value) = header_value(line, "Temperature (Hatree) =") {
        *temperature_hartree = Some(parse_f64(path, line_number, "temperature_hartree", value)?);
    } else if let Some(value) = header_value(line, "Electronic Temperature (eV) =") {
        *electronic_temperature_ev = Some(parse_f64(
            path,
            line_number,
            "electronic_temperature_ev",
            value,
        )?);
    } else if let Some(value) = header_value(line, "xloss =") {
        *loss_ev = Some(parse_f64(path, line_number, "loss_ev", value)?);
    } else if let Some(value) = header_value(line, "efermi =") {
        *fermi_energy_ev = Some(parse_f64(path, line_number, "fermi_energy_ev", value)?);
    } else if let Some(value) = header_value(line, "Number of poles =") {
        *pole_count = Some(parse_usize(path, line_number, "pole_count", value)?);
    } else if !line.starts_with("Omega(Hart)") {
        return parse_error(
            path,
            line_number,
            format!("unexpected raw.dat header line {line:?}"),
        );
    }
    Ok(())
}

fn header_value<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.strip_prefix(prefix)
        .and_then(|rest| rest.split_whitespace().next())
}

fn parse_numeric_row(
    path: &'static str,
    line_number: usize,
    line: &str,
    expected_width: usize,
) -> Result<Vec<f64>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != expected_width {
        return parse_error(
            path,
            line_number,
            format!(
                "row has {} token(s), expected {expected_width}",
                tokens.len()
            ),
        );
    }
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| parse_f64(path, line_number, numeric_field(index), token))
        .collect()
}

fn numeric_field(index: usize) -> &'static str {
    match index {
        0 => "column_0",
        1 => "column_1",
        2 => "column_2",
        3 => "column_3",
        4 => "column_4",
        5 => "column_5",
        _ => "column",
    }
}

fn validate_complex_table(path: &'static str, data: &XscorrComplexTable) -> Result<()> {
    if data.row_count() == 0 {
        return parse_error(path, 0, "at least one table row is required");
    }
    if data.values.len() != data.row_count() {
        return parse_error(
            path,
            0,
            format!(
                "value count {} does not match energy count {}",
                data.values.len(),
                data.row_count()
            ),
        );
    }
    for (index, (energy, value)) in data
        .energy_hartree
        .iter()
        .zip(data.values.iter())
        .enumerate()
    {
        let row = index + 1;
        validate_finite(path, "energy_hartree", *energy, row)?;
        validate_complex(path, "value", *value, row)?;
    }
    Ok(())
}

fn validate_curve_dat(data: &XscorrCurveDatData) -> Result<()> {
    let path = "curve.dat";
    if data.row_count() == 0 {
        return parse_error(path, 0, "at least one curve row is required");
    }
    if data.values.len() != data.row_count() {
        return parse_error(
            path,
            0,
            format!(
                "value count {} does not match energy count {}",
                data.values.len(),
                data.row_count()
            ),
        );
    }
    for (index, (energy, value)) in data.energy.iter().zip(data.values.iter()).enumerate() {
        let row = index + 1;
        validate_complex(path, "energy", *energy, row)?;
        validate_complex(path, "value", *value, row)?;
    }
    Ok(())
}

fn validate_raw_dat(data: &XscorrRawDatData) -> Result<()> {
    let path = "raw.dat";
    validate_finite(path, "temperature_hartree", data.temperature_hartree, 1)?;
    validate_finite(
        path,
        "electronic_temperature_ev",
        data.electronic_temperature_ev,
        2,
    )?;
    validate_finite(path, "loss_ev", data.loss_ev, 3)?;
    validate_finite(path, "fermi_energy_ev", data.fermi_energy_ev, 4)?;
    if data.row_count() == 0 {
        return parse_error(path, 0, "at least one raw data row is required");
    }
    let row_count = data.row_count();
    validate_len(path, "cchi", data.cchi.len(), row_count)?;
    validate_len(
        path,
        "one_minus_fermi",
        data.one_minus_fermi.len(),
        row_count,
    )?;
    validate_len(path, "xmu0", data.xmu0.len(), row_count)?;
    for (index, (((omega, cchi), one_minus_fermi), xmu0)) in data
        .omega_hartree
        .iter()
        .zip(data.cchi.iter())
        .zip(data.one_minus_fermi.iter())
        .zip(data.xmu0.iter())
        .enumerate()
    {
        let row = index + 1;
        validate_finite(path, "omega_hartree", *omega, row)?;
        validate_complex(path, "cchi", *cchi, row)?;
        validate_finite(path, "one_minus_fermi", *one_minus_fermi, row)?;
        validate_complex(path, "xmu0", *xmu0, row)?;
    }
    Ok(())
}

fn validate_len(
    path: &'static str,
    field: &'static str,
    len: usize,
    expected: usize,
) -> Result<()> {
    if len == expected {
        Ok(())
    } else {
        parse_error(
            path,
            0,
            format!("{field} has {len} row(s), expected {expected}"),
        )
    }
}

fn validate_complex(
    path: &'static str,
    field: &'static str,
    value: Complex64,
    row: usize,
) -> Result<()> {
    validate_finite(path, field, value.re, row)?;
    validate_finite(path, field, value.im, row)
}

fn validate_finite(path: &'static str, field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(path, row, format!("{field} must be finite"))
    }
}

fn is_numeric_line(line: &str) -> bool {
    line.split_whitespace()
        .next()
        .is_some_and(|token| token.replace(['D', 'd'], "E").parse::<f64>().is_ok())
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.replace(['D', 'd'], "E").parse::<f64>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })
}

fn parse_usize(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })
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
    use super::*;

    #[test]
    fn parses_three_column_complex_tables() -> Result<()> {
        let parsed = parse_prexmu_dat(PREXMU_DAT)?;
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.energy_hartree[0], -0.138_801_301_5);
        assert_eq!(parsed.values[0].re, -0.000_020_637_731_56);
        assert_eq!(parsed.values[1].im, 0.000_123_685_052_9);
        assert_eq!(parse_prexmu_dat(&prexmu_dat_string(&parsed)?)?, parsed);
        assert_eq!(parse_residue_dat(PREXMU_DAT)?.row_count(), 2);
        assert_eq!(parse_contour_dat(PREXMU_DAT)?.row_count(), 2);
        Ok(())
    }

    #[test]
    fn parses_curve_dat() -> Result<()> {
        let parsed = parse_curve_dat(CURVE_DAT)?;
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(
            parsed.energy[0],
            Complex64::new(-0.138_801_301_5, 0.000_183_746_545)
        );
        assert_eq!(
            parsed.values[1],
            Complex64::new(-0.000_028_683, 0.000_237_44)
        );
        assert_eq!(parse_curve_dat(&curve_dat_string(&parsed)?)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_xscorr_raw_dat() -> Result<()> {
        let parsed = parse_xscorr_raw_dat(RAW_DAT)?;
        assert_eq!(parsed.temperature_hartree, 0.0);
        assert_eq!(parsed.electronic_temperature_ev, 0.0);
        assert_eq!(parsed.loss_ev, 0.864_589_999_999_999_9);
        assert_eq!(parsed.fermi_energy_ev, -3.776_977_180_000_000_3);
        assert_eq!(parsed.pole_count, 0);
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.cchi[0].re, -0.000_016_299_5);
        assert_eq!(parsed.one_minus_fermi[1], 0.514_017_875_2);
        assert_eq!(
            parse_xscorr_raw_dat(&xscorr_raw_dat_string(&parsed)?)?,
            parsed
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_xscorr_dat_inputs() {
        assert!(parse_prexmu_dat("").is_err());
        assert!(parse_prexmu_dat("1 2\n").is_err());
        assert!(parse_prexmu_dat("1 NaN 2\n").is_err());
        assert!(parse_curve_dat("1 2 3\n").is_err());
        assert!(parse_curve_dat("1 2 3 NaN\n").is_err());
        assert!(parse_xscorr_raw_dat("").is_err());
        assert!(parse_xscorr_raw_dat(&RAW_DAT.replace("xloss =", "bad =")).is_err());
        assert!(parse_xscorr_raw_dat(&RAW_DAT.replace("0.5000000000E+000", "NaN")).is_err());
    }

    const PREXMU_DAT: &str = r#"  -0.1388013015E+000  -0.2063773156E-004   0.1203227708E-003
  -0.1374011587E+000  -0.2117776391E-004   0.1236850529E-003
"#;

    const CURVE_DAT: &str = r#"  -0.1388013015E+000   0.1837465450E-003  -0.2866200000E-004   0.2374800000E-003
  -0.1388013015E+000   0.3674930900E-003  -0.2868300000E-004   0.2374400000E-003
"#;

    const RAW_DAT: &str = r#" Temperature (Hatree) = 0
 Electronic Temperature (eV) = 0
 xloss =   0.86458999999999986       eV
 efermi =   -3.7769771800000003       eV
 Number of poles = 0
 Omega(Hart)    Re CCHI     Im CCHI   1-Fermi   Re xmu0    Im xmu0
  -0.1388013015E+000  -0.1629950000E-004   0.1152400000E-003   0.5000000000E+000  -0.3259900000E-004   0.2304800000E-003
  -0.1374011587E+000  -0.1689833765E-004   0.1185582229E-003   0.5140178752E+000  -0.3287500000E-004   0.2306500000E-003
"#;
}
