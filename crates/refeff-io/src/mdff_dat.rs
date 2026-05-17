//! FEFF `mdff.dat` EELS-MDFF spectrum codec.
//!
//! The EELS-MDFF program writes comment headers followed by rows containing
//! energy loss and complex spectrum channels. FEFF formats each complex value
//! as adjacent real and imaginary `G14.6` fields.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::write_fortran_g;

const MDFF_DAT_PATH: &str = "mdff.dat";

/// Parsed FEFF `mdff.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct MdffDatData {
    /// Header and comment lines before and around the numeric spectrum table.
    pub header_lines: Vec<String>,
    /// Energy loss in eV.
    pub energy_loss_ev: Array1<f64>,
    /// Complex MDFF/EELS spectrum channels, shaped `(energy, channel)`.
    pub spectrum: Array2<Complex64>,
}

impl MdffDatData {
    /// Number of spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_loss_ev.len()
    }

    /// Number of complex spectrum channels per row.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.spectrum.len_of(Axis(1))
    }
}

/// Parse FEFF `mdff.dat` text.
pub fn parse_mdff_dat(text: &str) -> Result<MdffDatData> {
    let mut header_lines = Vec::new();
    let mut energy_loss_ev = Vec::new();
    let mut spectrum = Vec::new();
    let mut channel_count = None;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let row = parse_mdff_row(line_number, &tokens)?;
            match channel_count {
                Some(expected) if row.channels.len() != expected => {
                    return parse_error(
                        line_number,
                        format!(
                            "row has {} channel(s), expected {expected}",
                            row.channels.len()
                        ),
                    );
                }
                Some(_) => {}
                None => channel_count = Some(row.channels.len()),
            }
            energy_loss_ev.push(row.energy_loss_ev);
            spectrum.extend(row.channels);
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let point_count = energy_loss_ev.len();
    let channel_count =
        channel_count.ok_or_else(|| parse_error_value(0, "at least one MDFF row is required"))?;
    let spectrum = Array2::from_shape_vec((point_count, channel_count), spectrum)
        .map_err(|source| parse_error_value(0, format!("invalid MDFF spectrum shape: {source}")))?;
    let data = MdffDatData {
        header_lines,
        energy_loss_ev: Array1::from_vec(energy_loss_ev),
        spectrum,
    };
    validate_mdff_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `mdff.dat` text.
pub fn mdff_dat_string(data: &MdffDatData) -> Result<String> {
    validate_mdff_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, row) in data
        .energy_loss_ev
        .iter()
        .zip(data.spectrum.axis_iter(Axis(0)))
    {
        write_fortran_g(&mut out, *energy, 14, 6)?;
        for value in row {
            write_fortran_g(&mut out, value.re, 14, 6)?;
            write_fortran_g(&mut out, value.im, 14, 6)?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Read FEFF `mdff.dat` text from a file.
pub fn read_mdff_dat(path: impl AsRef<Path>) -> Result<MdffDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_mdff_dat(&text)
}

/// Write FEFF `mdff.dat` text to a file.
pub fn write_mdff_dat(path: impl AsRef<Path>, data: &MdffDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, mdff_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

struct MdffRow {
    energy_loss_ev: f64,
    channels: Vec<Complex64>,
}

fn parse_mdff_row(line_number: usize, tokens: &[&str]) -> Result<MdffRow> {
    let payload_tokens = tokens.len().saturating_sub(1);
    if payload_tokens == 0 || !payload_tokens.is_multiple_of(2) {
        return parse_error(
            line_number,
            format!("row has {payload_tokens} spectrum token(s), expected a positive even count"),
        );
    }

    let energy_loss_ev = parse_f64(line_number, "energy loss", tokens[0])?;
    let channels = tokens[1..]
        .chunks_exact(2)
        .map(|pair| {
            let real = parse_f64(line_number, "channel real", pair[0])?;
            let imaginary = parse_f64(line_number, "channel imaginary", pair[1])?;
            Ok(Complex64::new(real, imaginary))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(MdffRow {
        energy_loss_ev,
        channels,
    })
}

fn validate_mdff_dat(data: &MdffDatData) -> Result<()> {
    if data.point_count() == 0 {
        return parse_error(0, "at least one MDFF row is required");
    }
    if data.spectrum.len_of(Axis(0)) != data.point_count() {
        return parse_error(
            0,
            format!(
                "spectrum row count {} does not match energy count {}",
                data.spectrum.len_of(Axis(0)),
                data.point_count()
            ),
        );
    }
    if data.channel_count() == 0 {
        return parse_error(0, "at least one MDFF channel is required");
    }
    for (index, energy) in data.energy_loss_ev.iter().enumerate() {
        let row = index + 1;
        validate_finite("energy loss", *energy, row)?;
        for value in data.spectrum.row(index) {
            validate_complex("spectrum", *value, row)?;
        }
    }
    Ok(())
}

fn validate_complex(field: &'static str, value: Complex64, row: usize) -> Result<()> {
    validate_finite(field, value.re, row)?;
    validate_finite(field, value.im, row)
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(row, format!("{field} must be finite"))
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: MDFF_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn parses_feff_mdff_rows() -> Result<()> {
        let data = parse_mdff_dat(MDFF_DAT)?;

        assert_eq!(
            data.header_lines,
            vec![
                "# Orientation sensitive EELS calculation - beam energy =    300keV",
                "#  Energy       total"
            ]
        );
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.channel_count(), 2);
        assert_eq!(data.energy_loss_ev[0], 10.0);
        assert_eq!(data.spectrum[(0, 0)], Complex64::new(1.0, 0.25));
        assert_eq!(data.spectrum[(1, 1)], Complex64::new(0.8, -0.05));
        Ok(())
    }

    #[test]
    fn roundtrips_mdff_text() -> Result<()> {
        let data = sample_mdff_dat()?;
        let rendered = mdff_dat_string(&data)?;

        assert_eq!(parse_mdff_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_mdff_text() {
        assert!(parse_mdff_dat("").is_err());
        assert!(parse_mdff_dat("1.0\n").is_err());
        assert!(parse_mdff_dat("1.0 2.0 3.0\n2.0 3.0 4.0 5.0 6.0\n").is_err());
        assert!(parse_mdff_dat("1.0 NaN 0.0\n").is_err());

        let mut spectrum = Array2::zeros((1, 1));
        spectrum[(0, 0)] = Complex64::new(f64::NAN, 0.0);
        let bad = MdffDatData {
            header_lines: Vec::new(),
            energy_loss_ev: Array1::from_vec(vec![1.0]),
            spectrum,
        };
        assert!(mdff_dat_string(&bad).is_err());
    }

    fn sample_mdff_dat() -> Result<MdffDatData> {
        Ok(MdffDatData {
            header_lines: vec!["# sample mdff".to_string()],
            energy_loss_ev: Array1::from_vec(vec![10.0, 12.5]),
            spectrum: Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.25),
                    Complex64::new(0.5, -0.1),
                    Complex64::new(1.2, 0.2),
                    Complex64::new(0.8, -0.05),
                ],
            )
            .map_err(|source| parse_error_value(0, format!("invalid test shape: {source}")))?,
        })
    }

    const MDFF_DAT: &str = concat!(
        "# Orientation sensitive EELS calculation - beam energy =    300keV\n",
        "#  Energy       total\n",
        "       10.0000       1.00000      0.250000      0.500000     -0.100000\n",
        "       12.5000       1.20000      0.200000      0.800000    -0.0500000\n",
    );
}
