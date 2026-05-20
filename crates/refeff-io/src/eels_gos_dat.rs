//! FEFF EELS generalized-oscillator-strength text codecs.
//!
//! `EELS/writeangulardependence3.f90` writes `gos1.txt` as a two-column
//! q-slice and `gos2.txt` as a compact table with hardcoded GOS metadata and
//! 20 q-grid values for every energy row.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use refeff_core::EelsGosTable;

use crate::error::{IoError, Result};
use crate::format::{fortran_list_directed_f64, write_fortran_zero_scaled_exp};

/// Parsed FEFF `gos1.txt` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsGos1DatData {
    /// FEFF GOS q values.
    pub q_values: Array1<f64>,
    /// GOS strength at FEFF's middle-energy slice, `xq(:, ne/2+1)`.
    pub strengths: Array1<f64>,
}

/// Parsed FEFF `gos2.txt` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsGos2DatData {
    /// FEFF element marker, hardcoded as `OXYG` in the reference routine.
    pub element_label: String,
    /// FEFF edge marker, hardcoded as `1S1/2` in the reference routine.
    pub edge_label: String,
    /// Header value `info1_1`.
    pub q_scale: f64,
    /// Header value `info1_2`.
    pub q_log_step: f64,
    /// Header value `info1_3`.
    pub edge_parameter: f64,
    /// Header value `info2_1`.
    pub energy_start_ev: f64,
    /// Header value `info2_2`.
    pub energy_step_ev: f64,
    /// FEFF `xq(1:nqq,1:ne)` strengths as `(q, energy)`.
    pub strengths: Array2<f64>,
}

impl EelsGos1DatData {
    /// Number of q rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.q_values.len()
    }
}

impl EelsGos2DatData {
    /// Number of q-grid rows.
    #[must_use]
    pub fn q_count(&self) -> usize {
        self.strengths.nrows()
    }

    /// Number of energy rows.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.strengths.ncols()
    }
}

/// Build FEFF `gos1.txt` and `gos2.txt` data from the core GOS table.
#[must_use]
pub fn eels_gos_dat_from_table(table: EelsGosTable) -> (EelsGos1DatData, EelsGos2DatData) {
    let middle_energy = table.strengths.ncols() / 2;
    let gos1 = EelsGos1DatData {
        q_values: table.q_values,
        strengths: table.strengths.column(middle_energy).to_owned(),
    };
    let gos2 = EelsGos2DatData {
        element_label: "OXYG".to_string(),
        edge_label: "1S1/2".to_string(),
        q_scale: table.q_scale,
        q_log_step: table.q_log_step,
        edge_parameter: table.edge_parameter,
        energy_start_ev: table.energy_start_ev,
        energy_step_ev: table.energy_step_ev,
        strengths: table.strengths,
    };
    (gos1, gos2)
}

/// Render FEFF-compatible `gos1.txt` text.
pub fn eels_gos1_dat_string(data: &EelsGos1DatData) -> Result<String> {
    validate_gos1_dat(data)?;
    let mut out = String::new();
    for (&q, &strength) in data.q_values.iter().zip(data.strengths.iter()) {
        writeln!(
            out,
            "{}{}",
            fortran_list_directed_f64(q),
            fortran_list_directed_f64(strength)
        )?;
    }
    Ok(out)
}

/// Render FEFF-compatible `gos2.txt` text.
pub fn eels_gos2_dat_string(data: &EelsGos2DatData) -> Result<String> {
    validate_gos2_dat(data)?;
    let mut out = String::new();
    writeln!(out, "{:<4}", data.element_label)?;
    writeln!(
        out,
        "{:>6}{:7.4}{:7.4}{:6.1}{:3}",
        data.edge_label,
        data.q_scale,
        data.q_log_step,
        data.edge_parameter,
        data.q_count()
    )?;
    writeln!(
        out,
        "{:8.2}{:8.2}{:3}",
        data.energy_start_ev,
        data.energy_step_ev,
        data.energy_count()
    )?;
    for energy in 0..data.energy_count() {
        for (index, value) in data.strengths.column(energy).iter().enumerate() {
            if index > 0 && index % 5 == 0 {
                out.push('\n');
            }
            write_fortran_zero_scaled_exp(&mut out, *value, 16, 8)?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Parse FEFF `gos1.txt` text.
pub fn parse_eels_gos1_dat(text: &str) -> Result<EelsGos1DatData> {
    let mut q_values = Vec::new();
    let mut strengths = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return Err(parse_error(
                "gos1.txt",
                line_number,
                format!("gos1.txt row has {} token(s), expected 2", tokens.len()),
            ));
        }
        q_values.push(parse_f64("gos1.txt", line_number, "q", tokens[0])?);
        strengths.push(parse_f64("gos1.txt", line_number, "strength", tokens[1])?);
    }
    let data = EelsGos1DatData {
        q_values: Array1::from_vec(q_values),
        strengths: Array1::from_vec(strengths),
    };
    validate_gos1_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `gos2.txt` text.
pub fn parse_eels_gos2_dat(text: &str) -> Result<EelsGos2DatData> {
    let mut lines = text.lines().enumerate();
    let Some((_, element_line)) = lines.next() else {
        return Err(parse_error("gos2.txt", 0, "missing element header"));
    };
    let Some((edge_line_number, edge_line)) = lines.next() else {
        return Err(parse_error("gos2.txt", 0, "missing edge header"));
    };
    let edge_tokens = edge_line.split_whitespace().collect::<Vec<_>>();
    if edge_tokens.len() != 5 {
        return Err(parse_error(
            "gos2.txt",
            edge_line_number + 1,
            format!(
                "gos2.txt edge header has {} token(s), expected 5",
                edge_tokens.len()
            ),
        ));
    }

    let Some((energy_line_number, energy_line)) = lines.next() else {
        return Err(parse_error("gos2.txt", 0, "missing energy header"));
    };
    let energy_tokens = energy_line.split_whitespace().collect::<Vec<_>>();
    if energy_tokens.len() != 3 {
        return Err(parse_error(
            "gos2.txt",
            energy_line_number + 1,
            format!(
                "gos2.txt energy header has {} token(s), expected 3",
                energy_tokens.len()
            ),
        ));
    }

    let q_count = parse_usize("gos2.txt", edge_line_number + 1, "q_count", edge_tokens[4])?;
    let energy_count = parse_usize(
        "gos2.txt",
        energy_line_number + 1,
        "energy_count",
        energy_tokens[2],
    )?;
    let mut flat = Vec::with_capacity(q_count * energy_count);
    for (index, raw) in lines {
        let line_number = index + 1;
        for token in raw.split_whitespace() {
            flat.push(parse_f64("gos2.txt", line_number, "strength", token)?);
        }
    }
    if flat.len() != q_count * energy_count {
        return Err(parse_error(
            "gos2.txt",
            0,
            format!(
                "gos2.txt has {} strength value(s), expected {}",
                flat.len(),
                q_count * energy_count
            ),
        ));
    }
    let mut strengths = Array2::zeros((q_count, energy_count));
    for energy in 0..energy_count {
        for q in 0..q_count {
            strengths[(q, energy)] = flat[energy * q_count + q];
        }
    }

    let data = EelsGos2DatData {
        element_label: element_line.trim().to_string(),
        edge_label: edge_tokens[0].to_string(),
        q_scale: parse_f64("gos2.txt", edge_line_number + 1, "q_scale", edge_tokens[1])?,
        q_log_step: parse_f64(
            "gos2.txt",
            edge_line_number + 1,
            "q_log_step",
            edge_tokens[2],
        )?,
        edge_parameter: parse_f64(
            "gos2.txt",
            edge_line_number + 1,
            "edge_parameter",
            edge_tokens[3],
        )?,
        energy_start_ev: parse_f64(
            "gos2.txt",
            energy_line_number + 1,
            "energy_start_ev",
            energy_tokens[0],
        )?,
        energy_step_ev: parse_f64(
            "gos2.txt",
            energy_line_number + 1,
            "energy_step_ev",
            energy_tokens[1],
        )?,
        strengths,
    };
    validate_gos2_dat(&data)?;
    Ok(data)
}

/// Write FEFF `gos1.txt`.
pub fn write_eels_gos1_dat(path: impl AsRef<Path>, data: &EelsGos1DatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, eels_gos1_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `gos2.txt`.
pub fn write_eels_gos2_dat(path: impl AsRef<Path>, data: &EelsGos2DatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, eels_gos2_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `gos1.txt`.
pub fn read_eels_gos1_dat(path: impl AsRef<Path>) -> Result<EelsGos1DatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_eels_gos1_dat(&text)
}

/// Read FEFF `gos2.txt`.
pub fn read_eels_gos2_dat(path: impl AsRef<Path>) -> Result<EelsGos2DatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_eels_gos2_dat(&text)
}

fn validate_gos1_dat(data: &EelsGos1DatData) -> Result<()> {
    if data.point_count() == 0 {
        return invalid_gos_dat("gos1.txt", "rows", "at least one q row is required");
    }
    if data.strengths.len() != data.point_count() {
        return invalid_gos_dat(
            "gos1.txt",
            "strengths",
            format!(
                "got {} strength value(s), expected {}",
                data.strengths.len(),
                data.point_count()
            ),
        );
    }
    validate_finite_array("gos1.txt", "q", data.q_values.view())?;
    validate_finite_array("gos1.txt", "strength", data.strengths.view())
}

fn validate_gos2_dat(data: &EelsGos2DatData) -> Result<()> {
    if data.element_label.trim().is_empty() {
        return invalid_gos_dat("gos2.txt", "element_label", "label cannot be empty");
    }
    if data.edge_label.trim().is_empty() {
        return invalid_gos_dat("gos2.txt", "edge_label", "label cannot be empty");
    }
    for (field, value) in [
        ("q_scale", data.q_scale),
        ("q_log_step", data.q_log_step),
        ("edge_parameter", data.edge_parameter),
        ("energy_start_ev", data.energy_start_ev),
        ("energy_step_ev", data.energy_step_ev),
    ] {
        if !value.is_finite() {
            return invalid_gos_dat("gos2.txt", field, format!("value is not finite: {value}"));
        }
    }
    if data.q_count() == 0 || data.energy_count() == 0 {
        return invalid_gos_dat("gos2.txt", "strengths", "strength matrix cannot be empty");
    }
    for (energy, column) in data.strengths.axis_iter(Axis(1)).enumerate() {
        validate_finite_array("gos2.txt", "strength", column).map_err(|_| {
            parse_error(
                "gos2.txt",
                0,
                format!(
                    "energy column {} contains a non-finite strength",
                    energy + 1
                ),
            )
        })?;
    }
    Ok(())
}

fn validate_finite_array(
    path: &'static str,
    field: &'static str,
    values: ndarray::ArrayView1<'_, f64>,
) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_gos_dat(
                path,
                field,
                format!("row {} is not finite: {value}", index + 1),
            );
        }
    }
    Ok(())
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|source| {
            parse_error(
                path,
                line,
                format!("could not parse {field} from {token:?}: {source}"),
            )
        })
}

fn parse_usize(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|source| {
        parse_error(
            path,
            line,
            format!("could not parse {field} from {token:?}: {source}"),
        )
    })
}

fn invalid_gos_dat(
    path: &'static str,
    field: &'static str,
    message: impl Into<String>,
) -> std::result::Result<(), IoError> {
    Err(parse_error(
        path,
        0,
        format!("invalid {field}: {}", message.into()),
    ))
}

fn parse_error(path: &'static str, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EelsGos1DatData, EelsGos2DatData, eels_gos1_dat_string, eels_gos2_dat_string,
        parse_eels_gos1_dat, parse_eels_gos2_dat,
    };
    use ndarray::{arr1, arr2};

    #[test]
    fn eels_gos1_dat_roundtrips() -> crate::Result<()> {
        let data = EelsGos1DatData {
            q_values: arr1(&[0.050_319_876_699, 0.106_573_941_39]),
            strengths: arr1(&[27_431.800_619_716, 84_730.813_273_548]),
        };

        let text = eels_gos1_dat_string(&data)?;

        assert_eq!(parse_eels_gos1_dat(&text)?, data);
        Ok(())
    }

    #[test]
    fn eels_gos2_dat_roundtrips() -> crate::Result<()> {
        let text = concat!(
            "OXYG\n",
            " 1S1/2 0.6859 0.1294 100.0  2\n",
            "  100.00   10.00  3\n",
            "  0.12001669E+07  0.38413545E+07\n",
            "  0.26069567E+06  0.81093118E+06\n",
            "  0.27431801E+05  0.84730813E+05\n",
        );

        let data = parse_eels_gos2_dat(text)?;

        assert_eq!(data.element_label, "OXYG");
        assert_eq!(data.edge_label, "1S1/2");
        assert_eq!(data.q_count(), 2);
        assert_eq!(data.energy_count(), 3);
        assert_eq!(
            data.strengths,
            arr2(&[
                [1_200_166.9, 260_695.67, 27_431.801],
                [3_841_354.5, 810_931.18, 84_730.813],
            ])
        );
        assert_eq!(parse_eels_gos2_dat(&eels_gos2_dat_string(&data)?)?, data);
        Ok(())
    }

    #[test]
    fn eels_gos_dat_rejects_bad_inputs() {
        assert!(parse_eels_gos1_dat("").is_err());
        assert!(parse_eels_gos1_dat("1.0\n").is_err());
        assert!(parse_eels_gos2_dat("OXYG\n").is_err());
        let bad = EelsGos2DatData {
            element_label: String::new(),
            edge_label: "1S1/2".to_string(),
            q_scale: 1.0,
            q_log_step: 1.0,
            edge_parameter: 100.0,
            energy_start_ev: 100.0,
            energy_step_ev: 10.0,
            strengths: arr2(&[[1.0]]),
        };
        assert!(eels_gos2_dat_string(&bad).is_err());
    }
}
