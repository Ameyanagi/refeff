//! FEFF NRIXS `xmul.dat` angular-decomposition spectrum support.
//!
//! FEFF writes `xmul.dat` from the NRIXS/LDEC paths in `ff2xmujas.f90` and
//! `ff2chijas.f90`. The first two columns are photon energy and momentum, then
//! one total single-electron spectrum column, one diagonal background column
//! for each decomposition channel `l = 0..ldecmx`, and finally the normalized
//! fine-structure matrix in FEFF loop order: `l*` outer, `l` inner.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

/// Parsed FEFF `xmul.dat` angular-decomposition output.
#[derive(Debug, Clone, PartialEq)]
pub struct XmulDatData {
    /// Header/comment lines before the numeric table.
    pub header_lines: Vec<String>,
    /// Maximum decomposition channel index, FEFF `ldecmx`.
    pub max_decomposition_channel: usize,
    /// Photon energy in eV.
    pub photon_energy_ev: Array1<f64>,
    /// Momentum-transfer wave number in inverse Angstrom.
    pub wave_number: Array1<f64>,
    /// Total single-electron NRIXS response, `S^0(q,w)`.
    pub total_single_electron: Array1<f64>,
    /// Diagonal background channel contributions, shaped `(energy, l)`.
    pub channel_background: Array2<f64>,
    /// Normalized fine-structure matrix, shaped `(energy, l_star, l)`.
    pub normalized_fine_structure: Array3<f64>,
}

/// Completed NRIXS decomposition rows used to build FEFF `xmul.dat`.
#[derive(Debug, Clone)]
pub struct XmulDatFromNrixsDecompositionInput<'a> {
    /// Header/comment lines before the numeric table.
    pub header_lines: &'a [String],
    /// Maximum decomposition channel index, FEFF `ldecmx`.
    pub max_decomposition_channel: usize,
    /// Photon energy in eV.
    pub photon_energy_ev: ArrayView1<'a, f64>,
    /// Momentum-transfer wave number in inverse Angstrom.
    pub wave_number: ArrayView1<'a, f64>,
    /// Diagonal background channel contributions, shaped `(energy, l)`.
    pub channel_background: ArrayView2<'a, f64>,
    /// Normalized fine-structure matrix, shaped `(energy, l_star, l)`.
    pub normalized_fine_structure: ArrayView3<'a, f64>,
}

impl XmulDatData {
    /// Number of spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.photon_energy_ev.len()
    }

    /// Number of angular-decomposition channels, equal to `ldecmx + 1`.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.max_decomposition_channel.saturating_add(1)
    }
}

/// Build FEFF NRIXS `xmul.dat` data from completed angular-decomposition rows.
///
/// The adapter preserves the caller's normalized fine-structure matrix and
/// computes FEFF's total single-electron response column as the sum over the
/// diagonal background channels for each energy row.
pub fn xmul_dat_from_nrixs_decomposition(
    input: XmulDatFromNrixsDecompositionInput<'_>,
) -> Result<XmulDatData> {
    validate_xmul_nrixs_input(&input)?;
    let header_lines =
        xmul_nrixs_header_lines(input.header_lines, input.max_decomposition_channel)?;
    let total_single_electron = input
        .channel_background
        .axis_iter(Axis(0))
        .map(|row| row.iter().sum())
        .collect::<Vec<_>>();
    let data = XmulDatData {
        header_lines,
        max_decomposition_channel: input.max_decomposition_channel,
        photon_energy_ev: input.photon_energy_ev.to_owned(),
        wave_number: input.wave_number.to_owned(),
        total_single_electron: Array1::from_vec(total_single_electron),
        channel_background: input.channel_background.to_owned(),
        normalized_fine_structure: input.normalized_fine_structure.to_owned(),
    };
    validate_xmul_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `xmul.dat` text.
pub fn parse_xmul_dat(text: &str) -> Result<XmulDatData> {
    let mut header_lines = Vec::new();
    let mut max_decomposition_channel = None;
    let mut photon_energy_ev = Vec::new();
    let mut wave_number = Vec::new();
    let mut total_single_electron = Vec::new();
    let mut channel_background = Vec::new();
    let mut normalized_fine_structure = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens
            .first()
            .is_some_and(|token| parse_f64_value(token).is_ok())
        {
            let max_channel = max_decomposition_channel
                .ok_or_else(|| invalid_xmul_dat("ldecmx", "missing ldecmx header"))?;
            let channel_count = checked_channel_count(max_channel)?;
            let expected = row_width(channel_count)?;
            if tokens.len() != expected {
                return Err(IoError::XmulDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected,
                });
            }

            photon_energy_ev.push(parse_f64(line_number, "photon_energy_ev", tokens[0])?);
            wave_number.push(parse_f64(line_number, "wave_number", tokens[1])?);
            total_single_electron.push(parse_f64(line_number, "total_single_electron", tokens[2])?);
            for token in &tokens[3..3 + channel_count] {
                channel_background.push(parse_f64(line_number, "channel_background", token)?);
            }
            for token in &tokens[3 + channel_count..] {
                normalized_fine_structure.push(parse_f64(
                    line_number,
                    "normalized_fine_structure",
                    token,
                )?);
            }
        } else {
            if let Some(value) = parse_ldecmx(line, line_number)? {
                max_decomposition_channel = Some(value);
            }
            header_lines.push(raw.to_string());
        }
    }

    let max_decomposition_channel =
        max_decomposition_channel.ok_or_else(|| invalid_xmul_dat("ldecmx", "missing header"))?;
    let channel_count = checked_channel_count(max_decomposition_channel)?;
    let point_count = photon_energy_ev.len();
    let channel_background =
        Array2::from_shape_vec((point_count, channel_count), channel_background).map_err(
            |source| invalid_xmul_dat("channel_background", format!("invalid shape: {source}")),
        )?;
    let normalized_fine_structure = Array3::from_shape_vec(
        (point_count, channel_count, channel_count),
        normalized_fine_structure,
    )
    .map_err(|source| {
        invalid_xmul_dat(
            "normalized_fine_structure",
            format!("invalid shape: {source}"),
        )
    })?;
    let data = XmulDatData {
        header_lines,
        max_decomposition_channel,
        photon_energy_ev: Array1::from_vec(photon_energy_ev),
        wave_number: Array1::from_vec(wave_number),
        total_single_electron: Array1::from_vec(total_single_electron),
        channel_background,
        normalized_fine_structure,
    };
    validate_xmul_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `xmul.dat` text.
pub fn xmul_dat_string(data: &XmulDatData) -> Result<String> {
    validate_xmul_dat(data)?;
    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for row in 0..data.point_count() {
        write!(
            out,
            " {:11.3}{:11.3}",
            data.photon_energy_ev[row], data.wave_number[row]
        )?;
        write_fortran_zero_scaled_exp(&mut out, data.total_single_electron[row], 11, 3)?;
        for value in data.channel_background.row(row) {
            write_fortran_zero_scaled_exp(&mut out, *value, 11, 3)?;
        }
        for value in data.normalized_fine_structure.index_axis(Axis(0), row) {
            write_fortran_zero_scaled_exp(&mut out, *value, 11, 3)?;
        }
        writeln!(out)?;
    }
    Ok(out)
}

/// Read FEFF `xmul.dat` text from a file.
pub fn read_xmul_dat(path: impl AsRef<Path>) -> Result<XmulDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xmul_dat(&text)
}

/// Write FEFF `xmul.dat` text to a file.
pub fn write_xmul_dat(path: impl AsRef<Path>, data: &XmulDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, xmul_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_ldecmx(line: &str, line_number: usize) -> Result<Option<usize>> {
    let lower = line.to_ascii_lowercase();
    let Some(position) = lower.find("ldecmx=") else {
        return Ok(None);
    };
    let value_start = position + "ldecmx=".len();
    let token = line[value_start..]
        .split_whitespace()
        .next()
        .ok_or_else(|| invalid_xmul_dat("ldecmx", "missing value after ldecmx="))?;
    parse_usize(line_number, "ldecmx", token).map(Some)
}

fn validate_xmul_dat(data: &XmulDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_xmul_dat(
            "rows",
            "at least one spectrum row is required",
        ));
    }
    let channel_count = checked_channel_count(data.max_decomposition_channel)?;
    if data.wave_number.len() != point_count {
        return Err(shape_error(
            "wave_number",
            vec![data.wave_number.len()],
            vec![point_count],
        ));
    }
    if data.total_single_electron.len() != point_count {
        return Err(shape_error(
            "total_single_electron",
            vec![data.total_single_electron.len()],
            vec![point_count],
        ));
    }
    if data.channel_background.shape() != [point_count, channel_count] {
        return Err(shape_error(
            "channel_background",
            data.channel_background.shape().to_vec(),
            vec![point_count, channel_count],
        ));
    }
    if data.normalized_fine_structure.shape() != [point_count, channel_count, channel_count] {
        return Err(shape_error(
            "normalized_fine_structure",
            data.normalized_fine_structure.shape().to_vec(),
            vec![point_count, channel_count, channel_count],
        ));
    }
    validate_finite_array1("photon_energy_ev", &data.photon_energy_ev)?;
    validate_finite_array1("wave_number", &data.wave_number)?;
    validate_finite_array1("total_single_electron", &data.total_single_electron)?;
    validate_finite_array2("channel_background", &data.channel_background)?;
    validate_finite_array3("normalized_fine_structure", &data.normalized_fine_structure)?;
    Ok(())
}

fn validate_xmul_nrixs_input(input: &XmulDatFromNrixsDecompositionInput<'_>) -> Result<()> {
    let point_count = input.photon_energy_ev.len();
    if point_count == 0 {
        return Err(invalid_xmul_dat(
            "photon_energy_ev",
            "at least one spectrum row is required",
        ));
    }
    if input.wave_number.len() != point_count {
        return Err(shape_error(
            "wave_number",
            vec![input.wave_number.len()],
            vec![point_count],
        ));
    }
    let channel_count = checked_channel_count(input.max_decomposition_channel)?;
    if input.channel_background.shape() != [point_count, channel_count] {
        return Err(shape_error(
            "channel_background",
            input.channel_background.shape().to_vec(),
            vec![point_count, channel_count],
        ));
    }
    if input.normalized_fine_structure.shape() != [point_count, channel_count, channel_count] {
        return Err(shape_error(
            "normalized_fine_structure",
            input.normalized_fine_structure.shape().to_vec(),
            vec![point_count, channel_count, channel_count],
        ));
    }
    validate_finite_array1_view("photon_energy_ev", input.photon_energy_ev)?;
    validate_finite_array1_view("wave_number", input.wave_number)?;
    validate_finite_array2_view("channel_background", input.channel_background)?;
    validate_finite_array3_view("normalized_fine_structure", input.normalized_fine_structure)?;
    Ok(())
}

fn xmul_nrixs_header_lines(
    input: &[String],
    max_decomposition_channel: usize,
) -> Result<Vec<String>> {
    let mut found_ldecmx = None;
    for (index, line) in input.iter().enumerate() {
        if let Some(value) = parse_ldecmx(line, index + 1)? {
            found_ldecmx = Some(value);
        }
    }
    if let Some(value) = found_ldecmx {
        if value != max_decomposition_channel {
            return Err(invalid_xmul_dat(
                "ldecmx",
                format!(
                    "header ldecmx {value} does not match max decomposition channel {max_decomposition_channel}"
                ),
            ));
        }
        return Ok(input.to_vec());
    }
    Ok(vec![
        "#  Decomposition of S(q,w) for a single electron".to_string(),
        "#  omega    k   S^0(qw)  S_{l=0,...,ldecmx}^0(qw)       chi^q_{l=0,..ldecmx,l^*=0,...,ldecmx}".to_string(),
        format!("# and ldecmx= {max_decomposition_channel:>5}"),
    ])
}

fn row_width(channel_count: usize) -> Result<usize> {
    channel_count
        .checked_mul(channel_count)
        .and_then(|matrix| 3_usize.checked_add(channel_count)?.checked_add(matrix))
        .ok_or_else(|| invalid_xmul_dat("ldecmx", "row width overflow"))
}

fn checked_channel_count(max_decomposition_channel: usize) -> Result<usize> {
    max_decomposition_channel
        .checked_add(1)
        .ok_or_else(|| invalid_xmul_dat("ldecmx", "channel count overflow"))
}

fn validate_finite_array1(field: &'static str, values: &Array1<f64>) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_xmul_dat(field, "all values must be finite"))
    }
}

fn validate_finite_array1_view(field: &'static str, values: ArrayView1<'_, f64>) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_xmul_dat(field, "all values must be finite"))
    }
}

fn validate_finite_array2(field: &'static str, values: &Array2<f64>) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_xmul_dat(field, "all values must be finite"))
    }
}

fn validate_finite_array2_view(field: &'static str, values: ArrayView2<'_, f64>) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_xmul_dat(field, "all values must be finite"))
    }
}

fn validate_finite_array3(field: &'static str, values: &Array3<f64>) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_xmul_dat(field, "all values must be finite"))
    }
}

fn validate_finite_array3_view(field: &'static str, values: ArrayView3<'_, f64>) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_xmul_dat(field, "all values must be finite"))
    }
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::XmulDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value = parse_f64_value(token).map_err(|_| IoError::XmulDatParse {
        field,
        line,
        token: token.to_string(),
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IoError::InvalidXmulDat {
            field,
            message: format!("line {line} value must be finite"),
        })
    }
}

fn parse_f64_value(token: &str) -> std::result::Result<f64, std::num::ParseFloatError> {
    token.replace(['D', 'd'], "E").parse::<f64>()
}

fn shape_error(field: &'static str, actual: Vec<usize>, expected: Vec<usize>) -> IoError {
    IoError::XmulDatShape {
        field,
        actual,
        expected,
    }
}

fn invalid_xmul_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidXmulDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xmul_decomposition_table() -> Result<()> {
        let data = parse_xmul_dat(XMUL_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.max_decomposition_channel, 2);
        assert_eq!(data.channel_count(), 3);
        assert_eq!(data.photon_energy_ev[0], 11_104.524);
        assert_eq!(data.wave_number[1], -1.200);
        assert_eq!(data.total_single_electron[0], 0.203e-5);
        assert_eq!(data.channel_background[[0, 1]], 0.173e-5);
        assert_eq!(data.normalized_fine_structure[[0, 0, 0]], 0.173);
        assert_eq!(data.normalized_fine_structure[[0, 1, 1]], -0.647e-1);
        assert_eq!(data.normalized_fine_structure[[1, 2, 2]], 0.547e-3);
        Ok(())
    }

    #[test]
    fn roundtrips_xmul_text() -> Result<()> {
        let data = parse_xmul_dat(XMUL_DAT)?;
        let rendered = xmul_dat_string(&data)?;
        assert_eq!(parse_xmul_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let data =
            parse_xmul_dat("# and ldecmx=     0\n  1.0D+0  2.0D+0  3.0D+0  4.0D+0  5.0D+0\n")?;
        assert_eq!(data.channel_count(), 1);
        assert_eq!(data.normalized_fine_structure[[0, 0, 0]], 5.0);
        Ok(())
    }

    #[test]
    fn builds_xmul_from_nrixs_decomposition_rows() -> Result<()> {
        let photon_energy_ev = Array1::from_vec(vec![11_104.5, 11_105.5]);
        let wave_number = Array1::from_vec(vec![-1.3, -1.2]);
        let channel_background =
            Array2::from_shape_vec((2, 2), vec![0.25, 1.75, 0.50, 2.25]).expect("test shape");
        let normalized_fine_structure =
            Array3::from_shape_vec((2, 2, 2), vec![0.1, 0.2, 0.3, 0.4, -0.2, -0.1, 0.6, 0.7])
                .expect("test shape");

        let data = xmul_dat_from_nrixs_decomposition(XmulDatFromNrixsDecompositionInput {
            header_lines: &[],
            max_decomposition_channel: 1,
            photon_energy_ev: photon_energy_ev.view(),
            wave_number: wave_number.view(),
            channel_background: channel_background.view(),
            normalized_fine_structure: normalized_fine_structure.view(),
        })?;

        assert_eq!(data.header_lines.len(), 3);
        assert!(data.header_lines[2].contains("ldecmx="));
        assert_eq!(data.max_decomposition_channel, 1);
        assert_eq!(data.total_single_electron[0], 2.0);
        assert_eq!(data.total_single_electron[1], 2.75);
        assert_eq!(data.channel_background, channel_background);
        assert_eq!(data.normalized_fine_structure, normalized_fine_structure);

        let parsed = parse_xmul_dat(&xmul_dat_string(&data)?)?;
        assert_eq!(parsed.point_count(), data.point_count());
        assert_eq!(parsed.channel_count(), data.channel_count());
        assert_eq!(
            parsed.max_decomposition_channel,
            data.max_decomposition_channel
        );
        Ok(())
    }

    #[test]
    fn xmul_from_nrixs_decomposition_preserves_matching_header() -> Result<()> {
        let header_lines = vec![
            "# custom NRIXS output".to_string(),
            "# and ldecmx=     1".to_string(),
        ];
        let photon_energy_ev = Array1::from_vec(vec![11_104.5]);
        let wave_number = Array1::from_vec(vec![-1.3]);
        let channel_background =
            Array2::from_shape_vec((1, 2), vec![0.25, 1.75]).expect("test shape");
        let normalized_fine_structure =
            Array3::from_shape_vec((1, 2, 2), vec![0.1, 0.2, 0.3, 0.4]).expect("test shape");

        let data = xmul_dat_from_nrixs_decomposition(XmulDatFromNrixsDecompositionInput {
            header_lines: &header_lines,
            max_decomposition_channel: 1,
            photon_energy_ev: photon_energy_ev.view(),
            wave_number: wave_number.view(),
            channel_background: channel_background.view(),
            normalized_fine_structure: normalized_fine_structure.view(),
        })?;

        assert_eq!(data.header_lines, header_lines);
        assert_eq!(
            parse_xmul_dat(&xmul_dat_string(&data)?)?.header_lines,
            header_lines
        );
        Ok(())
    }

    #[test]
    fn xmul_from_nrixs_decomposition_rejects_bad_shape_and_header() {
        let photon_energy_ev = Array1::from_vec(vec![11_104.5]);
        let wave_number = Array1::from_vec(vec![-1.3]);
        let channel_background = Array2::from_shape_vec((1, 1), vec![0.25]).expect("test shape");
        let normalized_fine_structure =
            Array3::from_shape_vec((1, 2, 2), vec![0.1, 0.2, 0.3, 0.4]).expect("test shape");

        let error = xmul_dat_from_nrixs_decomposition(XmulDatFromNrixsDecompositionInput {
            header_lines: &[],
            max_decomposition_channel: 1,
            photon_energy_ev: photon_energy_ev.view(),
            wave_number: wave_number.view(),
            channel_background: channel_background.view(),
            normalized_fine_structure: normalized_fine_structure.view(),
        })
        .expect_err("channel background must match ldecmx");

        assert!(matches!(
            error,
            IoError::XmulDatShape {
                field: "channel_background",
                actual,
                expected,
            } if actual == vec![1, 1] && expected == vec![1, 2]
        ));

        let channel_background =
            Array2::from_shape_vec((1, 2), vec![0.25, 1.75]).expect("test shape");
        let header_lines = vec!["# and ldecmx=     0".to_string()];
        let error = xmul_dat_from_nrixs_decomposition(XmulDatFromNrixsDecompositionInput {
            header_lines: &header_lines,
            max_decomposition_channel: 1,
            photon_energy_ev: photon_energy_ev.view(),
            wave_number: wave_number.view(),
            channel_background: channel_background.view(),
            normalized_fine_structure: normalized_fine_structure.view(),
        })
        .expect_err("header ldecmx must match the requested decomposition channels");

        assert!(matches!(
            error,
            IoError::InvalidXmulDat {
                field: "ldecmx",
                ..
            }
        ));
    }

    #[test]
    fn rejects_bad_xmul_inputs() {
        assert!(parse_xmul_dat("# no ldecmx\n1 2 3 4 5\n").is_err());
        assert!(parse_xmul_dat("# and ldecmx=     0\n").is_err());
        assert!(parse_xmul_dat("# and ldecmx=     1\n1 2 3 4 5\n").is_err());
        assert!(parse_xmul_dat("# and ldecmx=     0\n1 2 NaN 4 5\n").is_err());
        assert!(parse_xmul_dat("# and ldecmx= nope\n1 2 3 4 5\n").is_err());
    }

    const XMUL_DAT: &str = r#"#  Decomposition of S(q,w) for a single electron
#  omega    k   S^0(qw)  S_{l=0,...,ldecmx}^0(qw)       chi^q_{l=0,..ldecmx,l^*=0,...,ldecmx}
# and ldecmx=     2
   11104.524     -1.300  0.203E-05  0.264E-06  0.173E-05  0.358E-07  0.173E+00  0.000E+00  0.000E+00  0.000E+00 -0.647E-01  0.000E+00  0.000E+00  0.000E+00  0.471E-03
   11105.476     -1.200  0.241E-05  0.307E-06  0.206E-05  0.394E-07  0.191E+00  0.000E+00  0.000E+00  0.000E+00 -0.830E-01  0.000E+00  0.000E+00  0.000E+00  0.547E-03
"#;
}
